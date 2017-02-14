
from inferencemap.core import *
from inferencemap.tesselation import CellStats, Voronoi, KMeansMesh
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from copy import copy
try:
	from collections import ChainMap
except ImportError:
	from chainmap import ChainMap



class Local(Lazy):
	"""
	Spatially local subset of elements (e.g. translocations). Abstract class.

	Attributes:

		index (int):
			this cell's index as referenced in :class:`Distributed`.

		terminal (bool, ro property):
			does data contain terminal elements?

		data (collection of terminal elements or :class:`Local`):
			elements, either terminal or not.

		center (array-like):
			cell center coordinates.

		area (array-like):
			difference vectors from this cell's center to adjacent centers. Useful as 
			mixing coefficients for summing the multiple local gradients of a scalar 
			dynamic parameter.

		dim (int, property):
			dimension of the terminal elements.

		tcount (int, property):
			total number of terminal elements (e.g. translocations).

		ccount (int, ro property):
			number of elements at this level (e.g. cells or - if terminal - translocations).

	"""
	__slots__ = ['index', 'data', 'center', 'area']
	__lazy__  = []

	def __init__(self, index, data, center=None, area=None):
		Lazy.__init__(self)
		self.index = index
		self.data = data
		self.center = center
		self.area = area

	@property
	def terminal(self):
		raise NotImplementedError('abstract method')

	@property
	def dim(self):
		raise NotImplementedError('abstract method')

	@dim.setter
	def dim(self, d):
		self.__lazyassert__(d, 'data')

	@property
	def tcount(self):
		raise NotImplementedError('abstract method')

	@tcount.setter
	def tcount(self, c):
		self.__lazyassert__(c, 'data')

	@property
	def ccount(self):
		raise NotImplementedError('abstract method')

	@ccount.setter
	def ccount(self, c):
		self.__lazyassert__(c, 'data')

	def flatten(self):
		raise NotImplementedError('abstract method')



class Distributed(Local):
	"""
	Attributes:

		dim (int, ro property):
			dimension of the terminal elements.

		tcount (int, property):
			total number of terminal elements.

		ccount (int):
			total number of cells at the current scale.

		cells (dict, rw property for :attr:`data`):
			dictionnary cell indices as keys and :class:`Local` as elements.

		adjacency (:class:`~scipy.sparse.csr_matrix`):
			cell adjacency matrix.

		degree (array of ints, ro lazy property):
			number of adjacent cells.

		infered (array-like or None):
			infered dynamic parameters at the current scale, if available.

	"""
	__slots__ = Local.__slots__ + ['_adjacency', '_degree', '_terminal', '_tcount', '_dim', 'infered']
	__lazy__  = Local.__lazy__  + ['degree', 'terminal', 'tcount', 'dim']

	def __init__(self, cells, adjacency, index=None, center=None, area=None):
		if not (isinstance(cells, dict) and all([ isinstance(cell, Local) for cell in cells.values() ])):
			raise TypeError('`cells` argument is not a `dict` of `Local`s')
		Local.__init__(self, index, cells, center, area)
		self._adjacency = adjacency
		self.infered = None

	@property
	def cells(self):
		return self.data

	@cells.setter
	def cells(self, cells):
		if not (isinstance(cells, dict) and all([ isinstance(cell, Local) for cell in cells.values() ])):
			raise TypeError('`cells` argument is not a `dict` of `Local`s')
		if self.ccount < max(cells.keys()):
			self.adjacency = None
		self.data = cells
		self._terminal = None

	@property
	def terminal(self):
		if self._terminal is None:
			self._terminal = all([ cell.terminal for cell in self.data.values() ])
		return self._terminal

	@terminal.setter
	def terminal(self, t):
		ro_property_assert(self, t, 'data')

	@property
	def dim(self):
		if self._dim is None:
			self._dim = self.data.values().next().dim
		return self._dim

	@dim.setter
	def dim(self, d):
		ro_property_assert(self, d, 'data')

	@property
	def tcount(self):
		if self._tcount is None:
			self._tcount = sum([ cell.tcount for cell in self.data.values() ])
		return self._tcount

	@tcount.setter
	def tcount(self, c):
		self.__lazysetter__(c)

	@property
	def ccount(self):
		return self.adjacency.shape[0]

	@ccount.setter
	def ccount(self, c):
		self.__lazyassert__(c, 'data')

	@property
	def adjacency(self):
		return self._adjacency

	@adjacency.setter
	def adjacency(self, a):
		self._adjacency = a.tocsr()
		self._degree = None

	@property
	def degree(self):
		if self._degree is None:
			self._degree = np.diff(self._adjacency.indptr)
		return self._degree

	@degree.setter
	def degree(self, d):
		self.__lazyassert__(d, 'adjacency')

	def flatten(self):
		new = copy(self)
		new.data = {i: ChainMap(cell.values()) if isinstance(cell, dict) else cell \
			for i, cell in self.data.items() }
		return new

	def group(self, max_cell_count=None, cell_centers=None):
		new = copy(self)
		if max_cell_count or cell_centers is not None:
			points = np.array([ cell.center for cell in self.data.values() ])
			if cell_centers is None:
				avg_probability = float(max_cell_count) / float(points.shape[0])
				from inferencemap.tesselation import KMeansMesh
				grid = KMeansMesh(avg_probability=avg_probability)
				grid.tesselate(points)
			else:
				grid = Voronoi()
				grid.cell_centers = cell_centers
			I = grid.cellIndex(points)
			A = grid.cell_adjacency
			new.adjacency = sparse.csr_matrix((np.ones_like(A.data, dtype=bool), \
				A.indices, A.indptr), shape=A.shape)
			new.data = {}
			for j in I:
				J = I == j
				A = self.adjacency[J,:].tocsc()[:,J].tocsr()
				C = grid.cell_centers[j]
				D = {k: self.data[k] for k in J.nonzero()[0]}
				R = grid.cell_centers[self.adjacency[j].indices] - C
				new.data[j] = type(self)(D, A, index=j, center=C, area=R)
			new._terminal = False
			# _tcount is not supposed to change
		else:
			raise KeyError('`group` expects more input arguments')
		return new



class Cell(Local):
	"""
	Spatially constrained subset of translocations with associated intermediate calculations.

	Attributes:

		index (int):
			this cell's index as granted in :class:`Distributed`'s `cells` dict.

		translocations (array-like, property):
			translocations as a matrix of variations of coordinate and time with as many 
			columns as dimensions. Alias for :attr:`~Local.data`.

		center (array-like):
			cell center coordinates.

		area (array-like):
			difference vectors from this cell's center to adjacent centers. Useful as 
			mixing coefficients for summing the multiple local gradients of a scalar 
			dynamic parameter.

		dt (array-like, ro property):
			translocation durations.

		dxy (array-like, ro property):
			translocation changes in coordinate.

		time_col (int or string, lazy):
			column index for time.

		space_cols (list of ints or strings, lazy):
			column indices for coordinates.

		tcount (int):
			number of translocations.

		cache (any):
			depending on the inference approach and objective, caching an intermediate
			result may avoid repeating many times a same computation. Usage of this cache
			is totally free and comes without support for concurrency.

	"""
	__slots__ = Local.__slots__ + ['_time_col', '_space_cols', 'cache']
	__lazy__  = Local.__lazy__  + ['time_col', 'space_cols']

	def __init__(self, index, translocations, center=None, area=None):
		if not (isinstance(translocations, np.ndarray) or isinstance(translocations, pd.DataFrame)):
			raise TypeError('unsupported translocation type `{}`'.format(type(translocations)))
		Local.__init__(self, index, translocations, center, area)
		#self._tcount = translocations.shape[0]
		self._time_col = None
		self._space_cols = None
		self.cache = None
		#self.translocations = (self.dxy, self.dt)

	@property
	def translocations(self):
		return self.data

	@translocations.setter
	def translocations(self, tr):
		self.data = tr

	@property
	def time_col(self):
		if self._time_col is None:
			if isstructured(self.translocations):
				self._time_col = 't'
			else:
				self._time_col = 0
		return self._time_col

	@time_col.setter
	def time_col(self, col):
		# space_cols is left unchanged
		self.__lazysetter__(col)

	@property
	def space_cols(self):
		if self._space_cols is None:
			if isstructured(self.translocations):
				self._space_cols = columns(self.translocations)
				if isinstance(self._space_cols, pd.Index):
					self._space_cols = self._space_cols.drop(self.time_col)
				else:
					self._space_cols.remove(self.time_col)
			else:
				if self.time_col == 0:
					self._space_cols = np.arange(1, self.translocations.shape[1])
				else:
					self._space_cols = np.ones(self.translocations.shape[1], \
						dtype=bool)
					self._space_cols[self.time_col] = False
					self._space_cols, = self._space_cols.nonzero()
		return self._space_cols

	@space_cols.setter
	def space_cols(self, cols):
		# time_col is left unchanged
		self.__lazysetter__(cols)

	@property
	def dt(self):
		if not isinstance(self.translocations, tuple):
			self.translocations = (self._dxy(), self._dt())
		return self.translocations[1]

	def _dt(self):
		if isstructured(self.translocations):
			return np.asarray(self.translocations[self.time_col])
		else:
			return np.asarray(self.translocations[:,self.time_col])

	@property
	def dxy(self):
		if not isinstance(self.translocations, tuple):
			self.translocations = (self._dxy(), self._dt())
		return self.translocations[0]

	def _dxy(self):
		if isstructured(self.translocations):
			return np.asarray(self.translocations[self.space_cols])
		else:
			return np.asarray(self.translocations[:,self.space_cols])

	@property
	def terminal(self):
		return True

	@property
	def dim(self):
		return self.dxy.shape[1]

	@dim.setter
	def dim(self, d):
		self.__lazyassert__(d, 'data')

	@property
	def tcount(self):
		return self.dxy.shape[0]

	@tcount.setter
	def tcount(self, c):
		self.__lazyassert__(c, 'data')



def distributed(cells={}, adjacency=None, infered=None):
	if isinstance(cells, CellStats):
		# simplify the adjacency matrix
		if cells.tesselation.adjacency_label is None:
			_adjacency = cells.tesselation.cell_adjacency.tocsr()
		else:
			_adjacency = cells.tesselation.cell_adjacency.tocoo()
			ok = 0 < cells.tesselation.adjacency_label[_adjacency.data]
			row, col = _adjacency.row[ok], _adjacency.col[ok]
			data = np.ones(np.count_nonzero(ok)) # the values do not matter
			_adjacency = sparse.csr_matrix((data, (row, col)), \
				shape=_adjacency.shape)
		# reweight each row i as 1/n_i where n_i is the degree of cell i
		n = np.diff(_adjacency.indptr)
		_adjacency.data[...] = np.repeat(1.0 / np.maximum(1, n), n)
		# extract translocations and define point-translocation matching
		ncells = _adjacency.shape[0]
		if isstructured(cells.points):
			trajectory_col = 'n'
			coord_cols = columns(cells.points)
			if isinstance(coord_cols, pd.Index):
				coord_cols = coord_cols.drop(trajectory_col)
			else:
				coord_cols.remove(trajectory_col)
		else:
			trajectory_col = 0
			coord_cols = np.arange(1, cells.points.shape[1])
		if isinstance(cells.points, pd.DataFrame):
			translocations = cells.points.diff().ix[1:]
		else:
			translocations = np.diff(cells.points, axis=0)
		if isstructured(translocations):
			ok = translocations[trajectory_col]==0
			translocations = translocations[ok][coord_cols]
		else:
			ok = translocations[:,trajectory_col]==0
			translocations = translocations[ok][:,coord_cols]
		#ok, = ok.nonzero()
		#ok += 1 # will take into account only second location
		ok = np.concatenate(([False], ok))
		# time and space columns in translocations array
		if isstructured(translocations):
			time_col = 't'
			space_cols = columns(translocations)
			if isinstance(space_cols, pd.Index):
				space_cols = space_cols.drop(time_col)
			else:
				space_cols.remove(time_col)
		else:
			time_col = translocations.shape[1]
			if time_col == translocations.shape[1]:
				space_cols = np.arange(time_col)
			else:
				space_cols = np.ones(translocations.shape[1], dtype=bool)
				space_cols[time_col] = False
				space_cols, = space_cols.nonzero()
		# build every cells
		if sparse.issparse(cells.cell_index):
			cell_index = cells.cell_index.tocsr()
			#row, col, data = cell_index.row, cell_index.col, cell_index.data
			#cell_index = sparse.csr_matrix((data, (col, row)), shape=cell_index.shape)
		_cells = {}
		for j in range(ncells): # for each cell
			# get the translocations with second location in this cell
			if isinstance(cells.cell_index, np.ndarray):
				i = cells.cell_index[ok] == j
				i = i[1:] # translocations instead of locations
			elif isinstance(cells.cell_index, tuple):
				i = cells.cell_index[0][cells.cell_index[1] == j]
				i = i[ok[i]] - 1 # translocations instead of locations
			else: # sparse matrix
				i = cell_index[j].indices # and not cells.cell_index!
				i = i[ok[i]] - 1 # translocations instead of locations
			if isinstance(translocations, pd.DataFrame):
				if i.dtype is bool:
					transloc = translocations.loc[i]
				else:
					transloc = translocations.iloc[i]
			else:
				transloc = translocations[i]
			if transloc.size:
				# make cell object
				center = cells.tesselation.cell_centers[j]
				adj = _adjacency[j].indices
				area = cells.tesselation.cell_centers[adj]
				_cells[j] = Cell(j, transloc, center, area - center)
				_cells[j].time_col = time_col
				_cells[j].space_cols = space_cols
		self = Distributed(_cells, _adjacency)
		#self.tcount, self.dim = cells.points.shape
		#if isstructured(self.translocations):
		#	self.dt = np.asarray(cells.points[time_col])
		#	self.dxy = np.asarray(cells.points[space_cols])
		#else:
		#	self.dt = np.asarray(cells.points[:,time_col])
		#	self.dxy = np.asarray(cells.points[:,space_cols])
	else:
		self = Distributed(cells, adjacency)
		#self.tcount = sum([ cells[c].translocations.shape[0] for c in cells ])
		#self.dim = cells[list(cells.keys())[0]].center.size
		#self.dt = dt
		#self.dxy = dxy
	self.infered = infered
	return self


