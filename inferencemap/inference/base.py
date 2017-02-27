
from inferencemap.core import *
from inferencemap.tesselation import CellStats, Voronoi, KMeansMesh
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from copy import copy
from collections import OrderedDict
from multiprocessing import Pool, Lock
import six
from functools import partial



class Local(Lazy):
	"""
	Spatially local subset of elements (e.g. translocations). Abstract class.

	Attributes:

		index (int):
			this cell's index as referenced in :class:`Distributed`.

		data (collection of terminal elements or :class:`Local`):
			elements, either terminal or not.

		center (array-like):
			cell center coordinates.

		span (array-like):
			difference vectors from this cell's center to adjacent centers. Useful as 
			mixing coefficients for summing the multiple local gradients of a scalar 
			dynamic parameter.

		dim (int, property):
			dimension of the terminal elements.

		tcount (int, property):
			total number of terminal elements (e.g. translocations).

	"""
	__slots__ = ['index', 'data', 'center', 'span']
	__lazy__  = []

	def __init__(self, index, data, center=None, span=None):
		Lazy.__init__(self)
		self.index = index
		self.data = data
		self.center = center
		self.span = span

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



class Distributed(Local):
	"""
	Attributes:

		dim (int, ro property):
			dimension of the terminal elements.

		tcount (int, property):
			total number of terminal elements. Duplicates are ignored.

		ccount (int, property):
			total number of cells (not distributed). Duplicates are ignored.

		cells (list or OrderedDict, rw property for :attr:`data`):
			collection of :class:`Local`s. Indices may not match with the global 
			:attr:`~Local.index` attribute of the elements, but match with attributes 
			:attr:`central`, :attr:`adjacency` and :attr:`degree`.

		reverse (dict of ints, ro lazy property):
			get "local" indices from global ones.

		central (array of bools):
			margin cells are not central.

		adjacency (:class:`~scipy.sparse.csr_matrix`):
			cell adjacency matrix. Row and column indices are to be mapped with `indices`.

		degree (array of ints, ro lazy property):
			number of adjacent cells.

		boundary (array or list of arrays):
			polygons as vertex indices.

	"""
	__slots__ = Local.__slots__ + ['_reverse', '_adjacency', 'central', '_degree', \
		'_ccount', '_tcount', '_dim', 'boundary']
	__lazy__  = Local.__lazy__  + ['reverse', 'degree', 'ccount', 'tcount', 'dim']

	def __init__(self, cells, adjacency, index=None, center=None, span=None, central=None, \
		boundary=None):
		Local.__init__(self, index, OrderedDict(), center, span)
		self.cells = cells # let's `cells` setter perform the necessary checks 
		self.adjacency = adjacency
		self.central = central
		self.boundary = boundary

	@property
	def cells(self):
		return self.data

	@cells.setter
	def cells(self, cells):
		celltype = type(self.cells)
		assert(celltype is dict or celltype is OrderedDict)
		if not isinstance(cells, celltype):
			if isinstance(cells, dict):
				cells = celltype(sorted(cells.items(), key=lambda t: t[0]))
			elif isinstance(cells, list):
				cells = celltype(sorted(enumerate(cells), key=lambda t: t[0]))
			else:
				raise TypeError('`cells` argument is not a dictionnary (`dict` or `OrderedDict`)')
		if not all([ isinstance(cell, Local) for cell in cells.values() ]):
			raise TypeError('`cells` argument is not a dictionnary of `Local`s')
		#try:
		#	if self.ccount == len(cells): #.keys().reversed().next(): # max
		#		self.adjacency = None
		#		self.central = None
		#except:
		#	pass
		self.reverse = None
		self.ccount = None
		self.data = cells

	@property
	def indices(self):
		return np.array([ cell.index for cell in self.cells.values() ])

	@property
	def reverse(self):
		if self._reverse is None:
			self._reverse = {cell.index: i for i, cell in self.cells.items()}
		return self._reverse

	@reverse.setter
	def reverse(self, r): # ro
		self.__lazyassert__(r, 'cells')

	@property
	def dim(self):
		if self._dim is None:
			if self.center is None:
				self._dim = list(self.cells.values())[0].dim
			else:
				self._dim = self.center.size
		return self._dim

	@dim.setter
	def dim(self, d): # ro
		self.__lazyassert__(d, 'cells')

	@property
	def tcount(self):
		if self._tcount is None:
			if self.central is None:
				self._tcount = sum([ cell.tcount \
					for i, cell in self.cells.items() if self.central[i] ])
			else:
				self._tcount = sum([ cell.tcount for cell in self.cells.values() ])
		return self._tcount

	@tcount.setter
	def tcount(self, c):
		# write access allowed for performance issues, but `c` should equal self.tcount
		self.__lazysetter__(c)

	@property
	def ccount(self):
		#return self.adjacency.shape[0] # or len(self.cells)
		if self._ccount is None:
			self._ccount = sum([ cell.ccount if isinstance(cell, Distributed) else 1 \
				for cell in self.cells.values() ])
		return self._ccount

	@ccount.setter
	def ccount(self, c): # rw for performance issues, but `c` should equal self.ccount
		self.__lazysetter__(c)

	@property
	def adjacency(self):
		return self._adjacency

	@adjacency.setter
	def adjacency(self, a):
		if a is not None:
			a = a.tocsr()
		self._adjacency = a
		self._degree = None # `degree` is ro, hence set `_degree` instead

	@property
	def degree(self):
		if self._degree is None:
			self._degree = np.diff(self._adjacency.indptr)
		return self._degree

	@degree.setter
	def degree(self, d): # ro
		self.__lazyassert__(d, 'adjacency')

	def grad(self, i, X, index_map=None):
		cell = self.cells[i]
		adjacent = self.adjacency[i].indices
		if index_map is not None:
			i = index_map[i]
			adjacent = index_map[adjacent]
			ok = 0 <= adjacent
			if not np.any(ok):
				return None
			adjacent = adjacent[ok]
		if not isinstance(cell.cache, dict):
			cell.cache = dict(vanders=None)
		if cell.cache.get('vanders', None) is None:
			if index_map is None:
				span = cell.span
			else:
				span = cell.span[ok]
			cell.cache['vanders'] = [ np.vander(col, 3)[...,:2] for col in span.T ]
		dX = X[adjacent] - X[i]
		try:
			ok = np.logical_not(dX.mask)
			if not np.any(ok):
				#warn('Distributed.grad: all the points are masked', RuntimeWarning)
				return None
		except AttributeError:
			ok = slice(dX.size)
		gradX = np.array([ np.linalg.lstsq(vander[ok], dX[ok])[0][1] \
			for vander in cell.cache['vanders'] ])
		return gradX

	def gradSum(self, i, index_map=None):
		cell = self.cells[i]
		if not isinstance(cell.cache, dict):
			cell.cache = dict(area=None)
		area = cell.cache.get('area', None)
		if area is None:
			if index_map is None:
				area = cell.span
			else:
				ok = 0 <= index_map[self.adjacency[i].indices]
				if not np.any(ok):
					area = False
					cell.cache['area'] = area
					return area
				area = cell.span[ok]
			# we want prod_i(area_i) = area_tot
			# just a random approximation:
			area = np.sqrt(np.mean(area * area, axis=0))
			cell.cache['area'] = area
		return area

	def flatten(self):
		def concat(arrays):
			if isinstance(arrays[0], tuple):
				raise NotImplementedError
			elif isinstance(arrays[0], pd.DataFrame):
				return pd.concat(arrays, axis=0)
			else:
				return np.stack(arrays, axis=0)
		new = copy(self)
		new.cells = {i: Cell(i, concat([cell.data for cell in dist.cells.values()]), \
				dist.center, dist.span) \
			if isinstance(dist, Distributed) else dist \
			for i, dist in self.cells.items() }
		return new

	def group(self, ngroups=None, max_cell_count=None, cell_centers=None, \
		adjacency_margin=1):
		new = copy(self)
		if ngroups or max_cell_count or cell_centers is not None:
			points = np.full((self.adjacency.shape[0], self.dim), np.inf)
			ok = np.zeros(points.shape[0], dtype=bool)
			for i in self.cells:
				points[i] = self.cells[i].center
				ok[i] = True
			if cell_centers is None:
				avg_probability = 1.0
				if ngroups:
					avg_probability = min(1.0 / float(ngroups), avg_probability)
				if max_cell_count:
					avg_probability = min(float(max_cell_count) / \
						float(points.shape[0]), avg_probability)
				from inferencemap.tesselation import KMeansMesh
				grid = KMeansMesh(avg_probability=avg_probability)
				grid.tesselate(points[ok])
			else:
				grid = Voronoi()
				grid.cell_centers = cell_centers
			I = np.full(ok.size, -1, dtype=int)
			I[ok] = grid.cellIndex(points[ok], min_cell_size=1)
			#if not np.all(ok):
			#	print(ok.nonzero()[0])
			A = grid.cell_adjacency
			new.adjacency = sparse.csr_matrix((np.ones_like(A.data, dtype=bool), \
				A.indices, A.indptr), shape=A.shape) # macro-cell adjacency matrix
			J = np.unique(I)
			J = J[0 <= J]
			new.data = type(self.cells)()
			for j in J: # for each macro-cell
				K = I == j # find corresponding cells
				assert np.any(K)
				if 0 < adjacency_margin:
					L = np.copy(K)
					for k in range(adjacency_margin):
						# add adjacent cells for future gradient calculations
						K[self.adjacency[K,:].indices] = True
					L = L[K]
				A = self.adjacency[K,:].tocsc()[:,K].tocsr() # point adjacency matrix
				C = grid.cell_centers[j]
				D = OrderedDict([ (i, self.cells[k]) \
					for i, k in enumerate(K.nonzero()[0]) if k in self.cells ])
				for i in D:
					adj = A[i].indices
					if 0 < D[i].tcount and adj.size:
						span = np.stack([ D[k].center for k in adj ], axis=0)
					else:
						span = np.empty((0, D[i].center.size), \
							dtype=D[i].center.dtype)
					if span.shape[0] < D[i].span.shape[0]:
						D[i] = copy(D[i])
						D[i].span = span - D[i].center
				R = grid.cell_centers[new.adjacency[j].indices] - C
				new.cells[j] = type(self)(D, A, index=j, center=C, span=R)
				if 0 < adjacency_margin:
					new.cells[j].central = L
				#assert 0 < new.cells[j].tcount # unfortunately will have to deal with
			new.ccount = self.ccount
			# _tcount is not supposed to change
		else:
			raise KeyError('`group` expects more input arguments')
		return new

	def run(self, function, *args, **kwargs):
		if all([ isinstance(cell, Distributed) for cell in self.cells.values() ]):
			worker_count = kwargs.pop('worker_count', None)
			# if worker_count is None, Pool will use multiprocessing.cpu_count()
			cells = [ cell for cell in self.cells.values() if 0 < cell.tcount ]
			pool = Pool(worker_count)
			fargs = (function, args, kwargs)
			if six.PY3:
				ys = pool.map(partial(__run__, fargs), cells)
			elif six.PY2:
				import itertools
				ys = pool.map(__run_star__, \
					itertools.izip(itertools.repeat(fargs), cells))
			ys = [ y for y in ys if y is not None ]
			if ys:
				result = pd.concat(ys, axis=0).sort_index()
			else:
				result = None
		else:
			result = function(self, *args, **kwargs)
		return result



def __run__(func, cell):
	function, args, kwargs = func
	x = cell.run(function, *args, **kwargs)
	if x is None:
		return None
	else:
		i = cell.indices
		if cell.central is not None:
			try:
				x = x.iloc[cell.central[x.index]]
			except IndexError as e:
				if cell.central.size < x.index.max():
					raise IndexError('dataframe indices do no match with group-relative cell indices (maybe are they global ones)')
				else:
					print(x.shape)
					print((cell.central.shape, cell.central.max()))
					print(x.index.max())
					raise e
			i = i[x.index]
			if x.shape[0] != i.shape[0]:
				raise IndexError('not as many indices as values')
		x.index = i
		return x

def __run_star__(args):
	return __run__(*args)



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

		span (array-like):
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

	def __init__(self, index, translocations, center=None, span=None):
		if not (isinstance(translocations, np.ndarray) or isinstance(translocations, pd.DataFrame)):
			raise TypeError('unsupported translocation type `{}`'.format(type(translocations)))
		Local.__init__(self, index, translocations, center, span)
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
	def dim(self):
		return self.center.size

	@dim.setter
	def dim(self, d): # ro
		self.__lazyassert__(d, 'translocations')

	@property
	def tcount(self):
		return self.dxy.shape[0]

	@tcount.setter
	def tcount(self, c): # ro
		self.__lazyassert__(c, 'translocations')



def distributed(cells=OrderedDict(), adjacency=None, new=Distributed):
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
		_cells = OrderedDict()
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
				transloc = translocations.iloc[i]
			else:
				transloc = translocations[i]
			center = cells.tesselation.cell_centers[j]
			adj = _adjacency[j].indices
			span = cells.tesselation.cell_centers[adj]
			#if transloc.size:
			# make cell object
			_cells[j] = Cell(j, transloc, center, span - center)
			_cells[j].time_col = time_col
			_cells[j].space_cols = space_cols
		#print(sum([ c.tcount == 0 for c in _cells.values() ]))
		self = new(_cells, _adjacency)
		self.tcount = cells.points.shape[0]
		#self.dim = cells.points.shape[1]
		#if isstructured(self.translocations):
		#	self.dt = np.asarray(cells.points[time_col])
		#	self.dxy = np.asarray(cells.points[space_cols])
		#else:
		#	self.dt = np.asarray(cells.points[:,time_col])
		#	self.dxy = np.asarray(cells.points[:,space_cols])
	else:
		self = new(cells, adjacency)
		#self.tcount = sum([ cells[c].translocations.shape[0] for c in cells ])
		#self.dim = cells[list(cells.keys())[0]].center.size
		#self.dt = dt
		#self.dxy = dxy
	self.ccount = self.adjacency.shape[0]
	return self


