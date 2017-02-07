
from inferencemap.core import *
from inferencemap.tesselation import CellStats
import numpy as np
import pandas as pd
import scipy.sparse as sparse


class Cell(Lazy):
	"""
	Spatially constrained subset of translocations with associated intermediate calculations.

	Attributes:

		index (int):
			this cell's index as referenced in :class:`Distributed`.

		translocations (array-like):
			translocations as a matrix of variations of coordinate and time with as many 
			columns as dimensions.

		center (array-like):
			cell center coordinates.

		area (array-like):
			difference vectors from this cell's center to adjacent centers. Useful as 
			mixing coefficients for summing the multiple local gradients of a scalar 
			dynamic parameter.

		dt (array-like, getter-only):
			translocation durations.

		dxy (array-like, getter-only):
			translocation changes in coordinate.

		time_col (int or string, lazy):
			column index for time.

		space_cols (list of ints or strings, lazy):
			column indices for coordinates.

..		flags (array of bools, unused):
			boolean flags for process control.

		cache (any):
			depending on the inference approach and objective, caching an intermediate
			result may avoid repeating many times a same computation. Usage of this cache
			is totally free and comes without support for concurrency.

	"""
	__slots__ = ['index', 'translocations', 'center', 'area', 'flags', \
		'_time_col', '_space_cols', 'cache']
	__lazy__ = ['time_col', 'space_cols']

	def __init__(self, index, translocations, center=None, area=None):
		Lazy.__init__(self)
		self.index = index
		self.translocations = translocations
		self.center = center
		self.area = area
		self._time_col = None
		self._space_cols = None
		self.cache = None

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
		if isstructured(self.translocations):
			return np.asarray(self.translocations[self.time_col])
		else:
			return np.asarray(self.translocations[:,self.time_col])

	@property
	def dxy(self):
		if isstructured(self.translocations):
			return np.asarray(self.translocations[self.space_cols])
		else:
			return np.asarray(self.translocations[:,self.space_cols])



class Distributed(object):
	"""
	Translocations and dynamic parameters distributed across a tesselation.

	Attributes:

		dim (int):
			dimension of the translocation coordinates.

		tcount (int):
			total number of translocations.

		ccount (int):
			total number of cells (including empty ones that do not appear in `cells`).

		cells (dict of :class:`Cell`):
			dictionnary (key: cell index) of cells.

		adjacency (:class:`~scipy.sparse.csr_matrix`):
			adjacency matrix for the cells, such that weight at row i and column j is
			1/n_i where n_i is the degree of cell i if j is connected with i, or 0.

		infered (:class:`~pandas.DataFrame`):
			infered dynamic parameters.

	"""
	def __init__(self, cells={}, adjacency=None, infered=None):#, dt=None, dxy=None):
		if isinstance(cells, CellStats):
			# simplify the adjacency matrix
			if cells.tesselation.adjacency_label is None:
				self.adjacency = cells.tesselation.cell_adjacency.tocsr()
			else:
				self.adjacency = cells.tesselation.cell_adjacency.tocoo()
				ok = 0 < cells.tesselation.adjacency_label[self.adjacency.data]
				row, col = self.adjacency.row[ok], self.adjacency.col[ok]
				data = np.ones(np.count_nonzero(ok)) # the values do not matter
				self.adjacency = sparse.csr_matrix((data, (row, col)), \
					shape=self.adjacency.shape)
			# reweight each row i as 1/n_i where n_i is the degree of cell i
			n = np.diff(self.adjacency.indptr)
			self.adjacency.data[...] = np.repeat(1.0 / np.maximum(1, n), n)
			# extract translocations and define point-translocation matching
			ncells = self.adjacency.shape[0]
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
			self.cells = {}
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
					adj = self.adjacency[j].indices
					area = cells.tesselation.cell_centers[adj]
					self.cells[j] = Cell(j, transloc, center, area - center)
					self.cells[j].time_col = time_col
					self.cells[j].space_cols = space_cols
			self.tcount, self.dim = cells.points.shape
			#if isstructured(self.translocations):
			#	self.dt = np.asarray(cells.points[time_col])
			#	self.dxy = np.asarray(cells.points[space_cols])
			#else:
			#	self.dt = np.asarray(cells.points[:,time_col])
			#	self.dxy = np.asarray(cells.points[:,space_cols])
		else:
			self.cells = cells
			self.adjacency = adjacency
			self.tcount = sum([ cells[c].translocations.shape[0] for c in cells ])
			self.dim = cells[list(cells.keys())[0]].center.size
			#self.dt = dt
			#self.dxy = dxy
		self.infered = infered

	@property
	def ccount(self):
		return self.adjacency.shape[0]


	def updateCell(self, parameter, cell, update, *args, **kwargs):
		"""
		Update a dynamic parameter at one or more cells.

		Arguments:

			parameter (string):
				name of the dynamic parameter.

			cell (int or `Cell` or array-like):
				cell index (or indices) which value to update.

			update (float or array-like or function):
				if function: takes the current parameter value(s) and return their
				new value(s). Datatype and shape should be let unchanged.
				if scalar or vector: new parameter value(s).

		"""
		if self.infered is None:
			self.infered = pd.DataFrame(data=np.zeros(self.adjacency.shape[0]), \
				columns=[parameter])
		elif parameter_name not in self.infered:
			self.infered[parameter] = np.zeros(self.adjacency.shape[0])
		if isinstance(cell, Cell):
			cell = cell.index
		if callable(value):
			self.infered[parameter][cell] = \
				update(self.infered[parameter][cell], *args, **kwargs)
		else:
			self.infered[parameter][cell] = update


