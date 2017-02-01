
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
			this cell index as referenced in :class:`Distributed`.

		translocations (array-like):
			translocations as a matrix of variations of coordinate and time with as many 
			columns as dimensions.

		center (array-like):
			cell center coordinates.

		area (array-like, unused):
			vertices of the polygonal extent of the cell, if available.

		dt (array-like, getter-only):
			translocation durations.

		dxy (array-like, getter-only):
			translocation changes in coordinate.

		grad (array of floats):
			local gradient.

		rot (array of floats):
			local rotational.

		flags (array of bools, unused):
			flags as boolean masks.

		time_col (int or string, lazy):
			column index for time.

		space_cols (list of ints or strings, lazy):
			column indices for coordinates.

		cache (any):
			depending on the inference approach and objective, caching an intermediate
			result may avoid repeating many times a same computation. Usage of this cache
			is totally free and comes without support for concurrency.

	"""
	__slots__ = ['index', 'translocations', 'center', 'area', 'grad', 'rot', 'flags', \
		'_time_col', '_space_cols', 'cache']
	__lazy__ = ['time_col', 'space_cols']

	def __init__(self, index, translocations, center=None, area=None, grad=None, rot=None, \
		flags=None):
		Lazy.__init__(self)
		self.index = index
		self.translocations = translocations
		self.center = center
		self.area = area
		self.grad = grad
		self.rot  = rot
		self.flags = flags
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

		cells (dict of :class:`Cell`):
			dictionnary (key: cell index) of cells.

		adjacency (:class:`~scipy.sparse.csr_matrix`):
			adjacency matrix for the cells.

		infered (:class:`~pandas.DataFrame`):
			infered dynamic parameters.

	"""
	def __init__(self, cells={}, adjacency=None, infered=None):
		if isinstance(cells, CellStats):
			# simplify the adjacency matrix
			if cells.tesselation.adjacency_label is None:
				self.adjacency = cells.tesselation.cell_adjacency.tocsr()
			else:
				self.adjacency = cells.tesselation.cell_adjacency.tocoo()
				ok = 0 < cells.tesselation.adjacency_label[self.adjacency.data]
				row, col = self.adjacency.row[ok], self.adjacency.col[ok]
				data = np.ones(np.count_nonzero(ok), dtype=bool)
				self.adjacency = sparse.csr_matrix((data, (row, col)), \
					shape=self.adjacency.shape)
			# build each cell
			ncells = self.adjacency.shape[0]
			transloc = cells.points.diff()
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
			if sparse.issparse(cells.cell_index):
				cell_index = cells.cell_index.tocoo()
				row, col, data = cell_index.row, cell_index.col, cell_index.data
				cell_index = sparse.csr_matrix((data, (col, row)), shape=cell_index.shape)
			self.cells = {}
			for j in range(ncells): # for each cell
				# get the translocations
				if isinstance(cells.cell_index, np.ndarray):
					i = cells.cell_index == j
				elif isinstance(cells.cell_index, tuple):
					i = cells.cell_index[0][cells.cell_index[1] == j]
				else: # sparse matrix
					i = cell_index[j].indices
				if isinstance(cells.points, pd.DataFrame):
					if i.dtype is bool:
						points = cells.points.loc[i]
					else:
						points = cells.points.iloc[i]
				else:
					points = cells.points[i]
				transloc = points.diff()
				if isstructured(points):
					transloc = transloc[transloc[trajectory_col]==0][coord_cols]
				else:
					transloc = transloc[transloc[:,trajectory_col]==0,coord_cols]
				if transloc.size:
					self.cells[j] = Cell(j, transloc, cells.tesselation.cell_centers[j])
		else:
			self.cells = cells
			self.adjacency = adjacency
		self.infered = infered


