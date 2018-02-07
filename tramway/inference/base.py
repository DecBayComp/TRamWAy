# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import *
from tramway.tessellation import format_cell_index, nearest_cell
import tramway.tessellation as tessellation
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from copy import copy
from collections import OrderedDict
from multiprocessing import Pool, Lock
import six
from functools import partial
from warnings import warn



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
	"""
		boundary (array or list of arrays):
			polygons as vertex indices.

	"""
	__slots__ = ('index', 'data', 'center', 'span') #, 'boundary'

	def __init__(self, index, data, center=None, span=None): #, boundary=None
		Lazy.__init__(self)
		self.index = index
		self.data = data
		self.center = center
		self.span = span
		#self.boundary = boundary

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
			dimension of the terminal points.

		tcount (int, property):
			total number of terminal points. Duplicates are ignored.

		ccount (int, property):
			total number of terminal cells. Duplicates are ignored.

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

	"""
	__slots__ = ('_reverse', '_adjacency', 'central', '_degree', '_ccount', '_tcount', '_dim')
	__lazy__  = Local.__lazy__ + ('reverse', 'degree', 'ccount', 'tcount', 'dim')

	def __init__(self, cells, adjacency, index=None, center=None, span=None, central=None, \
		boundary=None):
		Local.__init__(self, index, OrderedDict(), center, span)
		self.cells = cells # let's `cells` setter perform the necessary checks 
		self.adjacency = adjacency
		self.central = central

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
		"""
		Local gradient.

		Arguments:

			i (int):
				cell index at which the gradient is evaluated.

			X (array):
				vector of a scalar measurement at every cell.

			index_map (array, optional):
				index map that converts cell indices to indices in X.

		Results:

			array:
				gradient as a matrix with as many rows as adjacent mesh points and 
				columns as dimensions.
		"""
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

	def grad_sum(self, i, index_map=None):
		"""
		Mixing matrix for the gradients at a given cell.

		Arguments:

			i (int):
				cell index.

			index_map (array):
				index mapping, useful to convert cell indices to positional indices in 
				an optimization array for example.

		Output:
			array:
				matrix that sums by dot product the adjacent components (e.g. gradient).

		"""
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

	def dx_dt(self, t, X):
		"""
		Time derivative.

		Arguments:

			t (float):
				time at which the gradient is evaluated.

			X (pandas.Series):
				time series with time `t` included.

		Results:

			float or array:
				time derivative of X at t.
		"""
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
		"""
		Make groups of cells.

		This builds up an extra hierarchical level. 
		For example if `self` is a `Distributed` of `Cell`, then this returns a `Distributed` 
		of `Distributed` (the groups) of `Cell`.

		Several grouping strategies are proposed.

		Arguments:

			ngroups (int):
				number of groups.

			max_cell_count (int):
				maximum number of cells per group.

			cell_centers (array-like):
				spatial centers of the groups. A cell is associated to the group which 
				center is the nearest one. If not provided, :meth:`group` will use 
				a k-means approach to positionning the centers.

			adjacency_margin (int):
				groups are dilated to the adjacent cells `adjacency_margin` times. 
				Defaults to 1.

		Returns:

			Distributed:
				a new object of the same type as `self` that contains other such
				objects as cells.

		"""
		new = copy(self)
		if ngroups or max_cell_count or cell_centers is not None:
			points = np.full((self.adjacency.shape[0], self.dim), np.inf)
			ok = np.zeros(points.shape[0], dtype=bool)
			for i in self.cells:
				points[i] = self.cells[i].center
				ok[i] = True
			if cell_centers is None:
				if max_cell_count == 1:
					grid = copy(self)
				else:
					avg_probability = 1.0
					if ngroups:
						avg_probability = min(1.0 / float(ngroups), avg_probability)
					if max_cell_count:
						avg_probability = min(float(max_cell_count) / \
							float(points.shape[0]), avg_probability)
					import tramway.tessellation.kmeans as kmeans
					grid = kmeans.KMeansMesh(avg_probability=avg_probability)
					try:
						grid.tessellate(points[ok])
					except ValueError:
						print(points)
						raise
			else:
				grid = tessellation.Voronoi()
				grid.cell_centers = cell_centers
			I = np.full(ok.size, -1, dtype=int)
			I[ok] = grid.cell_index(points[ok], min_location_count=1)
			#if not np.all(ok):
			#	print(ok.nonzero()[0])
			new.adjacency = grid.simplified_adjacency(format='csr') # macro-cell adjacency matrix
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
		"""
		Apply a function to the groups (:class:`Distributed`) of terminal cells.

		The results are merged into a single :class:`~pandas.DataFrame` array, 
		handling adjacency margins if any.

		Although this method was designed for `Distributed` of `Distributed`, its usage is 
		advised to call any function that returns a DataFrame with cell indices as indices.

		Multiples processes may be spawned.

		Arguments:

			function (callable):
				the function to be called on each terminal :class:`Distributed`. 
				Its first argument is the :class:`Distributed` object.
				It should return a :class:`~pandas.DataFrame`.

			args (list):
				positional arguments for `function` after the first one.

			kwargs (dict):
				keyword arguments for `function` from which are removed the ones below.

			worker_count (int):
				number of simultaneously working processing units.

		Returns:

			pandas.DataFrame:
				single merged array.

		"""
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

	# `dict` interface
	def __len__(self):
		return self.adjacency.shape[0]

	def __nonzero__(self):
		return nonzero(self.cells)

	def __getitem__(self, i):
		return self.cells[i]

	def __setitem__(self, i, cell):
		self.cells[i] = cell

	def __delitem__(self, i):
		self.cells.__delitem__(i)

	def keys(self):
		try:
			return self.cells.keys()
		except AttributeError:
			return range(len(self.cells))

	def values(self):
		try:
			return self.cells.values()
		except AttributeError:
			return self.cells

	def items(self):
		try:
			return self.cells.items()
		except AttributeError:
			return enumerate(self.cells)


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
	Spatially constrained subset of (trans-)locations with associated intermediate calculations.

	Attributes:

		index (int):
			this cell's index as granted in :class:`Distributed`'s `cells` dict.

		center (array-like):
			cell center coordinates.

		span (array-like):
			difference vectors from this cell's center to adjacent centers. Useful as 
			mixing coefficients for summing the multiple local gradients of a scalar 
			dynamic parameter.

		time_col (int or string, lazy):
			column index for time.

		space_cols (list of ints or strings, lazy):
			column indices for coordinates.

		tcount (int):
			number of (trans-)locations.

		cache (any):
			depending on the inference approach and objective, caching an intermediate
			result may avoid repeating many times a same computation. Usage of this cache
			is totally free and comes without support for concurrency.

	"""
	__slots__ = ('_time_col', '_space_cols', 'cache', 'fuzzy')
	__lazy__  = Local.__lazy__ + ('time_col', 'space_cols')

	def __init__(self, index, data, center=None, span=None):
		if not (isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame)):
			raise TypeError('unsupported (trans-)location type `{}`'.format(type(data)))
		Local.__init__(self, index, data, center, span)
		#self._tcount = data.shape[0]
		self._time_col = None
		self._space_cols = None
		self.cache = None
		#self.data = (self.space_data, self.time_data)
		self.fuzzy = None

	@property
	def time_col(self):
		if self._time_col is None:
			if isstructured(self.data):
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
			if isstructured(self.data):
				self._space_cols = columns(self.data)
				if isinstance(self._space_cols, pd.Index):
					self._space_cols = self._space_cols.drop(self.time_col)
				else:
					self._space_cols.remove(self.time_col)
			else:
				if self.time_col == 0:
					self._space_cols = np.arange(1, self.data.shape[1])
				else:
					self._space_cols = np.ones(self.data.shape[1], \
						dtype=bool)
					self._space_cols[self.time_col] = False
					self._space_cols, = self._space_cols.nonzero()
		return self._space_cols

	@space_cols.setter
	def space_cols(self, cols):
		# time_col is left unchanged
		self.__lazysetter__(cols)

	@property
	def dim(self):
		return len(self.space_cols)

	@dim.setter
	def dim(self, d): # ro
		self.__lazyassert__(d, 'data')

	@property
	def tcount(self):
		return self.space_data.shape[0]

	@tcount.setter
	def tcount(self, c): # ro
		self.__lazyassert__(c, 'data')

	@property
	def time_data(self):
		if not isinstance(self.data, tuple):
			self.data = (self._extract_space(), self._extract_time())
		return self.data[1]

	@time_data.setter
	def time_data(self, t):
		self.data = (self.space_data, t)

	def _extract_time(self):
		if isstructured(self.data):
			return np.asarray(self.data[self.time_col])
		else:
			return np.asarray(self.data[:,self.time_col])

	@property
	def space_data(self):
		if not isinstance(self.data, tuple):
			self.data = (self._extract_space(), self._extract_time())
		return self.data[0]

	@space_data.setter
	def space_data(self, xy):
		self.data = (xy, self.time_data)

	def _extract_space(self):
		if isstructured(self.data):
			return np.asarray(self.data[self.space_cols])
		else:
			return np.asarray(self.data[:,self.space_cols])

	def __nonzero__(self):
		if isinstance(self.data, tuple):
			return 0 < self.data[0].size
		else:
			return 0 < self.data.size


class Locations(Cell):
	"""
	Attributes:

		locations (array-like, property):
			locations as a matrix of coordinates and times with as many 
			columns as dimensions; this is an alias for :attr:`~Local.data`.

		t (array-like, ro property):
			location timestamps.

		xy (array-like, ro property):
			location coordinates.

	"""
	__slots__ = ()

	@property
	def locations(self):
		return self.data

	@locations.setter
	def locations(self, tr):
		self.data = tr

	@property
	def t(self):
		return self.time_data

	@t.setter
	def t(self, t):
		self.time_data = t

	@property
	def xy(self):
		return self.space_data

	@xy.setter
	def xy(self, xy):
		self.space_data = xy


class Translocations(Cell):
	"""
	Attributes:

		translocations (array-like, property):
			translocations as a matrix of variations of coordinate and time with as many 
			columns as dimensions; this is an alias for :attr:`~Local.data`.

		dt (array-like, ro property):
			translocation durations.

		dxy (array-like, ro property):
			translocation changes in coordinate.

		origins (array-like, ro property):
			initial locations (both spatial coordinates and times).

		t (array-like, ro property):
			initial timestamps.

	"""
	__slots__ = ('origins', 'destinations')

	def __init__(self, index, translocations, center=None, span=None, origins=None):
		Cell.__init__(self, index, translocations, center, span)
		self.origins = None
		self.destinations = None

	@property
	def translocations(self):
		return self.data

	@translocations.setter
	def translocations(self, tr):
		self.data = tr

	@property
	def dt(self):
		return self.time_data

	@dt.setter
	def dt(self, dt):
		self.time_data = dt

	@property
	def dxy(self):
		return self.space_data

	@dxy.setter
	def dxy(self, dxy):
		self.space_data = dxy

	@property
	def t(self):
		if isstructured(self.origins):
			return np.asarray(self.origins[self.time_col])
		else:
			return np.asarray(self.origins[:,self.time_col])



def identify_columns(points, trajectory_col=True):
	_has_trajectory = trajectory_col not in [False, None]
	_traj_undefined = trajectory_col is True
	if isstructured(points):
		coord_cols = columns(points)
		if _has_trajectory:
			if _traj_undefined:
				trajectory_col = 'n'
			try:
				if isinstance(coord_cols, pd.Index):
					coord_cols = coord_cols.drop(trajectory_col)
				else:
					coord_cols.remove(trajectory_col)
			except:
				if _traj_undefined:
					_has_trajectory = False
				else:
					raise
	else:
		if not _has_trajectory:
			coord_cols = np.arange(points.shape[1])
		elif _traj_undefined:
			trajectory_col = 0
			coord_cols = np.arange(1, points.shape[1])
		else:
			coord_cols = np.r_[np.arange(trajectory_col), \
				np.arange(trajectory_col+1, points.shape[1])]
	if not _has_trajectory:
		trajectory_col = None
	if isinstance(points, pd.DataFrame):
		def get_point(a, i):
			return a.iloc[i]
		def get_var(a, j):
			return a[j]
	else:
		if isstructured(points):
			def get_point(a, i):
				return a[i,:]
			def get_var(a, j):
				return a[j]
		else:
			def get_point(a, i):
				return a[i]
			def get_var(a, j):
				return a[:,j]
	return coord_cols, trajectory_col, get_var, get_point


def get_locations(points, index=None, coord_cols=None, get_var=None, get_point=None):
	if coord_cols is None:
		coord_cols, _, get_var, get_point = identify_columns(points)
	locations = get_var(points, coord_cols)
	location_cell = format_cell_index(index, 'array', nearest_cell, (locations.shape[0],))
	return locations, location_cell, get_point


def get_translocations(points, index=None, coord_cols=None, trajectory_col=True, get_var=None, get_point=None):
	if coord_cols is None:
		coord_cols, trajectory_col, get_var, get_point = identify_columns(points, trajectory_col)
	if trajectory_col is None:
		raise ValueError('cannot find trajectory indices')
	n = np.asarray(get_var(points, trajectory_col))
	final = np.r_[False, np.diff(n, axis=0) == 0]
	initial = np.r_[final[1:], False]
	# points
	points = get_var(points, coord_cols)
	initial_point = get_point(points, initial)
	final_point = get_point(points, final)
	# cell indices
	if index is None:
		initial_cell = final_cell = None
	elif isinstance(index, np.ndarray):
		initial_cell = index[initial]
		final_cell = index[final]
	elif isinstance(index, tuple):
		ix = np.full(points.shape[0], -1, dtype=index[1].dtype)
		if isinstance(points, pd.DataFrame):
			ix = pd.DataFrame(data=ix, index=points.index, columns=('cell',))
			ix.loc[index[0], 'cell'] = index[1]
			initial_cell = np.asarray(ix['cell'].iloc[initial])
			final_cell = np.asarray(ix['cell'].iloc[final])
		else:
			ix[index[0]] = index[1]
			initial_cell = ix[initial]
			final_cell = ix[final]
	elif sparse.issparse(index): # sparse matrix
		index = index.tocsr(True)
		initial_cell = index[initial].indices # and not cells.cell_index!
		final_cell = index[final].indices
	else:
		raise ValueError('wrong index format')
	return initial_point, final_point, initial_cell, final_cell, get_point


def distributed(cells, new_cell=None, new_mesh=Distributed, fuzzy=None,
		new_cell_kwargs={}, new_mesh_kwargs={}, fuzzy_kwargs={},
		new=None):
	if new is not None:
		# `new` is for backward compatibility
		warn('`new` is deprecated in favor of `new_cell`', DeprecationWarning)
		new_mesh = new
	if not isinstance(cells, tessellation.CellStats):
		raise TypeError('`cells` is not a `CellStats`')
	# simplify the adjacency matrix
	if cells.tessellation.cell_label is None:
		J = cells.location_count
	else:
		J = np.logical_and(0 < cells.location_count, 0 < cells.tessellation.cell_label)
	_adjacency = cells.tessellation.simplified_adjacency(label=J, format='csr')
	# reweight each row i as 1/n_i where n_i is the degree of cell i
	n = np.diff(_adjacency.indptr)
	_adjacency.data[...] = np.repeat(1.0 / np.maximum(1, n), n)
	ncells = _adjacency.shape[0]

	# format (trans-)locations
	precomputed = ()
	if isinstance(new_cell, Locations):
		are_translocations = False
	elif isinstance(new_cell, Translocations):
		are_translocations = True
	else:
		coord_cols, trajectory_col, get_var, get_point = identify_columns(cells.points)
		are_translocations = trajectory_col is not None
		if are_translocations:
			precomputed = (coord_cols, trajectory_col, get_var, get_point)
		else:
			precomputed = (coord_cols, get_var, get_point)
	if are_translocations:
		initial_point, final_point, initial_cell, final_cell, get_point = \
			get_translocations(cells.points, cells.cell_index, *precomputed)
		if new_cell is None:
			new_cell = Translocations
		fuzzy_args = (final_point, final_cell, initial_point, initial_cell, get_point)
	else:
		locations, location_index, get_point = \
			get_locations(cells.points, cells.cell_index, *precomputed)
		fuzzy_args = (locations, location_index, get_point)
		if new_cell is None:
			new_cell = Locations

	if fuzzy is None:
		if are_translocations:
			def f(tessellation, cell, final_point, final_cell,
				initial_point, initial_cell, get_point):
				## example handling:
				#j = np.logical_or(initial_cell == cell, final_cell == cell)
				#initial_point = get_point(initial_point, j)
				#final_point = get_point(final_point, j)
				#initial_cell = initial_cell[j]
				#final_cell = final_cell[j]
				#i = final_cell == cell # criterion i could be more sophisticated
				#j[j] = i
				#return j
				return final_cell == cell
		else:
			def f(tesselation, cell, locations, location_cell, get_point):
				return location_cell == cell
		fuzzy = f

	# time and space columns in (trans-)locations array
	if isstructured(cells.points):
		time_col = 't'
		space_cols = columns(cells.points)
		if 'n' in space_cols:
			not_space = ['n', time_col]
		else:
			not_space = [time_col]
		if isinstance(space_cols, pd.Index):
			space_cols = space_cols.drop(not_space)
		else:
			space_cols = [ c for c in space_cols if c not in not_space ]
	else:
		time_col = cells.points.shape[1] - 1
		if time_col == cells.points.shape[1] - 1:
			space_cols = np.arange(time_col)
		else:
			space_cols = np.ones(cells.points.shape[1], dtype=bool)
			space_cols[time_col] = False
			space_cols, = space_cols.nonzero()

	# build every cells
	J, = np.nonzero(J)
	_cells = OrderedDict()
	for j in J: # for each cell
		if cells.location_count[j] == 0:
			continue
		i = fuzzy(cells.tessellation, j, *fuzzy_args, **fuzzy_kwargs)
		if i.dtype in (bool, np.bool, np.bool8, np.bool_):
			_fuzzy = None
		else:
			_fuzzy = i[i != 0]
			i = i != 0
		if are_translocations:
			_origin = get_point(initial_point, i)
			_destination = get_point(final_point, i)
			__origin = _origin.copy() # make copy
			__origin.index += 1
			data = translocations = _destination - __origin
		else:
			data = locations
		try:
			center = cells.tessellation.cell_centers[j]
		except AttributeError:
			center = span = None
		else:
			adj = _adjacency[j].indices
			span = cells.tessellation.cell_centers[adj] - center
		#if translocations.size:
		# make cell object
		_cells[j] = new_cell(j, data, center, span, **new_cell_kwargs)
		_cells[j].time_col = time_col
		_cells[j].space_cols = space_cols
		if are_translocations:
			try:
				_cells[j].origins = _origin
			except AttributeError:
				pass
			try:
				_cells[j].destinations = _destination
			except AttributeError:
				pass
		try:
			_cells[j].fuzzy = _fuzzy
		except AttributeError:
			pass
	#print(sum([ c.tcount == 0 for c in _cells.values() ]))
	self = new_mesh(_cells, _adjacency, **new_mesh_kwargs)
	self.tcount = cells.points.shape[0]
	#self.dim = cells.points.shape[1]
	self.ccount = self.adjacency.shape[0]
	return self


class Maps(object):
	"""
	Basic container for maps and the associated parameters used to get the maps.
	"""
	def __init__(self, maps, mode=None):
		self.maps = maps
		self.mode = mode
		self.min_diffusivity = None
		self.localization_error = None
		self.diffusivity_prior = None
		self.potential_prior = None
		self.jeffreys_prior = None
		self.extra_args = None
		self.distributed_translocations = None # legacy attribute
		self.partition_file = None # legacy attribute
		self.tessellation_param = None # legacy attribute
		self.version = None # legacy attribute
		self.runtime = None

	def __nonzero__(self):
		return self.maps.__nonzero__()


class DiffusivityWarning(RuntimeWarning):
	def __init__(self, diffusivity, lower_bound):
		self.diffusivity = diffusivity
		self.lower_bound = lower_bound

	def __repr__(self):
		return 'DiffusivityWarning({}, {})'.format(self.diffusivity, self.lower_bound)

	def __str__(self):
		return 'diffusivity too low: {} < {}'.format(self.diffusivity, self.lower_bound)



__all__ = ['Local', 'Distributed', 'Cell', 'Locations', 'Translocations', 'Maps',
	'identify_columns', 'get_locations', 'get_translocations', 'distributed',
	'DiffusivityWarning']

