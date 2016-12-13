
from math import *
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import scipy.sparse as sparse
import scipy.spatial as spatial
from inferencemap.spatial.scaler import *


class CellStats(object):
	"""Container datatype for various results related to a sample and a tesselation."""
	__slots__ = ['coordinates', 'cell_index', 'cell_count', 'bounding_box', 'param']

	def __init__(self, cell_index=None, cell_count=None, bounding_box=None, coordinates=None, \
		param={}):
		self.coordinates = coordinates
		self.cell_index = cell_index
		self.cell_count = cell_count
		self.bounding_box = bounding_box
		self.param = param


def point_adjacency_matrix(tess, stats, symetric=True, cell_labels=None, adjacency_labels=None):
	"""returns an adjacency matrix of data points where a given pair of points is defined as
	adjacent iif they belong to adjacent and distinct cells.
	"""
	x = np.asarray(stats.coordinates)
	ij = np.arange(x.shape[0])
	x2 = np.sum(x * x, axis=1)
	x2.shape = (x2.size, 1)
	I = []
	J = []
	D = []
	n = []
	for i in np.arange(tess.cell_adjacency.shape[0]):
		if cell_labels is not None and not cell_labels(tess.cell_label[i]):
			continue
		_, js, k = sparse.find(tess.cell_adjacency[i])
		if js.size == 0:
			continue
		k = k[i < js]
		js = js[i < js]
		if js.size == 0:
			continue
		if adjacency_labels is not None:
			js = js[adjacency_labels(k)]
			if js.size == 0:
				continue
		if cell_labels is not None:
			js = js[cell_labels(tess.cell_label[js])]
			if js.size == 0:
				continue
		ii = ij[stats.cell_index == i]
		xi = np.asarray(stats.coordinates)[stats.cell_index == i]
		x2i = x2[stats.cell_index == i]
		for j in js:
			xj = np.asarray(stats.coordinates)[stats.cell_index == j]
			x2j = x2[stats.cell_index == j]
			d2 = x2i + x2j.T - 2 * np.dot(xi, xj.T)
			jj = ij[stats.cell_index == j]
			i2, j2 = np.meshgrid(ii, jj, indexing='ij')
			I.append(i2.flatten())
			J.append(j2.flatten())
			D.append(d2.flatten())
			if symetric:
				I.append(j2.flatten())
				J.append(i2.flatten())
				D.append(d2.flatten())
	I = np.concatenate(I)
	J = np.concatenate(J)
	D = np.sqrt(np.concatenate(D))
	n = stats.coordinates.shape[0]
	return sparse.csr_matrix((D, (I, J)), shape=(n, n))



class Tesselation(object):
	"""Abstract class for tesselations. Main methods to be implemented are :meth:`tesselate` and
	:meth:`cellIndex`. :meth:`cellStats` is for convenience."""
	def __init__(self, scaler=Scaler()):
		self.scaler = scaler
		self._cell_adjacency = None
		self._cell_label = None
		self._adjacency_label = None

	def _preprocess(self, points):
		if self._cell_centers is None and self.scaler.euclidian is None:
			# initialize
			if isinstance(points, pd.DataFrame):
				self.scaler.euclidian = ['x', 'y']
				if not ('x' in points and 'y' in points): # enforce presence of 'x' and 'y'
					raise AttributeError('missing ''x'' or ''y'' in input dataframe.')
				if 'z' in points:
					self.scaler.euclidian.append('z')
			else:	self.scaler.euclidian = np.arange(0, points.shape[1])
		return self.scaler.scalePoint(points)

	def tesselate(self, points, **kwargs):
		''':meth:`tesselate` takes a :class:`pandas.core.frame.DataFrame` or :class:`numpy.array`.
		Columns of the `DataFrame` are treated according to their name: 'x', 'y' and 'z' are 
		spatial variables and are included in :attr:`~mesh.scaler.Scaler.euclidian`.'''
		raise NotImplementedError

	def cellIndex(self, points):
		raise NotImplementedError

	def cellStats(self, points, **kwargs):
		cell_index = self.cellIndex(points, **kwargs)
		valid_cell_centers, center_to_point, _cell_count = \
			np.unique(cell_index, return_inverse=True, return_counts=True)
		cell_count = np.zeros(self._cell_centers.shape[0], dtype=_cell_count.dtype)
		cell_count[valid_cell_centers] = _cell_count
		center_to_point = valid_cell_centers[center_to_point]
		xmin = points.min(axis=0)
		xmax = points.max(axis=0)
		if isinstance(points, pd.DataFrame):
			bounding_box = pd.concat([xmin, xmax], axis=1).T
			bounding_box.index = ['min', 'max']
		else:
			bounding_box = np.vstack([xmin, xmax]).T.flatten()
		return CellStats(cell_index=cell_index, cell_count=cell_count, \
			bounding_box=bounding_box, coordinates=points)

	# cell_label property
	@property
	def cell_label(self):
		return self._cell_label

	@cell_label.setter
	def cell_label(self, label):
		self._cell_label = label

	# cell_adjacency property
	@property
	def cell_adjacency(self):
		return self._cell_adjacency

	@cell_adjacency.setter
	def cell_adjacency(self, matrix):
		self._cell_adjacency = matrix

	# adjacency_label property
	@property
	def adjacency_label(self):
		return self._adjacency_label

	@adjacency_label.setter
	def adjacency_label(self, label):
		self._adjacency_label = label


class Delaunay(Tesselation):
	def __init__(self, scaler=Scaler()):
		Tesselation.__init__(self, scaler)
		self._cell_centers = None

	def tesselate(self, points):
		self._cell_centers = self._preprocess(points)

	def cellIndex(self, points, knn=None, metric='euclidian', **kwargs):
		D = cdist(np.asarray(self.scaler.scalePoint(points, inplace=False)), \
				self._cell_centers, metric, **kwargs)
		if knn:
			I = np.argsort(D, axis=0)[:knn].flatten()
			J = np.repeat(range(0, D.shape[1]), knn)
			K = np.argmin(D, axis=1)[I]
			I[J != K] = -1
			return I
		else:
			return np.argmin(D, axis=1)

	# cell_centers property
	@property
	def cell_centers(self):
		if isinstance(self.scaler.factor, pd.Series):
			return self.scaler.unscalePoint(pd.DataFrame(self._cell_centers, \
				columns=self.scaler.factor.index))
		else:
			return self.scaler.unscalePoint(self._cell_centers)

	@cell_centers.setter
	def cell_centers(self, centers):
		self._cell_centers = self.scaler.scalePoint(centers)


class Voronoi(Delaunay):
	def __init__(self, scaler=Scaler()):
		Delaunay.__init__(self, scaler)
		self._cell_vertices = None
		self._ridge_vertices = None

	# cell_vertices property
	@property
	def cell_vertices(self):
		if self._cell_centers is not None and self._cell_vertices is None:
			self._postprocess()
		if isinstance(self.scaler.factor, pd.Series):
			return self.scaler.unscalePoint(pd.DataFrame(self._cell_vertices, \
				columns=self.scaler.factor.index))
		else:
			return self.scaler.unscalePoint(self._cell_vertices)

	@cell_vertices.setter
	def cell_vertices(self, vertices):
		self._cell_vertices = self.scaler.scalePoint(vertices)

	# cell_adjacency property
	@property
	def cell_adjacency(self):
		if self._cell_centers is not None and self._cell_adjacency is None:
			self._postprocess()
		return self._cell_adjacency

	# whenever you redefine a getter you have to redefine the corresponding setter
	@cell_adjacency.setter # copy/paste
	def cell_adjacency(self, matrix):
		self._cell_adjacency = matrix

	# ridge_vertices property
	@property
	def ridge_vertices(self):
		if self._cell_centers is not None and self._ridge_vertices is None:
			self._postprocess()
		return self._ridge_vertices

	@ridge_vertices.setter
	def ridge_vertices(self, ridges):
		self._ridge_vertices = ridges

	def _postprocess(self):
		if self._cell_centers is None:
			raise NameError('`cell_centers` not defined; tesselation has not been grown yet')
		else:
			voronoi = spatial.Voronoi(np.asarray(self._cell_centers))
			self._cell_vertices = voronoi.vertices
			n_centers = self._cell_centers.shape[0]
			self._ridge_vertices = np.asarray(voronoi.ridge_vertices)
			if self._cell_adjacency is None:
				n_ridges = voronoi.ridge_points.shape[0]
				self._cell_adjacency = sparse.csr_matrix((\
					np.arange(0, n_ridges * 2, dtype=np.uint), (\
					voronoi.ridge_points.flatten(), \
					np.fliplr(voronoi.ridge_points).flatten())), \
					shape=(n_centers, n_centers))
			return voronoi

class RegularMesh(Voronoi):
	def __init__(self, scaler=None, lower_bound=None, upper_bound=None, count_per_dim=None, min_probability=None, max_probability=None, avg_probability=None, **kwargs):
		Voronoi.__init__(self) # just ignore `scaler`
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound
		self.count_per_dim = count_per_dim
		self.min_probability = min_probability
		self.max_probability = max_probability
		self.avg_probability = avg_probability

	def tesselate(self, points, **kwargs):
		#points = self._preprocess(points)
		if self.lower_bound is None:
	 		self.lower_bound = points.min(axis=0)
		elif isinstance(points, pd.DataFrame) and not isinstance(self.lower_bound, pd.Series):
			self.lower_bound = pd.Series(self.lower_bound, index=points.columns)
		if self.upper_bound is None:
			self.upper_bound = points.max(axis=0)
		elif isinstance(points, pd.DataFrame) and not isinstance(self.upper_bound, pd.Series):
			self.upper_bound = pd.Series(self.upper_bound, index=points.columns)
		if self.count_per_dim is None:
			size = self.upper_bound - self.lower_bound
			if self.avg_probability:
				n_cells = 1.0 / self.avg_probability
			else:
				raise NotImplementedError
			increment = exp(log(np.asarray(size).prod() / n_cells) / len(points.columns))
			if isinstance(size, pd.Series):
				self.count_per_dim = pd.Series.round(size / increment)
			else:
				self.count_per_dim = np.round(size / increment)
		elif isinstance(points, pd.DataFrame) and not isinstance(self.count_per_dim, pd.Series):
			self.count_per_dim = pd.Series(self.count_per_dim, index=points.columns)
		if isinstance(points, pd.DataFrame):
			grid = pd.concat([self.lower_bound, self.upper_bound, self.count_per_dim + 1], axis=1).T
			self.grid = [ np.linspace(*col.values) for _, col in grid.iteritems() ]
		else: raise NotImplementedError
		cs = np.meshgrid(*[ (g[:-1] + g[1:]) / 2 for g in self.grid ], indexing='ij')
		self._cell_centers = np.column_stack([ c.flatten() for c in cs ])

	def _postprocess(self):
		pass

	# cell_centers property
	@property
	def cell_centers(self):
		return self._cell_centers

	@cell_centers.setter
	def cell_centers(self, centers):
		self._cell_centers = centers

	# cell_vertices property
	@property
	def cell_vertices(self):
		if self._cell_vertices is None:
			vs = np.meshgrid(*self.grid, indexing='ij')
			self._cell_vertices = np.column_stack([ v.flatten() for v in vs ])
		return self._cell_vertices

	@cell_vertices.setter
	def cell_vertices(self, vertices):
		self._cell_vertices = vertices

	# cell_adjacency property
	@property
	def cell_adjacency(self):
		if self._cell_adjacency is None:
			cix = np.meshgrid(*[ np.arange(0, len(g) - 1) for g in self.grid ], \
				indexing='ij')
			cix = np.column_stack([ g.flatten() for g in cix ])
			c2  = np.atleast_2d(np.sum(cix * cix, axis=1))
			self._cell_adjacency = sparse.csr_matrix(\
				np.abs(c2 + c2.T - 2 * np.dot(cix, cix.T) - 1.0) < 1e-6)
		return self._cell_adjacency

	@cell_adjacency.setter # copy/paste
	def cell_adjacency(self, matrix):
		self._cell_adjacency = matrix

	# ridge_vertices property
	@property
	def ridge_vertices(self):
		if self._ridge_vertices is None:
			vix = np.meshgrid(*[ np.arange(0, len(g)) for g in self.grid ], \
				indexing='ij')
			vix = np.column_stack([ g.flatten() for g in vix ])
			v2  = np.atleast_2d(np.sum(vix * vix, axis=1))
			vix = sparse.coo_matrix(np.abs(v2 + v2.T - 2 * np.dot(vix, vix.T) - 1.0) < 1e-6)
			self._ridge_vertices = np.column_stack((vix.row, vix.col))
		return self._ridge_vertices

	@ridge_vertices.setter # copy/paste
	def ridge_vertices(self, ridges):
		self._ridge_vertices = ridges

