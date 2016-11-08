
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import scipy.sparse as sparse
import scipy.spatial as spatial
from inferencemap.spatial.scaler import *


class CellStats:
	"""Container datatype for various results related to a sample and a tesselation."""
	def __init__(self, cell_index=None, cell_count=None, cell_mask=None, cell_to_point=None, \
		point_mask=None, bounding_box=None):
		self.cell_index = cell_index
		self.cell_count = cell_count
		self.cell_to_point = cell_to_point
		self.bounding_box = bounding_box


class Tesselation:
	def __init__(self, scaler=Scaler()):
		self.scaler = scaler
		self._cell_centers = None
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

	def tesselate(self, points):
		''':meth:`tesselate` takes a :class:`pandas.core.frame.DataFrame` or :class:`numpy.array`.
		Columns of the `DataFrame` are treated according to their name: 'x', 'y' and 'z' are 
		spatial variables and are included in :attr:`~mesh.scaler.Scaler.euclidian`.'''
		self._cell_centers = self._preprocess(points)
		self._postprocess()

	def _postprocess(self):
		pass

	def cellIndex(self, points):
		return np.argmin(cdist(np.asarray(self.scaler.scalePoint(points.copy(deep=False))), \
				self._cell_centers), 
			axis=1)

	def cellStats(self, points):
		cell_index = self.cellIndex(points)
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
		return CellStats(cell_index, cell_count, cell_to_point=center_to_point, \
			bounding_box=bounding_box)

	@property
	def cell_centers(self):
		if isinstance(self.scaler.factor, pd.Series):
			return self.scaler.unscalePoint(pd.DataFrame(self._cell_centers, \
				columns=self.scaler.factor.index))
		else:
			return self.scaler.unscalePoint(self._cell_centers)

	@property
	def cell_label(self):
		return self._cell_label

	@property
	def cell_adjacency(self):
		return self._cell_adjacency

	@property
	def adjacency_label(self):
		return self._adjacency_label



class Voronoi(Tesselation):
	def __init__(self, scaler=Scaler()):
		Tesselation.__init__(self, scaler)
		self._cell_vertices = None
		self._ridge_vertices = None

	@property
	def cell_vertices(self):
		if isinstance(self.scaler.factor, pd.Series):
			return self.scaler.unscalePoint(pd.DataFrame(self._cell_vertices, \
				columns=self.scaler.factor.index))
		else:
			return self.scaler.unscalePoint(self._cell_vertices)

	@property
	def ridge_vertices(self):
		return self._ridge_vertices

	def _postprocess(self):
		if self._cell_centers is None:
			raise NameError('`cell_centers` not defined; tesselation has not been grown yet')
		else:
			voronoi = spatial.Voronoi(np.asarray(self._cell_centers))
			self._cell_vertices = voronoi.vertices
			n_centers = self._cell_centers.shape[0]
			self._ridge_vertices = np.asarray(voronoi.ridge_vertices)
			if self.cell_adjacency is None:
				n_ridges = voronoi.ridge_points.shape[0]
				self._cell_adjacency = sparse.csr_matrix((\
					np.arange(0, n_ridges * 2, dtype=np.uint), (\
					voronoi.ridge_points.flatten(), \
					np.fliplr(voronoi.ridge_points).flatten())), \
					shape=(n_centers, n_centers))
			return voronoi

