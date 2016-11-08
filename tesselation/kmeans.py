
from .base import *
from inferencemap.spatial.scaler import *
from math import *
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans


class KMeansMesh(Voronoi):
	"""K-Means and Voronoi based tesselation."""
	def __init__(self, scaler=Scaler(), min_probability=None, avg_probability=None, **kwargs):
		Voronoi.__init__(self, scaler)
		self.min_probability = min_probability
		self.max_probability = None
		self.avg_probability = avg_probability
		self.local_probability = None

	def _preprocess(self, points):
		points = Voronoi._preprocess(self, points)
		self.lower_bound = points.min(axis=0)
		self.upper_bound = points.max(axis=0)
		size = self.upper_bound - self.lower_bound
		n_cells = 1.0 / self.avg_probability
		increment = sqrt(np.asarray(size).prod() / n_cells)
		count_per_dim = pd.Series.round(size / increment)
		grid = pd.concat([self.lower_bound, self.upper_bound, count_per_dim], axis=1).T
		grid = np.meshgrid(*[ np.linspace(*col.values) for _, col in grid.iteritems() ])
		grid = np.vstack([ g.flatten() for g in grid ]).T
		self._cell_centers = grid
		return points

	def tesselate(self, points, tol=1e-3):
		points = self._preprocess(points)
		self._cell_centers, _ = kmeans(np.asarray(points), self._cell_centers, \
			thresh=tol)
		self._postprocess()


