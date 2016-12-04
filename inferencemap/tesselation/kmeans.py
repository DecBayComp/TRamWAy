
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
		#self.min_probability = min_probability
		#self.max_probability = None
		self.avg_probability = avg_probability
		#self.local_probability = None

	def _preprocess(self, points):
		points = Voronoi._preprocess(self, points)
		grid = RegularMesh(avg_probability=self.avg_probability)
		grid.tesselate(points)
		self._cell_centers = grid._cell_centers
		return points

	def tesselate(self, points, tol=1e-3):
		points = self._preprocess(points)
		self._cell_centers, _ = kmeans(np.asarray(points), self._cell_centers, \
			thresh=tol)
		#self._postprocess()


