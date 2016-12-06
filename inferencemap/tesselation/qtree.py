
from .base import *
from inferencemap.spatial.scaler import *
from math import *
import numpy as np
import pandas as pd


class QTreeMesh(Voronoi):
	"""QuadTree and Voronoi based tesselation."""
	def __init__(self, scaler=None, lower_bound=None, upper_bound=None, avg_probability=None, **kwargs):
		Voronoi.__init__(self, Scaler())
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound
		self.avg_probability = avg_probability

	def tesselate(self, points, tol=1e-3):
		points = self._preprocess(points)
		if self.lower_bound is None:
	 		self.lower_bound = points.min(axis=0)
		elif isinstance(points, pd.DataFrame) and not isinstance(self.lower_bound, pd.Series):
			self.lower_bound = pd.Series(self.lower_bound, index=points.columns)
		if self.upper_bound is None:
			self.upper_bound = points.max(axis=0)
		elif isinstance(points, pd.DataFrame) and not isinstance(self.upper_bound, pd.Series):
			self.upper_bound = pd.Series(self.upper_bound, index=points.columns)
		self._cell_centers = 0.5 * (np.asarray(self.lower_bound) + np.asarray(self.upper_bound))
		#...
		raise NotImplementedError
		#self._postprocess()


