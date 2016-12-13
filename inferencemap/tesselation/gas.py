
from math import *
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from .base import *
from inferencemap.spatial.scaler import *
from inferencemap.spatial.gas import Gas
from scipy.spatial.distance import cdist
import time


class GasMesh(Voronoi):
	"""GWR based tesselation."""
	def __init__(self, scaler=Scaler(), min_distance=None, avg_distance=None, max_distance=None, \
		avg_probability=None, **kwargs):
		Voronoi.__init__(self, scaler)
		self.gas = None
		self._min_distance = min_distance
		if avg_distance or not max_distance:
			self._avg_distance = avg_distance
		else:
			self._avg_distance = max_distance * 0.25
		if max_distance or not avg_distance:
			self._max_distance = max_distance
		else:
			self._max_distance = avg_distance * 4
		self.avg_probability = avg_probability

	def _preprocess(self, points, batch_size=10000, tau=333.0, trust=1.0, lifetime=50, **kwargs):
		init = self.scaler.init
		points = Voronoi._preprocess(self, points)
		if init:
			self._min_distance = self.scaler.scaleDistance(self._min_distance)
			self._avg_distance = self.scaler.scaleDistance(self._avg_distance)
			self._max_distance = self.scaler.scaleDistance(self._max_distance)
		if self.gas is None:
			self.gas = Gas(np.asarray(points))
			if self._max_distance:
				self.gas.insertion_threshold = (self._avg_distance, self._max_distance)
				if self.avg_probability:
					self.gas.knn = int(round(2.0 * self.avg_probability * \
						points.shape[0]))
				else:
					self.gas.knn = 40
			self.gas.trust = trust
			self.gas.batch_size = batch_size
			if not isinstance(tau, tuple):
				tau = (tau, tau)
			self.gas.habituation_tau = tau
			self.gas.edge_lifetime = lifetime
			if self._min_distance:
				self.gas.collapse_below = self._min_distance * 0.9
		return points

	def tesselate(self, points, pass_count=(1,3), residual_factor=.7, error_count_tol=5e-3, \
		min_growth=1e-4, collapse_tol=.01, stopping_criterion=0, verbose=False, \
		plot=False, **kwargs):
		"""See :meth:`inferencemap.spatial.gas.Gas.train` for more information on the input
		parameters.
		`residual_max` is replaced by `residual_factor` which conveniently multiplies with 
		the scaled `max_distance` if available."""
		#np.random.seed(15894754) # to benchmark and compare between Graph implementations
		points = self._preprocess(points, **kwargs)
		if self._max_distance:
			residual_factor *= self._max_distance # important: do this after _preprocess!
		self.residuals = self.gas.train(np.asarray(points), \
			pass_count=pass_count, \
			residual_max=residual_factor, \
			error_count_tol=error_count_tol, \
			min_growth=min_growth, \
			collapse_tol=collapse_tol, \
			stopping_criterion=stopping_criterion, \
			verbose=verbose, plot=plot)
		# build alternative representation of the gas (or Delaunay graph)
		[self._cell_adjacency, V, _] = self.gas.export()
		self._cell_centers = V['weight']
		self._cell_adjacency.data = np.ones_like(self._cell_adjacency.data, dtype=np.int)
		self._postprocess(points, verbose)

	def _postprocess(self, points=None, verbose=False):
		# build the Voronoi graph
		voronoi = Voronoi._postprocess(self)
		# clean and extend the adjacency matrix with the Delaunay graph
		adjacency = self._cell_adjacency
		delaunay = sparse.csr_matrix(([2] * voronoi.ridge_points.shape[0], \
			(voronoi.ridge_points[:,0], voronoi.ridge_points[:,1])), shape=adjacency.shape)
		adjacency += delaunay
		self._adjacency_label = adjacency.data # 1=gas only, 2=voronoi only, 3=both
		adjacency.data = np.arange(0, self._adjacency_label.size) # edge indices for _adjacency_label and _ridge_vertices
		self._cell_adjacency = adjacency # probably useless since `adjacency` may be a view of `_cell_adjacency`
		# map the ridges index matrices
		order = adjacency[voronoi.ridge_points[:,0], voronoi.ridge_points[:,1]]
		new_ridge_vertices = -np.ones((self._adjacency_label.shape[0], 2), \
			dtype=self._ridge_vertices.dtype)
		new_ridge_vertices[order,:] = self._ridge_vertices
		self._ridge_vertices = new_ridge_vertices
		# reintroduce edges missing in the gas
		if points is not None:
			#t = time.time()
			points = np.asarray(points)
			ix = np.argmin(cdist(points, self._cell_centers), axis=1)
			I = np.repeat(np.arange(self._cell_adjacency.indptr.size - 1), \
				np.diff(self._cell_adjacency.indptr))
			J = self._cell_adjacency.indices
			for k, e in enumerate(self._cell_adjacency.data):
				if self._adjacency_label[e] == 2:
					xi = points[ix == I[k]]
					xj = points[ix == J[k]]
					if xi.size and xj.size:
						dij = np.dot(xi, xj.T)
						xi2 = np.sum(xi * xi, axis=1, keepdims=True)
						dij -= 0.5 * xi2
						xj2 = np.sum(xj * xj, axis=1, keepdims=True)
						dij -= 0.5 * xj2.T
						dij = dij.flatten()
						dij.sort()
						dij = dij[-self.gas.knn/4]
						dij = np.sqrt(-2.0 * dij)
						if dij < self._min_distance:
							self._adjacency_label[e] = 4
					elif verbose:
						print((k, I[k], J[k], xi.shape, xj.shape))

