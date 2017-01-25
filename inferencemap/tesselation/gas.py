
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
	"""GWR based tesselation.

	Attributes:
		gas (:class:`~inferencemap.spatial.gas.Gas`):
			internal graph representation of the gas.
		min_probability (float):
			minimum probability of a point to be in any given cell.

	Other Attributes:
		_min_distance (float):
			scaled minimum distance between adjacent cell centers.
		_avg_distance (float):
			upper bound on the average scaled distance between adjacent cell centers.
		_max_distance (float):
			scaled maximum distance between adjacent cell centers."""
	def __init__(self, scaler=Scaler(), min_distance=None, avg_distance=None, max_distance=None, \
		min_probability=None, avg_probability=None, **kwargs):
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
		self.min_probability = min_probability
		#self.avg_probability = avg_probability

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
				# distances are diameters, while insertion thresholds should be radii
				self.gas.insertion_threshold = (self._avg_distance * 0.5, \
					self._max_distance * 0.5)
				if self.min_probability:
					self.gas.knn = int(round(self.min_probability * \
						points.shape[0]))
				else:
					self.gas.knn = 20
			self.gas.trust = trust
			self.gas.batch_size = batch_size
			if not isinstance(tau, tuple):
				tau = (tau, tau)
			self.gas.habituation_tau = tau
			self.gas.edge_lifetime = lifetime
			if self._min_distance:
				self.gas.collapse_below = self._min_distance# * 0.9
		return points

	def tesselate(self, points, pass_count=(1,3), residual_factor=.7, error_count_tol=5e-3, \
		min_growth=1e-4, collapse_tol=.01, stopping_criterion=0, verbose=False, \
		plot=False, **kwargs):
		"""Grow the tesselation.

		Arguments:
			points: see :meth:`~inferencemap.tesselation.base.Tesselation.tesselate`.
			pass_count (pair of floats): minimum and maximum numbers of times the data
				should (in principle) be consumed.
			residual_factor (float): multiplies with `_max_distance` to determine 
				`residual_max` in :meth:`~inferencemap.spatial.gas.Gas.train`.
			error_count_tol (float): (see :meth:`~inferencemap.spatial.gas.Gas.train`)
			min_growth (float): (see :meth:`~inferencemap.spatial.gas.Gas.train`)
			collapse_tol (float): (see :meth:`~inferencemap.spatial.gas.Gas.train`)
			stopping_criterion (int): (see :meth:`~inferencemap.spatial.gas.Gas.train`)
			verbose (bool): verbose output.
			batch_size (int): (see :class:`~inferencemap.spatial.gas.Gas`)
			tau (float): (see :class:`~inferencemap.spatial.gas.Gas`)
			trust (float): (see :class:`~inferencemap.spatial.gas.Gas`)
			lifetime (int): (see :class:`~inferencemap.spatial.gas.Gas`)

		Returns:
			See :meth:`~inferencemap.tesselation.base.Tesselation.tesselate`.

		See also:
			:class:`inferencemap.spatial.gas.Gas` and 
			:meth:`inferencemap.spatial.gas.Gas.train`.
		"""
		#np.random.seed(15894754) # to benchmark and compare between Graph implementations
		points = self._preprocess(points, **kwargs)
		if self._avg_distance:
			residual_factor *= self._avg_distance # important: do this after _preprocess!
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
		self._cell_adjacency.data = np.ones_like(self._cell_adjacency.data, dtype=int)
		self._postprocess(points, verbose)

	def _postprocess(self, points=None, verbose=False):
		# build the Voronoi graph
		voronoi = Voronoi._postprocess(self)
		# clean and extend the adjacency matrix with the Delaunay graph
		adjacency = self._cell_adjacency
		delaunay = sparse.csr_matrix(([2] * (voronoi.ridge_points.shape[0] * 2), \
			(voronoi.ridge_points.flatten(), np.fliplr(voronoi.ridge_points).flatten())), \
			shape=adjacency.shape)
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
			A = sparse.tril(self._cell_adjacency, format='coo') # in future scipy version, check that tril does not remove explicit zeros
			for i, j, k in zip(A.row, A.col, A.data):
				if self._adjacency_label[k] == 2: # only in Voronoi
					xi = points[ix == i]
					xj = points[ix == j]
					if xi.size and xj.size:
						dij = cdist(xi, xj).flatten()
						#dij = np.dot(xi, xj.T)
						#xi2 = np.sum(xi * xi, axis=1, keepdims=True)
						#dij -= 0.5 * xi2
						#xj2 = np.sum(xj * xj, axis=1, keepdims=True)
						#dij -= 0.5 * xj2.T
						#dij = dij.flatten()
						dij.sort()
						try:
							#dij = dij[-int(ceil(self.gas.knn/4))]
							dij = dij[int(ceil(self.gas.knn/4))-1]
						except IndexError:
							if 1 < verbose:
								print('skipping edge {:d} between cell {:d} (card = {:d}) and cell {:d} (card = {:d}): number of between-cell pairs = {:d} (expected: {:d})'.format(k, i, xi.shape[0], j, xj.shape[0], dij.size, int(ceil(self.gas.knn/4))))
							continue
						#dij = np.sqrt(-2.0 * dij)
						if dij < self._avg_distance:
							self._adjacency_label[k] = 4 # mark edge as 'not congruent but valid'
					elif 1 < verbose:
						print('skipping edge {:d} between cell {:d} (card = {:d}) and cell {:d} (card = {:d})'.format(k, i, xi.shape[0], j, xj.shape[0]))
		new_labels = np.array([0,-1,-2,1,2])
		# before: 0=[none], 1=not congruent (gas only), 2=not congruent (voronoi only), 
		#         3=congruent, 4=voronoi added
		# after: -2=not congruent (voronoi only), -1=not congruent (gas only), 0=[none], 
		#         1=congruent, 2=voronoi added
		self._adjacency_label = new_labels[self._adjacency_label]

