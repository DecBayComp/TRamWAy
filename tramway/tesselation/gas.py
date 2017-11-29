# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import *
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.stats as stats
from .base import *
from tramway.spatial.scaler import *
from tramway.spatial.gas import Gas
from scipy.spatial.distance import cdist
import time


class GasMesh(Voronoi):
	"""GWR based tesselation.

	Attributes:
		gas (:class:`~tramway.spatial.gas.Gas`):
			internal graph representation of the gas.
		min_probability (float):
			minimum probability of a point to be in any given cell.
		_min_distance (float, private):
			scaled minimum distance between adjacent cell centers.
		_avg_distance (float, private):
			upper bound on the average scaled distance between adjacent cell centers.
		_max_distance (float, private):
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
			self._min_distance = self.scaler.scale_distance(self._min_distance)
			self._avg_distance = self.scaler.scale_distance(self._avg_distance)
			self._max_distance = self.scaler.scale_distance(self._max_distance)
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

	def tesselate(self, points, pass_count=(), residual_factor=.7, error_count_tol=5e-3, \
		min_growth=1e-4, collapse_tol=.01, stopping_criterion=0, verbose=False, \
		plot=False, alpha_risk=1e-15, **kwargs):
		"""Grow the tesselation.

		Arguments:
			points: see :meth:`~tramway.tesselation.Tesselation.tesselate`.
			pass_count (float or pair of floats): 
				number of points to sample (with replacement) from data `points`, as
				a multiple of the size of the data.
				If `pass_count` is a pair of numbers, they are the lower and the upper
				bounds on the number of samples.
				If `pass_count` is a single number, it is interpreted as the lower
				bound, and the upper bound is set equal to ``2 * pass_count``.
			residual_factor (float): multiplies with `_max_distance` to determine 
				`residual_max` in :meth:`~tramway.spatial.gas.Gas.train`.
			error_count_tol (float): (see :meth:`~tramway.spatial.gas.Gas.train`)
			min_growth (float): (see :meth:`~tramway.spatial.gas.Gas.train`)
			collapse_tol (float): (see :meth:`~tramway.spatial.gas.Gas.train`)
			stopping_criterion (int): (see :meth:`~tramway.spatial.gas.Gas.train`)
			verbose (bool): verbose output.
			batch_size (int): (see :class:`~tramway.spatial.gas.Gas`)
			tau (float): (see :class:`~tramway.spatial.gas.Gas`)
			trust (float): (see :class:`~tramway.spatial.gas.Gas`)
			lifetime (int): (see :class:`~tramway.spatial.gas.Gas`)
			alpha_risk (float): location distributions of potential neighbor cells
				are compared with a t-test

		Returns:
			See :meth:`~tramway.tesselation.Tesselation.tesselate`.

		See also:
			:class:`tramway.spatial.gas.Gas` and 
			:meth:`tramway.spatial.gas.Gas.train`.
		"""
		#np.random.seed(15894754) # to benchmark and compare between Graph implementations
		points = self._preprocess(points, **kwargs)
		if self._avg_distance:
			residual_factor *= self._avg_distance # important: do this after _preprocess!
		if pass_count is not None:
			if pass_count is (): # () has been chosen to denote default (not-None) value
				n = points.shape[0]
				p = .95
				# sample size for each point to have probability p of being chosen once
				pass_count = log(1.0 - p) / log(1.0 - 1.0 / float(n))
				# convert in number of passes
				pass_count /= float(n)
			try:
				len(pass_count) # sequence?
			except TypeError:
				pass_count = (pass_count, 2 * pass_count)
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
		self.alpha_risk = alpha_risk
		self._postprocess(points, verbose)

	def _postprocess(self, points=None, verbose=False):
		# build the Voronoi graph
		voronoi = Voronoi._postprocess(self)
		# clean and extend the adjacency matrix with the Delaunay graph
		adjacency = self._cell_adjacency # shorter name
		# fix for issue on reload
		if 4 < adjacency.data[-1]:
			adjacency.data[:] = 1
		delaunay = sparse.csr_matrix( \
			(2 * np.ones(2 * voronoi.ridge_points.shape[0], dtype=int), \
			(voronoi.ridge_points.flatten('F'), \
			np.fliplr(voronoi.ridge_points).flatten('F'))), \
			shape=adjacency.shape)
		A = sparse.tril(adjacency + delaunay, format='coo')
		self._adjacency_label = A.data # labels are: 1=gas only, 2=voronoi only, 3=both
		# edge indices for _adjacency_label
		adjacency = sparse.csr_matrix( \
			(np.tile(np.arange(0, self._adjacency_label.size), 2), \
			(np.concatenate((A.row, A.col)), np.concatenate((A.col, A.row)))), \
			shape=adjacency.shape)
		self._cell_adjacency = adjacency
		## map the ridges index matrices
		#order = adjacency[voronoi.ridge_points[:,0], voronoi.ridge_points[:,1]]
		#new_ridge_vertices = -np.ones((self._adjacency_label.shape[0], 2), \
		#	dtype=self._ridge_vertices.dtype)
		#new_ridge_vertices[order,:] = self._ridge_vertices
		#self._cell_vertices = new_ridge_vertices
		# reintroduce edges missing in the gas
		if points is not None:
			#t = time.time()
			points = np.asarray(points)
			ix = np.argmin(cdist(points, self._cell_centers), axis=1)
			ref = int( ceil(float(self.gas.knn) / 8.0) ) # int and float for PY2
			#ref = -ref # with alternative to cdist, index from the end
			ref -= 1 # with cdist, index
			A = sparse.tril(self._cell_adjacency, format='coo') # in future scipy version, check that tril does not remove explicit zeros
			# compute the median distance between adjacent cell centers
			pts_i = np.stack([ self._cell_centers[i] for i in A.row ])
			pts_j = np.stack([ self._cell_centers[j] for j in A.col ])
			ref_d = np.sqrt(np.median(np.sum((pts_i - pts_j)**2, axis=1)))
			for i, j, k in zip(A.row, A.col, A.data):
				if self._adjacency_label[k] == 2: # only in Voronoi
					xi = points[ix == i]
					xj = points[ix == j]
					if 1 < xi.shape[0] and 1 < xj.shape[0]:
						dij = cdist(xi, xj)
						#dij = np.dot(xi, xj.T)
						#xi2 = np.sum(xi * xi, axis=1, keepdims=True)
						#dij -= 0.5 * xi2
						#xj2 = np.sum(xj * xj, axis=1, keepdims=True)
						#dij -= 0.5 * xj2.T
						dij = dij.flatten()
						#kij = dij.argsort()
						dij.sort()
						if ref_d * .9 < dij[0]:
							continue
						try:
							dij = dij[ref]
						except IndexError:
							if 1 < verbose:
								print('skipping edge {:d} between cell {:d} (card = {:d}) and cell {:d} (card = {:d}): number of between-cell pairs = {:d} (expected: {:d})'.format(k, i, xi.shape[0], j, xj.shape[0], dij.size, ref))
							continue
						#dij = np.sqrt(-2.0 * dij)
						if dij < self._min_distance:
							self._adjacency_label[k] = 4 # mark edge as 'not congruent but valid'
							continue
						ci, cj = np.mean(xi, axis=0), np.mean(xj, axis=0)
						yi = np.sort(np.dot(xi - ci, cj - ci))
						yj = np.sort(np.dot(xj - ci, cj - ci))
						# throttle the number of points down to control the p-value
						n0 = 10
						yi = yi[::-1][:min(max(n0, (ref+1)*2), yi.size)]
						yj = yj[:min(max(n0, (ref+1)*2), yj.size)]
						t, p = stats.ttest_ind(yi.T, yj.T, equal_var=False)
						if self.alpha_risk < p:
							self._adjacency_label[k] = 4 # mark edge as 'not congruent but valid'
						continue
						# debug candidate edges
						# comment out the above `continue` statement
						# and uncomment the `argsort` line
						cell_i_k, cell_j_k = np.unravel_index(kij[ref], (xi.shape[0], xj.shape[0]))
						if not hasattr(self, 'candidate_edges'):
							self.candidate_edges = {}
						self.candidate_edges[k] = (xi[cell_i_k], xj[cell_j_k])
						#
					elif 1 < verbose:
						print('skipping edge {:d} between cell {:d} (card = {:d}) and cell {:d} (card = {:d})'.format(k, i, xi.shape[0], j, xj.shape[0]))
		new_labels = np.array([0,-1,-2,1,2])
		# before: 0=[none], 1=not congruent (gas only), 2=not congruent (voronoi only), 
		#         3=congruent, 4=congruent after post-processing (initially voronoi only)
		# after: -2=not congruent (voronoi only), -1=not congruent (gas only), 0=[none], 
		#         1=congruent, 2=congruent after post-processing (initially voronoi only)
		self._adjacency_label = new_labels[self._adjacency_label]
