# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
from tramway.spatial.scaler import *
from math import *
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist


class KMeansMesh(Voronoi):
	"""K-Means and Voronoi based tesselation.

	Attributes:
		avg_probability (float): probability of a point to be in a given cell (controls the
			number of cells and indirectly their size).
	
	Other Attributes:
		_min_distance (float): scaled minimum distance between adjacent cell centers.
	"""
	def __init__(self, scaler=Scaler(), min_probability=None, avg_probability=None, \
		min_distance=None, **kwargs):
		Voronoi.__init__(self, scaler)
		#self.min_probability = min_probability
		#self.max_probability = None
		self.avg_probability = avg_probability
		#self.local_probability = None
		self._min_distance = min_distance

	def _preprocess(self, points):
		init = self.scaler.init
		points = Voronoi._preprocess(self, points)
		if init:
			self._min_distance = self.scaler.scaleDistance(self._min_distance)
		grid = RegularMesh(avg_probability=self.avg_probability)
		grid.tesselate(points)
		self._cell_centers = grid._cell_centers
		self.lower_bound = grid.lower_bound
		self.upper_bound = grid.upper_bound
		self.roi_subset_size = 10000
		self.roi_subset_count = 10
		return points

	def tesselate(self, points, tol=1e-3, prune=True, plot=False, **kwargs):
		points = self._preprocess(points)
		"""Grow the tesselation.

		Attributes:
			points: see :meth:`~tramway.tesselation.base.Tesselation.tesselate`.
			tol (float, optional): error tolerance. 
				Passed as `thresh` to :func:`scipy.cluster.vq.kmeans`.
			prune (bool, optional): prunes the Voronoi and removes the longest edges.
		"""
		self._cell_centers, _ = kmeans(np.asarray(points), self._cell_centers, \
			thresh=tol)

		if False: # one-class SVM based ROIs for pruning; slow and not efficient
			from sklearn.svm import OneClassSVM
			if self.roi_subset_size < points.shape[0]:
				permutation = np.random.permutation(points.shape[0])
				subsets = [ np.asarray(points)[permutation[i*self.roi_subset_size:(i+1)*self.roi_subset_size]] \
					for i in range(min(self.roi_subset_count, floor(points.shape[0]/self.roi_subset_size))) ]
			else:
				subsets = [np.asarray(points)]
			#self._postprocess()
			self.roi = []
			selected_centers = np.zeros(self._cell_centers.shape[0], dtype=bool)
			selected_vertices = np.zeros(self.cell_vertices.shape[0], dtype=bool)
			for subset in subsets:
				roi = OneClassSVM(nu=0.01, kernel='rbf', gamma=1, max_iter=1e5)
				roi.fit(subset)
				selected_centers = np.logical_or(selected_centers, roi.predict(self._cell_centers) == 1)
				selected_vertices = np.logical_or(selected_vertices, roi.predict(self._cell_vertices) == 1)
				self.roi.append(roi)
			self._adjacency_label = np.ones(self._cell_adjacency.data.size, dtype=bool)
			# copy/paste from tesselation.gas
			points = np.asarray(points)
			ix = np.argmin(cdist(points, self._cell_centers), axis=1)
			I = np.repeat(np.arange(self._cell_adjacency.indptr.size - 1), \
				np.diff(self._cell_adjacency.indptr))
			J = self._cell_adjacency.indices
			#
			for edge, ridge in enumerate(self.ridge_vertices):
				v0, v1 = ridge
				if not (selected_vertices[v0] and selected_vertices[v1]):
					if (not selected_vertices[v0]) and (not selected_vertices[v1]):
						# delete link
						self._adjacency_label[edge] = False
					else:
						# check just like tesselation.gas
						xi = points[ix == I[edge]]
						xj = points[ix == J[edge]]
						if xi.size and xj.size:
							dij = np.dot(xi, xj.T)
							xi2 = np.sum(xi * xi, axis=1, keepdims=True)
							dij -= 0.5 * xi2
							xj2 = np.sum(xj * xj, axis=1, keepdims=True)
							dij -= 0.5 * xj2.T
							dij = dij.flatten()
							dij.sort()
							try:
								dij = dij[-5] # 5 hard coded!
							except: # disconnect
								self._adjacency_label[edge] = False
								continue
							dij = np.sqrt(-2.0 * dij)
							if self._min_distance * 5 < dij: # disconnect
								self._adjacency_label[edge] = False
						elif verbose:
							print((edge, I[edge], J[edge], xi.shape, xj.shape))
			if plot:
				if points.shape[1] == 2:
					import matplotlib.pyplot as plt
					if isinstance(self.lower_bound, pd.DataFrame):
						x_ = 'x'
						y_ = 'y'
					else:
						x_ = 0
						y_ = 1
					xx, yy = np.meshgrid(np.linspace(self.lower_bound[x_], self.upper_bound[x_], 500), \
						np.linspace(self.lower_bound[y_], self.upper_bound[y_], 500))
					zz = self.roi[0].decision_function(np.c_[xx.ravel(), yy.ravel()])
					for roi in self.roi[1:]:
						zz = np.maximum(zz, roi.decision_function(np.c_[xx.ravel(), yy.ravel()]))
					zz = zz.reshape(xx.shape)
					subset = np.concatenate(subsets, axis=0)
					plt.plot(subset[:,0], subset[:,1], 'k.', markersize=8)
					plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors='darkred')
					plt.scatter(self._cell_centers[selected_centers,0], self._cell_centers[selected_centers,1], c='blueviolet', s=40)
					plt.scatter(self._cell_centers[np.logical_not(selected_centers),0], self._cell_centers[np.logical_not(selected_centers),1], c='gold', s=40)
					plt.axis('equal')
					plt.show()
				else:
					raise AttributeError('can plot only 2D data')

		if prune: # inter-center-distance-based pruning
			A = sparse.tril(self.cell_adjacency, format='coo')
			i, j, k = A.row, A.col, A.data
			if self._adjacency_label is None:
				self._adjacency_label = np.ones(np.max(k)+1, dtype=bool)
			else:
				l = 0 < self._adjacency_label[k]
				i, j, k = i[l], j[l], k[l]
			x = self._cell_centers
			d = x[i] - x[j]
			d = np.sum(d * d, axis=1) # square distance
			d0 = np.median(d)
			edge = k[d0 * 2.5 < d] # edges to be discarded; 2.5 is empirical
			if edge.size:
				self._adjacency_label[edge] = False

