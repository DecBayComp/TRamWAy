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
from threading import Lock
import itertools
import scipy.sparse as sparse
from tramway.spatial.dichotomy import ConnectedDichotomy


def _face_hash(v1, n1):
	'''key that identify a same face.'''
	return tuple(v1) + tuple(n1)


class KDTreeMesh(Voronoi):
	"""k-dimensional tree (quad tree in 2D) based tesselation.

	Attributes:
		scaler: see :class:`Tesselation`.
		min_probability (float): minimum probability of a point to be in a given cell.
		max_probability (float): maximum probability of a point to be in a given cell.
		max_level (float): maximum level, considering that the smallest cells are at level 0
			and the level increments each time the cell size doubles.
		_min_distance (float, private): scaled minimum distance between neighbor cell centers.
		_avg_distance (float, private): scaled average distance between neighbor cell centers.
	"""
	def __init__(self, scaler=Scaler(), min_distance=None, avg_distance=None, \
		min_probability=None, max_probability=None, max_level=None, **kwargs):
		Voronoi.__init__(self, scaler)
		self._min_distance = min_distance
		self._avg_distance = avg_distance
		self.min_probability = min_probability
		if min_probability and not max_probability:
			max_probability = 10.0 * min_probability
		self.max_probability = max_probability
		self.max_level = max_level

	def cellIndex(self, points, knn=None, prefered='index', \
		min_cell_size=None, metric='chebyshev', **kwargs):
		if isinstance(knn, tuple):
			min_nn, max_nn = knn
		else:
			min_nn, max_nn = knn, None
		if prefered == 'force index':
			min_nn = None
		#if max_nn: # valid only if points are those the tesselation was grown with
		#	max_cell_size = int(floor(self.max_probability * points.shape[0]))
		#	if max_nn < max_cell_size:
		#		max_nn = None
		if metric == 'chebyshev':
			if min_nn or max_nn:
				raise NotImplementedError('knn support has evolved and KDTreeMesh still lacks a proper support for it. You can still call cellIndex with argument metric=\'euclidean\'')
			# TODO: pass relevant kwargs to cdist
			points = self.scaler.scalePoint(points, inplace=False)
			D = cdist(self.descriptors(points, asarray=True), \
				self._cell_centers, metric) # , **kwargs
			dmax = self.dichotomy.reference_length[self.level[np.newaxis,:] + 1]
			I, J = np.nonzero(D <= dmax)
			if min_cell_size:
				K, count = np.unique(J, return_counts=True)
				K = K[count < min_cell_size]
				for k in K:
					J[J == k] = -1
			if I[0] == 0 and I.size == points.shape[0] and I[-1] == points.shape[0] - 1:
				return J
			else:
				K = -np.ones(points.shape[0], dtype=J.dtype)
				K[I] = J
				return K
		else:
			return Delaunay.cellIndex(self, points, knn=knn, prefered=prefered, \
				min_cell_size=min_cell_size, metric=metric, **kwargs)

	def tesselate(self, points, **kwargs):
		init = self.scaler.init
		points = self._preprocess(points)
		if init:
			if self._min_distance:
				self._min_distance = self.scaler.scaleDistance(self._min_distance)
			if self._avg_distance:
				self._avg_distance = self.scaler.scaleDistance(self._avg_distance)
		min_distance = None
		avg_distance = None
		if self._avg_distance:
			avg_distance = self._avg_distance
		elif self._min_distance:
			min_distance = self._min_distance
		min_count = int(round(self.min_probability * points.shape[0]))
		max_count = int(round(self.max_probability * points.shape[0]))
		self.dichotomy = ConnectedDichotomy(self.descriptors(points, asarray=True), \
			min_count=min_count, max_count=max_count, \
			min_edge=min_distance, base_edge=avg_distance, max_level=self.max_level)
		del self.max_level
		self.dichotomy.split()
		self.dichotomy.subset = {} # clear memory
		self.dichotomy.subset_counter = 0
		
		origin, level = zip(*[ self.dichotomy.cell[c][:2] for c in range(self.dichotomy.cell_counter) ])
		origin = np.vstack(origin)
		self.level = np.array(level)
		self._cell_centers = origin + self.dichotomy.reference_length[self.level[:, np.newaxis] + 1]
		adjacency = np.vstack([ self.dichotomy.adjacency[e] for e in range(self.dichotomy.edge_counter) \
			if e in self.dichotomy.adjacency ]) # not symmetric
		#print(np.sum(np.all(adjacency==np.fliplr(adjacency[0][np.newaxis,:])))) # displays 0
		nedges = adjacency.shape[0]
		self._cell_adjacency = sparse.csr_matrix((np.tile(np.arange(nedges), 2), \
				(adjacency.flatten('F'), np.fliplr(adjacency).flatten('F'))), \
			shape=(self.dichotomy.cell_counter, self.dichotomy.cell_counter))
		# (adjacency[i,0], adjacency[i,1], i)

		# quick and dirty Voronoi construction: vertices are introduced as many times as they
		# appear in a ridge, and as many ridges are introduced as four times the number of 
		# centers. In a second step, duplicate vertices are removed.
		def unique_rows(data, *args, **kwargs):
			uniq = np.unique(data.view(data.dtype.descr * data.shape[1]), *args, **kwargs)
			if isinstance(uniq, tuple):
				return (uniq[0].view(data.dtype).reshape(-1, data.shape[1]),) + tuple(uniq[1:])
			else:
				return uniq.view(data.dtype).reshape(-1, data.shape[1])
		n = origin.shape[0]
		self._vertices = []
		ridge_vertices = []
		for i, v1 in enumerate(self.dichotomy.unit_hypercube):
			self._vertices.append(origin + \
				np.float_(v1) * self.dichotomy.reference_length[self.level[:, np.newaxis]])
			for jj, v2 in enumerate(self.dichotomy.unit_hypercube[i+1:]):
				if np.sum(v1 != v2) == 1: # neighbors in the voronoi
					j = i + 1 + jj
					ridge_vertices.append(np.vstack(\
						np.hstack((np.arange(i * n, (i+1) * n)[:,np.newaxis], \
							np.arange(j * n, (j+1) * n)[:,np.newaxis]))))
		self._vertices, I = unique_rows(np.concatenate(self._vertices, axis=0), \
			return_inverse=True)
		ridge_vertices = I[np.concatenate(ridge_vertices, axis=0)]
		u, v = ridge_vertices.T
		nverts = self._vertices.shape[0]
		self._vertex_adjacency = sparse.coo_matrix(\
			(np.ones(2*u.size, dtype=bool), (np.r_[u, v], np.r_[v, u])), \
			shape=(nverts, nverts))
		self._cell_vertices = { i: I[i+n*np.arange(self.dichotomy.unit_hypercube.shape[0])] \
			for i in range(n) }
		#self._postprocess()

	def _postprocess(self):
		pass

