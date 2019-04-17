# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..base import *
from tramway.core.scaler import *
from math import *
import numpy as np
import pandas as pd
from threading import Lock
import itertools
import scipy.sparse as sparse
from scipy.spatial.distance import cdist
from .dichotomy import ConnectedDichotomy
from collections import OrderedDict


def _face_hash(v1, n1):
    '''key that identify a same face.'''
    return tuple(v1) + tuple(n1)


class KDTreeMesh(Voronoi):
    """k-dimensional tree (quad tree in 2D) based tessellation.

    Attributes:
        scaler: see :class:`~tramway.tessellation.base.Tessellation`.
        min_probability (float): minimum probability of a point to be in a given cell.
        max_probability (float): maximum probability of a point to be in a given cell.
        max_level (float): maximum level, considering that the smallest cells are at level 0
            and the level increments each time the cell size doubles.
        _min_distance (float, private): scaled minimum distance between neighbor cell centers.
        _avg_distance (float, private): scaled average distance between neighbor cell centers.
    """
    def __init__(self, scaler=None, min_distance=None, avg_distance=None, \
        min_probability=None, max_probability=None, max_level=None, **kwargs):
        Voronoi.__init__(self, scaler)
        self._min_distance = min_distance
        self._avg_distance = avg_distance
        self.min_probability = min_probability
        if min_probability and not max_probability:
            max_probability = 10.0 * min_probability
        self.max_probability = max_probability
        self.max_level = max_level

    def cell_index(self, points, format=None, select=None, knn=None,
        min_location_count=None, metric='chebyshev', filter=None,
        filter_descriptors_only=False, **kwargs):
        if isinstance(knn, tuple):
            min_nn, max_nn = knn
        else:
            min_nn, max_nn = knn, None
        if format == 'force array':
            min_nn = None
        #if max_nn: # valid only if points are those the tessellation was grown with
        #       max_location_count = int(floor(self.max_probability * points.shape[0]))
        #       if max_nn < max_location_count:
        #           max_nn = None
        if metric == 'chebyshev':
            if min_nn or max_nn:
                raise NotImplementedError('knn support has evolved and KDTreeMesh still lacks a proper support for it. You can still call cell_index with argument metric=\'euclidean\'')
            # TODO: pass relevant kwargs to cdist
            points = self.scaler.scale_point(points, inplace=False)
            X = self.descriptors(points, asarray=True)
            D = cdist(X, self._cell_centers, metric) # , **kwargs
            I, J = np.nonzero(D <= self.reference_length)
            if min_location_count or filter is not None:
                K, count = np.unique(J, return_counts=True)
                if min_location_count:
                    for k in K[count < min_location_count]:
                        J[J == k] = -1
                if filter is not None:
                    if min_location_count:
                        K = K[min_location_count <= K]
                    for k in K:
                        cell = J == k
                        if filter_descriptors_only:
                            x = X[cell]
                        else:
                            x = points[cell]
                        if not filter(self, k, x):
                            J[cell] = -1

            if I[0] == 0 and I.size == points.shape[0] and I[-1] == points.shape[0] - 1:
                K = J
            else:
                K = -np.ones(points.shape[0], dtype=J.dtype)
                K[I] = J
            if format == 'force array':
                format = 'array' # for :func:`format_cell_index`
            return format_cell_index(K, format=format, select=select,
                shape=(points.shape[0], self.cell_adjacency.shape[0]))
        else:
            return Delaunay.cell_index(self, points, format=format, select=select, knn=knn,
                min_location_count=min_location_count, metric=metric, **kwargs)

    def tessellate(self, points, **kwargs):
        init = self.scaler.init
        points = self._preprocess(points)
        if init:
            if self._min_distance is not None:
                self._min_distance = self.scaler.scale_distance(self._min_distance)
            if self._avg_distance is not None:
                self._avg_distance = self.scaler.scale_distance(self._avg_distance)
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
        level = np.array(level)
        self._cell_centers = origin + self.dichotomy.reference_length[level[:, np.newaxis] + 1]
        adjacency = np.vstack([ self.dichotomy.adjacency[e] for e in range(self.dichotomy.edge_counter) \
            if e in self.dichotomy.adjacency ]) # not symmetric
        #print(np.sum(np.all(adjacency==np.fliplr(adjacency[0][np.newaxis,:])))) # displays 0
        nedges = adjacency.shape[0]
        self.cell_adjacency = sparse.csr_matrix((np.tile(np.arange(nedges), 2), \
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
        vertices = []
        ridge_vertices = []
        for i, v1 in enumerate(self.dichotomy.unit_hypercube):
            vertices.append(origin + \
                np.float_(v1) * self.dichotomy.reference_length[level[:, np.newaxis]])
            for jj, v2 in enumerate(self.dichotomy.unit_hypercube[i+1:]):
                if np.sum(v1 != v2) == 1: # neighbors in the voronoi
                    j = i + 1 + jj
                    ridge_vertices.append(np.vstack(\
                        np.hstack((np.arange(i * n, (i+1) * n)[:,np.newaxis], \
                            np.arange(j * n, (j+1) * n)[:,np.newaxis]))))
        vertices, I = unique_rows(np.concatenate(vertices, axis=0), \
            return_inverse=True)
        self._vertices = vertices
        self._lazy['vertices'] = False
        ridge_vertices = I[np.concatenate(ridge_vertices, axis=0)]
        u, v = ridge_vertices.T
        nverts = vertices.shape[0]
        self.vertex_adjacency = sparse.coo_matrix(\
            (np.ones(2*u.size, dtype=bool), (np.r_[u, v], np.r_[v, u])), \
            shape=(nverts, nverts))
        self.cell_vertices = { i: I[i+n*np.arange(self.dichotomy.unit_hypercube.shape[0])] \
            for i in range(n) }
        # for `cell_index` even after call to `freeze`
        self.reference_length = self.dichotomy.reference_length[level[np.newaxis,:] + 1]
        #self._postprocess()

    def _postprocess(self):
        pass

    def freeze(self):
        self.dichotomy = None

    def delete_cell(self, i, adjacency_label=True, metric='chebyshev', pack_indices=True):
        Voronoi.delete_cell(self, i, adjacency_label, metric, pack_indices)


setup = {
    'make': KDTreeMesh,
    'make_arguments': OrderedDict((
        ('min_distance', ()),
        ('avg_distance', ()),
        ('min_probability', ()),
        ('max_probability', ()),
        ('max_location_count', ('-S', dict(type=int, help='maximum number of locations per cell'))),
        ('max_level', ('-ll', '--lower-levels', dict(type=int, help='number of levels below the smallest one', metavar='LOWER_LEVELS'))),
        )),
    }

__all__ = ['KDTreeMesh', 'setup']

