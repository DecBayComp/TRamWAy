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
from ..grid import RegularMesh
from tramway.core.scaler import *
from math import *
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from collections import OrderedDict


class KMeansMesh(Voronoi):
        """K-Means and Voronoi based tessellation.

        Attributes:

                avg_probability (float): probability of a point to be in a given cell (controls the
                        number of cells and indirectly their size).
        
        Other Attributes:

                _min_distance (float): scaled minimum distance between adjacent cell centers;
                        *not used*.
        """
        def __init__(self, scaler=None, min_probability=None, avg_probability=None, \
                min_distance=None, initial='grid', **kwargs):
                Voronoi.__init__(self, scaler)
                #self.min_probability = min_probability
                #self.max_probability = None
                self.avg_probability = avg_probability
                #self.local_probability = None
                self._min_distance = min_distance
                self.initial = initial

        def _preprocess(self, points):
                init = self.scaler.init
                points = Voronoi._preprocess(self, points)
                if init and self._min_distance is not None:
                        self._min_distance = self.scaler.scale_distance(self._min_distance)
                if self.initial == 'grid':
                        grid = RegularMesh(avg_probability=self.avg_probability,
                                min_distance=self._min_distance)
                        grid.tessellate(points)
                        self._cell_centers = grid._cell_centers
                        #self.lower_bound = grid.lower_bound
                        #self.upper_bound = grid.upper_bound
                elif self.initial == 'random':
                        lower_bound = np.asarray(points).min(axis=0, keepdims=True)
                        upper_bound = np.asarray(points).max(axis=0, keepdims=True)
                        if self.avg_probability:
                                n_cells = int(round(1. / self.avg_probability))
                        else:
                                raise ValueError('avg_probability (or avg_location_count) not defined')
                        self._cell_centers = np.random.rand(n_cells, points.shape[1])
                        self._cell_centers = self._cell_centers * (upper_bound - lower_bound) + lower_bound
                self.roi_subset_size = 10000
                self.roi_subset_count = 10
                return points

        def tessellate(self, points, tol=1e-6, prune=2.5, plot=False, **kwargs):
                """Grow the tessellation.

                Attributes:
                        points: see :meth:`~tramway.tessellation.base.Tessellation.tessellate`.
                        tol (float): error tolerance. 
                                Passed as `thresh` to :func:`scipy.cluster.vq.kmeans`.
                        prune (bool or float): prunes the Voronoi and removes the edges which length
                                is greater than `prune` times the median edge length;
                                ``True`` is translated to the default value.
                """
                points = self._preprocess(points)
                self._cell_centers, _ = kmeans(np.asarray(points), self._cell_centers, \
                        thresh=tol)

                if prune: # inter-center-distance-based pruning
                        if prune is True: # backward compatibility
                                prune = 2.5 # 2.5 is empirical
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
                        edge = k[d0 * prune < d] # edges to be discarded
                        if edge.size:
                                self._adjacency_label[edge] = False



def _metric(knn=None, **kwargs):
        if isinstance(knn, (tuple, list)):
                knn = knn[0]
        if knn is None:
                return None
        else:
                return 'euclidean'

setup = {
        'make': KMeansMesh,
        'make_arguments': OrderedDict((
                ('min_distance', ()),
                ('avg_probability', ()),
                ('avg_location_count', dict(args=('-c', '--location-count'), kwargs=dict(type=int, default=80, help='average number of locations per cell'), translate=True)),
                ('metric', dict(parse=_metric)),
                )),
        }

__all__ = ['KMeansMesh', 'setup']

