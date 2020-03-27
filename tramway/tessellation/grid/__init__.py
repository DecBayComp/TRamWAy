# -*- coding: utf-8 -*-

# Copyright © 2017-2018, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import *
from tramway.core import *
from ..base import Voronoi, sparse_to_dict
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import itertools
from collections import Counter, OrderedDict


class RegularMesh(Voronoi):
    """Regular k-D grid.

    Attributes:
        lower_bound (pandas.Series or numpy.ndarray): (scaled)
        upper_bound (pandas.Series or numpy.ndarray): (scaled)
        count_per_dim (list of ints):
        min_probability (float): (not used)
        avg_probability (float):
        max_probability (float): (not used)
        min_distance (float):
            minimum distance between adjacent cell centers;
            ignored if `avg_distance` is defined.
        avg_distance (float):
            average distance between adjacent cell centers;
            ignored if `avg_probability` is defined.

    """

    __lazy__ = Voronoi.__lazy__ + ('diagonal_adjacency',)

    def __init__(self, scaler=None, lower_bound=None, upper_bound=None, count_per_dim=None, \
        min_probability=None, max_probability=None, avg_probability=None, \
        min_distance=None, avg_distance=None, **kwargs):
        Voronoi.__init__(self, scaler)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.count_per_dim = count_per_dim
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.avg_probability = avg_probability
        self.min_distance = min_distance
        self.avg_distance = avg_distance
        self._diagonal_adjacency = None

    def tessellate(self, points, **kwargs):
        points = self._preprocess(points)
        if self.lower_bound is None:
            self.lower_bound = points.min(axis=0)
        elif isinstance(points, pd.DataFrame) and not isinstance(self.lower_bound, pd.Series):
            self.lower_bound = pd.Series(self.lower_bound, index=points.columns)
        if self.upper_bound is None:
            self.upper_bound = points.max(axis=0)
        elif isinstance(points, pd.DataFrame) and not isinstance(self.upper_bound, pd.Series):
            self.upper_bound = pd.Series(self.upper_bound, index=points.columns)
        if self.count_per_dim is None:
            size = self.upper_bound - self.lower_bound
            if self.avg_probability:
                n_cells = 1.0 / self.avg_probability
                increment = exp(log(np.asarray(size).prod() / n_cells) / points.shape[1])
                if self.min_distance is not None:
                    increment = max(increment, self.min_distance)
            elif self.avg_distance:
                increment = self.avg_distance
                if self.min_probability is not None:
                    # TODO
                    pass
            else:
                raise ValueError('both `avg_probability` and `avg_distance` undefined')
            if isinstance(size, pd.Series):
                self.count_per_dim = pd.Series.round(size / increment)
            else:
                self.count_per_dim = np.round(size / increment)
        elif isinstance(points, pd.DataFrame) and not isinstance(self.count_per_dim, pd.Series):
            self.count_per_dim = pd.Series(self.count_per_dim, index=points.columns)
        _linspace = lambda _start, _stop, _nsteps: np.linspace(_start, _stop, int(_nsteps))
        if isinstance(points, pd.DataFrame):
            grid = pd.concat([self.lower_bound, self.upper_bound, self.count_per_dim + 1], axis=1).T
            self.grid = [ _linspace(*grid[col].values) for col in grid ]
        else:
            grid = np.stack((self.lower_bound, self.upper_bound, self.count_per_dim + 1), axis=0)
            self.grid = [ _linspace(col[0], col[1], col[2]) for col in grid.T ]
        cs = np.meshgrid(*[ (g[:-1] + g[1:]) / 2 for g in self.grid ], indexing='ij')
        self._cell_centers = np.column_stack([ c.flatten() for c in cs ])

    def _preprocess(self, points):
        Voronoi._preprocess(self, points) # initialize `scaler`
        return points # ... but do not scale

    def _postprocess(self):
        pass

    # cell_centers property
    @property
    def cell_centers(self):
        return self._cell_centers

    @cell_centers.setter
    def cell_centers(self, centers):
        self._cell_centers = centers

    # vertices property
    @property
    def vertices(self):
        if self._vertices is None:
            vs = np.meshgrid(*self.grid, indexing='ij')
            self._vertices = np.column_stack([ v.flatten() for v in vs ])
        return self.__returnlazy__('vertices', self._vertices)

    @vertices.setter
    def vertices(self, vertices):
        self.__lazysetter__(vertices)

    # cell_adjacency property
    @property
    def cell_adjacency(self):
        if self._cell_adjacency is None:
            cix = np.meshgrid(*[ np.arange(0, len(g) - 1) for g in self.grid ], \
                indexing='ij')
            cix = np.column_stack([ g.flatten() for g in cix ])
            c2  = np.atleast_2d(np.sum(cix * cix, axis=1))
            self._cell_adjacency = sparse.csr_matrix(\
                np.abs(c2 + c2.T - 2 * np.dot(cix, cix.T) - 1.0) < 1e-6)
        return self.__returnlazy__('cell_adjacency', self._cell_adjacency)

    @cell_adjacency.setter # copy/paste
    def cell_adjacency(self, matrix):
        self.__lazysetter__(matrix)

    # vertex_adjacency property
    @property
    def vertex_adjacency(self):
        if self._vertex_adjacency is None:
            vix = np.meshgrid(*[ np.arange(0, len(g)) for g in self.grid ], \
                indexing='ij')
            vix = np.column_stack([ g.flatten() for g in vix ])
            v2  = np.atleast_2d(np.sum(vix * vix, axis=1))
            self._vertex_adjacency = sparse.csr_matrix(\
                np.abs(v2 + v2.T - 2 * np.dot(vix, vix.T) - 1.0) < 1e-6)
        return self.__returnlazy__('vertex_adjacency', self._vertex_adjacency)

    @vertex_adjacency.setter # copy/paste
    def vertex_adjacency(self, matrix):
        self.__lazysetter__(matrix)

    # cell_vertices property
    @property
    def cell_vertices(self):
        if self._cell_vertices is None:
            cs, vs = self.cell_centers, self.vertices
            c2 = np.atleast_2d(np.sum(cs * cs, axis=1))
            v2 = np.atleast_2d(np.sum(vs * vs, axis=1))
            d2 = np.sum((cs[0] - vs[0]) ** 2)
            self._cell_vertices = sparse.dok_matrix(\
                np.abs(c2.T + v2 - 2 * np.dot(cs, vs.T) - d2) < 1e-6)
            #assert self._cell_vertices.tocsr() == dict_to_sparse(sparse_to_dict(self._cell_vertices), shape=self._cell_vertices.shape)
            self._cell_vertices = sparse_to_dict(self._cell_vertices)
        return self.__returnlazy__('cell_vertices', self._cell_vertices)

    @cell_vertices.setter # copy/paste
    def cell_vertices(self, matching):
        self.__lazysetter__(matching)

    @property
    def diagonal_adjacency(self):
        if self._diagonal_adjacency is None:
            A = self.cell_adjacency.tocsr()
            indptr, indices = np.zeros_like(A.indptr), []
            for i in range(A.shape[0]):
                I = A.indices[A.indptr[i]:A.indptr[i+1]].tolist()
                J = Counter()
                for j in I:
                    J.update(A.indices[A.indptr[j]:A.indptr[j+1]])
                I += [ j for j in J if i != j and 1 < J[j] ]
                I.sort()
                indices.append(I)
                indptr[i+1] = len(I)
            indices = np.array(list(itertools.chain(*indices)))
            indptr = np.cumsum(indptr)
            self._diagonal_adjacency = sparse.csr_matrix(
                (np.ones(indices.shape, dtype=bool), indices, indptr),
                shape=A.shape)
        return self.__returnlazy__('diagonal_adjacency', self._diagonal_adjacency)

    @diagonal_adjacency.setter
    def diagonal_adjacency(self, matrix):
        self.__lazysetter__(matrix)

    def contour(self, cell, distance=1, **kwargs):
        """
        See also :func:`tramway.feature.adjacency.contour`.
        """
        #if kwargs.get('fallback', False):
        #       warn("no fallback support", exceptions.MissingSupportWarning)
        if 'dilation_adjacency' not in kwargs:
            kwargs['dilation_adjacency'] = self.diagonal_adjacency.tocsr()
        return Voronoi.contour(self, cell, distance, **kwargs)



setup = {
    'make': RegularMesh,
    'make_arguments': OrderedDict((
        ('min_probability', ()),
        ('avg_probability', ()),
        ('max_probability', ()),
        ('min_distance', ()),
        ('avg_distance', ()),
        # avg_location_count allows to control avg_probability from the commandline
        ('avg_location_count', dict(args=('-c', '--location-count'), kwargs=dict(type=int, default=80, help='average number of locations per cell'), translate=True)),
        )),
    }

__all__ = ['RegularMesh', 'setup']

