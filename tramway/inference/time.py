# -*- coding: utf-8 -*-

# Copyright © 2018, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
import numpy as np
import pandas as pd
import scipy.sparse as sparse


class DynamicTranslocations(Translocations):
        __slots__ = ('center_t',)
        def __init__(self, index, translocations, center=None, span=None, boundary=None):
                Translocations.__init__(self, index, translocations, center, span, boundary)
                rt = self.center
                if isinstance(self.data, pd.DataFrame):
                        t_pos = list(self.data.columns).index(self.time_col)
                else:
                        t_pos = self.time_col
                self.center_t = rt[t_pos].tolist()
                others = np.ones(rt.size, dtype=bool)
                others[t_pos] = False
                self.center_r = rt[others]

        @property
        def center_r(self):
                return self.center
        @center_r.setter
        def center_r(self, r):
                self.center = r


class DynamicCells(Distributed):

        def __init__(self, cells, adjacency, index=None, center=None, span=None, central=None,
                boundary=None, temporal_adjacency=None):
                Distributed.__init__(self, cells, adjacency, index, center, span, central, boundary)
                if temporal_adjacency is None and adjacency.dtype not in (bool, np.bool_):
                        # separate spatial adjacency and temporal adjacency
                        A = adjacency.tocoo()
                        ok = 1 < A.data
                        temporal_adjacency = sparse.csr_matrix((np.ones(np.sum(ok), dtype=bool), (A.row[ok], A.col[ok])), shape=A.shape)
                        spatial_adjacency = adjacency
                        spatial_adjacency.data = spatial_adjacency.data == 1
                        spatial_adjacency.eliminate_zeros()
                        self.spatial_adjacency = spatial_adjacency
                self.temporal_adjacency = temporal_adjacency

        @property
        def spatial_adjacency(self):
                return self.adjacency

        @spatial_adjacency.setter
        def spatial_adjacency(self, matrix):
                self.adjacency = matrix

        def time_derivative(self, i, X, index_map=None, **kwargs):
                cell = cells[i]
                # below, the measurement is renamed y and the coordinates are X
                t0 = cell.center_t

                # cache neighbours (indices and center locations)
                if not isinstance(cell.cache, dict):
                        cell.cache = {}
                try:
                        i, adjacent, t = cell.cache['time_derivative']
                except KeyError:
                        A = cells.temporal_adjacency
                        adjacent = _adjacent = A.indices[A.indptr[i]:A.indptr[i+1]]
                        if index_map is not None:
                                adjacent = index_map[_adjacent]
                                ok = 0 <= adjacent
                                assert np.all(ok)
                                #adjacent, _adjacent = adjacent[ok], _adjacent[ok]
                        if _adjacent.size:
                                t = np.vstack([ cells[j].center_t for j in _adjacent ])
                                before, after = t<t0, t0<t

                                # pre-compute the "t" term
                                u, v = below, above
                                if not np.any(u):
                                        u = None
                                if not np.any(v):
                                        v = None

                                if u is None:
                                        if v is None:
                                                t = None
                                        else:
                                                t = 1. / (t0 - np.mean(t[v]))
                                elif v is None:
                                        t = 1. / (t0 - np.mean(t[u]))
                                else:
                                        t = np.r_[t0, np.mean(t[u]), np.mean(t[v])]


                        else:
                                t = []

                        if index_map is not None:
                                i = index_map[i]
                        cell.cache['time_derivative'] = (i, adjacent, t)

                if not t:
                        return None

                x0, x = X[i], X[adjacent]

                # compute the derivative
                #u, v, t= before, after, t term
                if u is None:
                        if v is None:
                                deriv = 0.
                        else:
                                # 1./X = X0[j] - np.mean(X[v,j])
                                deriv = (x0 - np.mean(x[v])) * t
                elif v is None:
                        deriv = (x0 - np.mean(x[u])) * t
                else:
                        deriv = _vander(t, np.r_[x0, np.mean(x[u]), np.mean(x[v])])

                return np.asarray(deriv)

__all__ = ['DynamicTranslocations', 'DynamicCells']

