# -*- coding: utf-8 -*-

# Copyright © 2018-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
import tramway.inference.gradient as grad
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
        try:
            self.center_t = rt[t_pos].tolist()
        except IndexError:
            raise RuntimeError('missing time column; was the tessellation made with defined time_dimension?')
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
        boundary=None, spatial_adjacency=None, temporal_adjacency=None, time_adjacency=None):
        Distributed.__init__(self, cells, adjacency, index, center, span, central, boundary)
        if time_adjacency is None:
            time_adjacency = temporal_adjacency
        if time_adjacency is None and spatial_adjacency is None and \
                adjacency.dtype not in (bool, np.bool_):
            # separate spatial adjacency and temporal adjacency
            A = adjacency.tocoo()
            ok = 1 < A.data
            time_adjacency = sparse.csr_matrix((np.ones(np.sum(ok), dtype=bool), (A.row[ok], A.col[ok])), shape=A.shape)
            ok = A.data == 1
            spatial_adjacency = sparse.csr_matrix((np.ones(np.sum(ok), dtype=bool), (A.row[ok], A.col[ok])), shape=A.shape)
        self.spatial_adjacency = spatial_adjacency
        self.time_adjacency = time_adjacency

    @property
    def temporal_adjacency(self):
        """
        alias for :attr:`time_adjacency`.
        """
        return self.time_adjacency

    def neighbours(self, i):
        """
        Indices of spatial-neighbour cells.

        Argument:

            i (int): cell index.

        Returns:

            numpy.ndarray: indices of the neighbour cells of cell *i*.

        """
        return self.spatial_adjacency.indices[self.spatial_adjacency.indptr[i]:self.spatial_adjacency.indptr[i+1]]

    def time_neighbours(self, i):
        """
        Indices of time-neighbour cells.

        Argument:

            i (int): cell index.

        Returns:

            numpy.ndarray: indices of the neighbour cells of cell *i*.

        """
        return self.time_adjacency.indices[self.time_adjacency.indptr[i]:self.time_adjacency.indptr[i+1]]

    def time_derivative(self, i, X, index_map=None, na=np.nan, **kwargs):
        cell = self.cells[i]
        t0 = cell.center_t

        # cache neighbours (indices and center locations)
        if not isinstance(cell.cache, dict):
            cell.cache = {}
        try:
            i, adjacent, t = cell.cache['time_derivative']
        except KeyError:
            A = self.time_adjacency
            adjacent = _adjacent = A.indices[A.indptr[i]:A.indptr[i+1]]
            if index_map is not None:
                adjacent = index_map[_adjacent]
                ok = 0 <= adjacent
                assert np.all(ok)
                #adjacent, _adjacent = adjacent[ok], _adjacent[ok]
            if _adjacent.size:
                t = np.array([ self.cells[j].center_t for j in _adjacent ])
                before, after = t<t0, t0<t

                # pre-compute the "t" term
                u, v = before, after
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

                if t is not None:
                    t = (u, v, t)

            else:
                t = None

            if index_map is not None:
                i = index_map[i]
            cell.cache['time_derivative'] = (i, adjacent, t)

        if t is None:
            return None

        x0, x = X[i], X[adjacent]

        # compute the derivative
        u, v, t = t
        #u, v, t= before, after, t term
        if u is None:
            if v is None:
                deriv = na
            else:
                # 1./X = X0[j] - np.mean(X[v,j])
                deriv = (x0 - np.mean(x[v])) * t
        elif v is None:
            deriv = (x0 - np.mean(x[u])) * t
        else:
            deriv = grad._vander(t, np.r_[x0, np.mean(x[u]), np.mean(x[v])])

        return deriv

    def time_variation(self, i, X, index_map=None, na=0., **kwargs):
        cell = self.cells[i]
        t0 = cell.center_t

        # cache neighbours (indices and center locations)
        if not isinstance(cell.cache, dict):
            cell.cache = {}
        try:
            i, adjacent, t = cell.cache['time_derivative']
        except KeyError:
            A = self.time_adjacency
            adjacent = _adjacent = A.indices[A.indptr[i]:A.indptr[i+1]]
            if index_map is not None:
                adjacent = index_map[_adjacent]
                ok = 0 <= adjacent
                assert np.all(ok)
                #adjacent, _adjacent = adjacent[ok], _adjacent[ok]
            if _adjacent.size:
                t = np.array([ self.cells[j].center_t for j in _adjacent ])
                before, after = t<t0, t0<t

                # pre-compute the "t" term
                u, v = before, after
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

                if t is not None:
                    t = (u, v, t)

            else:
                t = None

            if index_map is not None:
                i = index_map[i]
            cell.cache['time_derivative'] = (i, adjacent, t)

        if t is None:
            return None

        x0, x = X[i], X[adjacent]

        # compute the derivative
        u, v, t = t
        #u, v, t= before, after, t term
        if u is None:
            if v is None:
                delta = np.r_[na, na]
            else:
                # 1./X = X0[j] - np.mean(X[v,j])
                delta = np.r_[na, (x0 - np.mean(x[v])) * t]
        elif v is None:
            delta = np.r_[(x0 - np.mean(x[u])) * t, na]
        else:
            t0, tu, tv = t
            delta = np.r_[
                (x0 - np.mean(x[u])) / (t0 - tu),
                (x0 - np.mean(x[v])) / (t0 - tv),
                ]
            #delta = np.abs(delta)

        return delta

    def temporal_variation(self, *args, **kwargs):
        """
        alias for :meth:`time_variation`.
        """
        return self.time_variation(*args, **kwargs)

    def spatial_variation(self, *args, **kwargs):
        """
        alias for :meth:`local_variation`.
        """
        return self.local_variation(*args, **kwargs)


__all__ = ['DynamicTranslocations', 'DynamicCells']

