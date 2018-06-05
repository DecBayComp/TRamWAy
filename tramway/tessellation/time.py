# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import *
from tramway.tessellation.base import *
import numpy as np
import pandas as pd
import copy
import scipy.sparse as sparse


class TimeLattice(Tessellation):
        """Proxy `Tessellation` for time lattice expansion.

        If `time_lattice` contains integers, these elements are regarded as frame indices,
        whereas if it contains floats, the elements represent time.

        Functional dependencies:

        * setting `time_lattice` unsets `cell_label`, `cell_adjacency` and `adjacency_label`
        * setting `spatial_mesh` unsets `cell_centers`, `cell_label`, `cell_adjacency` and `adjacency_label`
        * `cell_centers` and `split_frames` are available only when `spatial_mesh` is defined

        """
        __slots__ = ('_spatial_mesh', '_time_lattice', 'time_edge', '_cell_centers')

        __lazy__ = Tessellation.__lazy__ + \
                ('cell_centers', 'cell_adjacency', 'cell_label', 'adjacency_label')

        def __init__(self, scaler=None, segments=None, label=None, mesh=None):
                Tessellation.__init__(self, scaler) # scaler is ignored
                self._time_lattice = segments
                self.time_edge = label
                self._cell_adjacency = None
                self._cell_label = None
                self._adjacency_label = None
                self._spatial_mesh = mesh
                self._cell_centers = None

        @property
        def time_lattice(self):
                return self._time_lattice

        @time_lattice.setter
        def time_lattice(self, segments):
                self.cell_label = None
                self.cell_adjacency = None
                self.adjacency_label = None
                self._time_lattice = segments

        @property
        def spatial_mesh(self):
                return self._spatial_mesh

        @spatial_mesh.setter
        def spatial_mesh(self, tessellation):
                self.cell_label = None
                self.cell_adjacency = None
                self.adjacency_label = None
                self._spatial_mesh = tessellation
                self.cell_centers = None

        def tessellate(self, points, **kwargs):
                if self.spatial_mesh is not None:
                        self.spatial_mesh.tessellate(points, **kwargs)

        def cell_index(self, points, *args, **kwargs):
                exclude = kwargs.pop('exclude_cells_by_location_count', None)
                # extract the timestamps
                time_col = kwargs.pop('time_col', 't')
                if isstructured(points):
                        ts = points[time_col]
                        if isinstance(ts, (pd.Series, pd.DataFrame)):
                                ts = ts.values
                else:
                        ts = points[:,time_col]
                time = self.time_lattice
                nsegments = time.shape[0]
                if time.dtype == int:
                        t0 = ts.min()
                        dt = np.unique(np.diff(np.sort(ts)))
                        if dt[0] == 0:
                                dt = dt[1]
                        else:
                                dt = dt[0]
                        time = (time * dt) + t0
                # 
                if self.spatial_mesh is None:
                        spatial_index = None
                        count_shape = (nsegments,)
                else:
                        spatial_index = self.spatial_mesh.cell_index(points, *args, **kwargs)
                        ncells = self.spatial_mesh.cell_adjacency.shape[0]
                        count_shape = (ncells, nsegments)
                if exclude:
                        location_count = np.zeros(count_shape, dtype=int)
                ps, cs = [], []
                for t in range(nsegments):
                        t0, t1 = time[t]
                        segment = np.logical_and(t0 <= ts, ts < t1)
                        pts, = np.nonzero(segment)
                        if pts.size:
                                if spatial_index is None:
                                        ids = np.full_like(pts, t)
                                else:
                                        if isinstance(spatial_index, np.ndarray):
                                                ids = spatial_index[segment]
                                        else:
                                                raise NotImplementedError
                                        if exclude:
                                                vs, count = np.unique(ids, return_counts=True)
                                                location_count[vs, t] = count
                                        ids += t * ncells
                                #if isinstance(points, DataFrame):
                                #       pts = points.index.values[pts] # NO!
                                ps.append(pts)
                                cs.append(ids)
                if ps:
                        if exclude and not count_shape[1:]:
                                ok = exclude(location_count)
                                ok = ok[0 < location_count]
                                if not np.any(ok):
                                        return ([], [])
                                ps, cs = zip(*[ (p, c) for p, c, b in zip(ps, cs, ok) if b ])
                        ps = np.concatenate(ps)
                        cs = np.concatenate(cs)
                        if exclude and count_shape[1:]:
                                i, t = exclude(location_count).nonzero()
                                ok = np.ones(cs.size, dtype=bool)
                                for c in t * ncells + i:
                                        ok[cs == c] = False
                                ps = ps[ok]
                                cs = cs[ok]
                return (ps, cs)

        # descriptors
        def descriptors(self, points, *args, **kwargs):
                if self.spatial_mesh is None:
                        if args:
                                asarray = args.pop(0)
                        else:
                                asarray = kwargs.pop('asarray', False)
                        time_col = kwargs.pop('time_col', 't')
                        if isstructured(points):
                                timestamps = points[time_col]
                        else:
                                timestamps = points[:,time_col]
                        if asarray:
                                return np.asarray(timestamps)
                        else:
                                return timestamps
                else:
                        return self.spatial_mesh.descriptors(points, *args, **kwargs)

        # cell_adjacency property
        @property
        def cell_adjacency(self):
                if self._cell_adjacency is None:
                        nsegments = self.time_lattice.shape[0]
                        try:
                                past_edge, future_edge = self.time_edge
                        except TypeError:
                                past_edge = future_edge = self.time_edge
                                self.time_edge = (past_edge, future_edge)
                        if self.spatial_mesh is None:
                                cell_ids = np.arange(nsegments)
                                row, col, data = [], [], []
                                labels = [min(past_edge, future_edge) - 1] # fake label at index 0
                                if past_edge is not None:
                                        row.append(cell_ids[:-1])
                                        col.append(cell_ids[1:])
                                        data.append(np.full(nsegments - 1, len(labels), dtype=int))
                                        labels.append(past_edge)
                                if future_edge is not None:
                                        row.append(cell_ids[1:])
                                        col.append(cell_ids[:-1])
                                        data.append(np.full(nsegments - 1, len(labels), dtype=int))
                                        labels.append(future_edge)
                                if data:
                                        data = np.concatenate(data)
                                        row = np.concatenate(row)
                                        col = np.concatenate(col)
                                        self._cell_adjacency = sparse.coo_matrix((data, (row, col)),
                                                shape=(nsegments, nsegments))
                                        self._adjacency_label = np.array(labels)
                                else:
                                        self._cell_adjacency = sparse.coo_matrix((nsegments, nsegments),
                                                dtype=bool)
                                        self._adjacency_label = []
                        else:
                                if self.spatial_mesh.adjacency_label is None:
                                        A = sparse.triu(self.spatial_mesh.cell_adjacency, format='coo')
                                        ncells = A.shape[0]
                                        edge_max = int(A.data.max())
                                        if 1 < edge_max:
                                                raise ValueError('non boolean values in the adjacency matrix are no reference to labels')
                                        n_spatial_edges = A.data.size
                                        A = sparse.coo_matrix((np.tile(np.arange(n_spatial_edges), 2), \
                                                        (np.r_[A.row, A.col], np.r_[A.col, A.row])), \
                                                shape=A.shape).tocsr()
                                        self._adjacency_label = np.ones(n_spatial_edges)
                                else:
                                        self._adjacency_label = self.spatial_mesh.adjacency_label
                                        A = self.spatial_mesh.cell_adjacency.tocsr()
                                        edge_max = int(A.data.max())
                                        n_spatial_edges = int(self._adjacency_label.max())
                                ncells = A.shape[0]
                                active_cells, = np.where(0 < np.diff(A.indptr))
                                edge_ptr = edge_max + 1
                                past = sparse.coo_matrix( \
                                        (np.arange(edge_ptr, edge_ptr + active_cells.size), \
                                                (active_cells, active_cells)), \
                                        shape=(ncells, ncells))
                                edge_ptr += active_cells.size
                                future = sparse.coo_matrix( \
                                        (np.arange(edge_ptr, edge_ptr + active_cells.size), \
                                                (active_cells, active_cells)), \
                                        shape=(ncells, ncells))
                                edge_ptr += active_cells.size
                                blocks = [[A, future] + [None] * (nsegments - 2)]
                                for k in range(1, nsegments - 1):
                                        blocks.append([None] * (k - 1) + [past, A, future] + \
                                                [None] * (nsegments - 2 - k))
                                blocks.append([None] * (nsegments - 2) + [past, A])
                                self._cell_adjacency = sparse.bmat(blocks, format='csr')
                                if past_edge is None:
                                        past_edge = edge_max + 1
                                        edge_max += 1
                                if future_edge is None:
                                        future_edge = edge_max + 1
                                        edge_max += 1
                                dtype = self._adjacency_label.dtype
                                self._adjacency_label = np.r_[self._adjacency_label, \
                                        np.full(active_cells.size, past_edge, dtype=dtype), \
                                        np.full(active_cells.size, future_edge, dtype=dtype)]
                return self.__returnlazy__('cell_adjacency', self._cell_adjacency)

        @cell_adjacency.setter
        def cell_adjacency(self, matrix):
                self.__lazysetter__(matrix)

        # past/future properties
        @property
        def past_edge(self):
                return self.time_edge[0]

        @property
        def future_edge(self):
                return self.time_edge[1]

        # cell_label
        @property
        def cell_label(self):
                if self._cell_label is None:
                        if self.spatial_mesh is None or self.spatial_mesh.cell_label is None:
                                return None
                        else:
                                nsegments = self.time_lattice.shape[0]
                                return np.tile(self.spatial_mesh.cell_label, nsegments)
                else:
                        return self.__returnlazy__('cell_label', self._cell_label)

        @cell_label.setter
        def cell_label(self, label):
                self.__lazysetter__(label)

        # adjacency_label
        @property
        def adjacency_label(self):
                if self._adjacency_label is None:
                        self.cell_adjacency
                return self.__returnlazy__('adjacency_label', self._adjacency_label)

        @adjacency_label.setter
        def adjacency_label(self, label):
                self.__lazysetter__(label)

        ## Delaunay properties and methods
        @property
        def cell_centers(self):
                if self._cell_centers is None:
                        if self.time_lattice.dtype == int:
                                raise ValueError('cannot return timestamps')
                        nsegments = self.time_lattice.shape[0]
                        if self.spatial_mesh is None:
                                raise AttributeError('`cell_centers` is defined only for time lattices combined with spatial tessellations')
                                self._cell_centers = np.mean(self.time_lattice, axis=1) # valid only if `time_lattice` are timestamps!
                        else:
                                self._cell_centers = self.spatial_mesh.cell_centers
                                ncells = self._cell_centers.shape[0]
                                self._cell_centers = np.tile(self._cell_centers, (nsegments, 1))
                                self._cell_centers = np.hstack((self._cell_centers, \
                                        np.repeat(np.mean(self.time_lattice, axis=1), \
                                                ncells)[:,np.newaxis]))
                return self.__returnlazy__('cell_centers', self._cell_centers)

        @cell_centers.setter
        def cell_centers(self, pts):
                if pts is None:
                        self.__lazysetter__(pts)
                elif self.spatial_mesh is None:
                        raise AttributeError('`cell_centers` is defined only for time lattices combined with spatial tessellations')
                else:
                        raise AttributeError('`cell_centers` is read-only')


        def split_frames(self, df, return_times=False):
                if not isinstance(df, pd.DataFrame):
                        raise TypeError('implemented only for `pandas.DataFrame`s')
                if self.spatial_mesh is None:
                        raise NotImplementedError('missing spatial tessellation')
                ncells = self.spatial_mesh.cell_adjacency.shape[0]
                nsegments = self.time_lattice.shape[0]
                try:
                        # not tested yet
                        segment, cell = np.divmod(df.index, ncells) # 1.13.0 <= numpy
                except AttributeError:
                        try:
                                segment = df.index // ncells
                        except TypeError:
                                print(df.index)
                                raise
                        cell = np.mod(df.index, ncells)
                ts = []
                for t in range(nsegments):
                        xt = df[segment == t]
                        xt.index = cell[segment == t]
                        if return_times:
                                if self.time_lattice.dtype == int:
                                        raise ValueError('cannot return timestamps')
                                ts.append((self.time_lattice[t], xt))
                        else:
                                ts.append(xt)
                return ts

        def freeze(self):
                if self.spatial_mesh is not None:
                        self.spatial_mesh.freeze()


def with_time_lattice(cells, frames, exclude_cells_by_location_count=None, **kwargs):
        dynamic_cells = copy.deepcopy(cells)
        dynamic_cells.tessellation = TimeLattice(mesh=cells.tessellation, segments=frames)
        dynamic_cells.cell_index = dynamic_cells.tessellation.cell_index(cells.points, \
                exclude_cells_by_location_count=exclude_cells_by_location_count, **kwargs)
        return dynamic_cells


__all__ = ['TimeLattice', 'with_time_lattice']

