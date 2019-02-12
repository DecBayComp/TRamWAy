# -*- coding: utf-8 -*-

# Copyright © 2018, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import *
from .base import Voronoi
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.optimize import minimize_scalar
from collections import OrderedDict
from warnings import warn


class HexagonalMesh(Voronoi):
    """2D hexagonal mesh.

    Attributes:
        tilt (float):
            angle of the "main" axis, in pi/6 radians;
            0= the main axis is the x-axis; 1= the main axis is the y-axis.
        hexagon_radius (float): hexagon radius.
        hexagon_count (pair of ints):
            number of hexagons along the "main" axis,
            number of hexagons along the perpendical axis.
        min_probability (float): (not used)
        avg_probability (float):
            inverse of the average number of points per cell.
        max_probability (float): (not used)
        min_distance (float):
            minimum distance between adjacent cell centers;
            ignored if `avg_distance` is defined.
        avg_distance (float):
            average distance between adjacent cell centers;
            ignored if `avg_probability` is defined.

    """

    __slots__ = ('tilt', 'hexagon_radius', 'hexagon_count',
        'min_probability', 'avg_probability', 'max_probability',
        'min_distance', 'avg_distance')

    def __init__(self, scaler=None, tilt=0., \
        min_probability=None, max_probability=None, avg_probability=None, \
        min_distance=None, avg_distance=None, **kwargs):
        Voronoi.__init__(self, scaler)
        self.tilt = tilt
        self.min_probability = min_probability
        self.avg_probability = avg_probability
        self.max_probability = max_probability
        self.min_distance = min_distance
        self.avg_distance = avg_distance

    def tessellate(self, points, **kwargs):
        if isinstance(points, pd.DataFrame):
            points = points.copy()
        points = self._preprocess(points)
        # center
        mean_kwargs = {}
        if not isinstance(points, pd.DataFrame):
            mean_kwargs['keepdims'] = True
        center = self.descriptors(points).mean(axis=0, **mean_kwargs)
        points -= center
        if isinstance(points, pd.DataFrame):
            _center = center[self.scaler.columns]
            center = center.values
            pts = points[self.scaler.columns].values
        else:
            pts = points
        # bounds
        lower_bound = kwargs.get('lower_bound', None)
        if lower_bound is None:
            lower_bound = points.min(axis=0)
        elif isinstance(points, pd.DataFrame):
            lower_bound -= _center
        else:
            lower_bound -= center
        upper_bound = kwargs.get('upper_bound', None)
        if upper_bound is None:
            upper_bound = points.max(axis=0)
        elif isinstance(points, pd.DataFrame):
            upper_bound -= _center
        else:
            upper_bound -= center
        # rotate
        if self.tilt:
            theta = self.tilt * pi / 6.
            s, c = sin(-theta), cos(-theta)
            rot = np.array([[c, -s], [s, c]])
            if isinstance(points, pd.DataFrame):
                pts = np.dot(pts, rot.T)
                points.loc[:,self.scaler.columns] = pts
            else:
                points = pts = np.dot(points, rot.T)
        #
        x0, y0 = (lower_bound + upper_bound) * .5
        size = upper_bound - lower_bound
        hex_sin, hex_cos = sin(pi/6.), cos(pi/6.)
        if self.avg_probability:
            desired_n_cells = n_cells = 1. / self.avg_probability
            def err(r):
                if r == 0:
                    return desired_n_cells * desired_n_cells
                dx, dy = 2 * r, 1.5 * r / hex_cos
                m, n = ceil(size[0] / dx), ceil(size[1] / dy) + 1
                e = m * n + ceil(n / 2) - desired_n_cells
                return e * e + r * r
            hex_radius = minimize_scalar(err).x
            if self.min_distance is not None:
                hex_radius = max(hex_radius, .5 * self.min_distance)
            hex_side = hex_radius / hex_cos
            dx = 2. * hex_radius
            dy = 1.5 * hex_side# * (2. - hex_sin)
            m = int(ceil(size[0] / dx))
            n = int(ceil((size[1] + 2. * (hex_side * hex_sin - hex_radius)) / dy) + 1)
            lower_center_x = x0 -.5 * float(m) * dx
            lower_center_y = y0 -.5 * float(n - 1) * dy
            centers = []
            for k in range(n):
                _centers_x = lower_center_x + float(k % 2) * hex_radius + dx * np.arange(m + 1 - (k % 2))
                _centers_y = np.full_like(_centers_x, lower_center_y + float(k) * dy)
                centers.append(np.stack((_centers_x, _centers_y), axis=-1))
            self._cell_centers = np.vstack(centers)
            I = np.unique(self.cell_index(points))
            assert np.all(0 <= I)
            n_cells = I.size
        elif self.avg_distance:
            hex_radius = .5 * self.avg_distance
            if self.min_probability is not None:
                # TODO
                pass
            hex_side = hex_radius / hex_cos
            dx = 2. * hex_radius
            dy = 1.5 * hex_side# * (2. - hex_sin)
            m = int(ceil(size[0] / dx))
            n = int(ceil((size[1] + 2. * (hex_side * hex_sin - hex_radius)) / dy) + 1)
            lower_center_x = x0 -.5 * float(m) * dx
            lower_center_y = y0 -.5 * float(n - 1) * dy
            centers = []
            for k in range(n):
                _centers_x = lower_center_x + float(k % 2) * hex_radius + dx * np.arange(m + 1 - (k % 2))
                _centers_y = np.full_like(_centers_x, lower_center_y + float(k) * dy)
                centers.append(np.stack((_centers_x, _centers_y), axis=-1))
            self._cell_centers = np.vstack(centers)
        else:
            raise ValueError('both `avg_probability` and `avg_distance` undefined')
        # rotate/center the centers back into the original space
        if self.tilt:
            s, c = sin(theta), cos(theta)
            rot = np.array([[c, -s], [s, c]])
            self._cell_centers = np.dot(self._cell_centers, rot.T)
        self._cell_centers += center
        self.hexagon_radius = hex_radius
        self.hexagon_count = (m, n)

    def _preprocess(self, points):
        if points.shape[1] != 2:
            msg = 'the number of dimensions is not 2'
            try:
                points = points[['x', 'y']]
            except:
                raise ValueError(msg)
            else:
                warn(msg, RuntimeWarning)
        Voronoi._preprocess(self, points) # initialize `scaler`
        return points # ... but do not scale

    # cell_centers property
    @property
    def cell_centers(self):
        return self._cell_centers

    @cell_centers.setter
    def cell_centers(self, centers):
        self._cell_centers = centers

    # cell_adjacency property
    @property
    def cell_adjacency(self):
        if self._cell_adjacency is None:
            n_cells = self._cell_centers.shape[0]
            m, n = self.hexagon_count
            A = sparse.dok_matrix((n_cells, n_cells), dtype=bool)
            k = i = 0
            for k in range(n-1):
                m_no_more = k % 2
                A[i, i+1] = True
                A[i, i+m+1] = True
                if m_no_more:
                    A[i, i+m] = True
                i += 1
                for j in range(1, m - m_no_more):
                    A[i, i+1] = True
                    A[i, i+m+1] = True
                    A[i, i+m] = True
                    i += 1
                if m_no_more:
                    A[i, i+m+1] = True
                A[i, i+m] = True
                i += 1
            k += 1
            for j in range(m - (k % 2)):
                A[i, i+1] = True
                i += 1
            assert A.shape == (n_cells, n_cells)
            self._cell_adjacency = (A + A.T).tocsr()
        return self.__returnlazy__('cell_adjacency', self._cell_adjacency)

    @cell_adjacency.setter # copy/paste
    def cell_adjacency(self, matrix):
        self.__lazysetter__(matrix)

    def _postprocess(self):
        if self._cell_centers is None:
            raise NameError('`cell_centers` not defined; tessellation has not been grown yet')
        s, c = sin(pi/6.), cos(pi/6.)
        dr = np.array([[0., -1.], [-c, -s], [-c, s], [c, -s], [c, s], [0., 1.]])
        if self.tilt:
            theta = self.tilt * pi / 6.
            _s, _c = sin(theta), cos(theta)
            rot = np.array([[_c, -_s], [_s, _c]])
            dr = np.dot(dr, rot.T)
        dr *= self.hexagon_radius / c
        #
        n_cells = self._cell_centers.shape[0]
        m, n = self.hexagon_count
        C = self._cell_centers
        _v = np.full((n_cells, 6), -1, dtype=int)
        # bottom left hexagon
        i = 0
        V = [C[i]+dr] # full hexagon
        _v[i] = np.arange(6)
        A = [(0,1), (0,3), (1,2), (2,5), (3,4), (4,5)]
        v = 6
        # vertices are numbered the following way:
        #
        #     5
        #  2     4
        #  1     3
        #     0
        #
        for j in range(1, m+1):
            i += 1
            V.append(C[i]+dr[[0,3,4,5]])
            _v[i, 1] = v - 3
            _v[i, 2] = v - 2
            _v[i, [0,3,4,5]] = np.arange(v, v+4)
            A += [(v-3,v), (v-2,v+3), (v,v+1), (v+1,v+2), (v+2,v+3)]
            v += 4
        i_prev = -1 # index of the bottom right neighbour (should be 0 after first increment)
        for k in range(1, n):
            m_plus = 1 - (k % 2)
            i += 1
            i_prev += 1
            # i_prev is the first (left-most) cell of the previous row
            if m_plus:
                # i_prev is already the bottom right neighbour
                V.append(C[i]+dr[[1,2,4,5]])
                _v[i, 0] = _v[i_prev, 2]
                _v[i, 3] = _v[i_prev, 5]
                _v[i, [1,2,4,5]] = np.arange(v, v+4)
                A += [
                    # bottom right 2 (=0) -> 1
                    (_v[i_prev,2],v),
                    # bottom right 5 (=3) -> 4
                    (_v[i_prev,5],v+2),
                    # 1 -> 2
                    (v,v+1),
                    # 2 -> 5
                    (v+1,v+3),
                    # 4 -> 5
                    (v+2,v+3),
                    ]
                v += 4
            else:
                # i_prev points to the bottom left neighbour
                i_prev += 1 # now bottom right
                V.append(C[i]+dr[[2,4,5]])
                _v[i, 0] = _v[i_prev, 2]
                _v[i, 1] = _v[i_prev-1, 5]
                _v[i, 3] = _v[i_prev, 5]
                _v[i, [2,4,5]] = np.arange(v, v+3)
                A += [
                    # bottom left 5 (=1) -> 2
                    (_v[i_prev-1,5],v),
                    # bottom right 5 (=3) -> 4
                    (_v[i_prev,5],v+1),
                    # 2 -> 5
                    (v,v+2),
                    # 4 -> 5
                    (v+1,v+2),
                    ]
                v += 3
            for j in range(1, m):
                i += 1
                i_prev += 1
                # j stops at m => there is always a bottom right neighbour => vertex 3 already exists
                V.append(C[i]+dr[[4,5]])
                _v[i, 0] = _v[i_prev, 2]
                _v[i, 1] = _v[i-1, 3] # =v-3
                _v[i, 2] = _v[i-1, 4] # =v-2
                _v[i, 3] = _v[i_prev, 5]
                _v[i, [4,5]] = np.arange(v, v+2)
                A += [
                    # bottom right 5 (=3) -> 4
                    (_v[i_prev,5],v),
                    # left 4 (=2) -> 5
                    (v-2,v+1),
                    # 4 -> 5
                    (v,v+1),
                    ]
                v += 2
            if m_plus:
                i += 1
                # there is no bottom right neighbour
                # let i_prev point to the bottom left neighbour
                V.append(C[i]+dr[[3,4,5]])
                _v[i, 0] = _v[i_prev, 4]
                _v[i, 1] = _v[i-1, 3] # =v-3
                _v[i, 2] = _v[i-1, 4] # =v-2
                _v[i, [3,4,5]] = np.arange(v, v+3)
                A += [
                    # bottom left 4 (=0) -> 3
                    (_v[i_prev,4],v),
                    # left 4 (=2) -> 5
                    (v-2,v+2),
                    # 3 -> 4
                    (v,v+1),
                    # 4 -> 5
                    (v+1,v+2),
                    ]
                v += 3
        self._vertices = np.vstack(V)
        n_vertices = self._vertices.shape[0]
        i, j = zip(*A)
        i, j = np.array(i+j), np.array(j+i)
        self._vertex_adjacency = sparse.csr_matrix(
            (np.ones(i.size, dtype=bool), (i, j)),
            shape=(n_vertices, n_vertices))
        self._cell_vertices = { i: v[0<=v] for i, v in enumerate(_v) }

    # vertices property
    @property
    def vertices(self):
        if self._vertices is None and self._cell_centers is not None:
            self._postprocess()
        return self.__returnlazy__('vertices', self._vertices) # no scaling

    @vertices.setter # copy/paste
    def vertices(self, vertices):
        self.__lazysetter__(vertices)

    # cell_volume property
    @property
    def cell_volume(self):
        if self._cell_volume is None:
            n_cells = self._cell_centers.shape[0]
            cell_area = 3 * self.hexagon_radius ** 2 / cos(pi/6.)
            self._cell_volume = np.full(n_cells, cell_area, dtype=float)
        return self._cell_volume

    @cell_volume.setter
    def cell_volume(self, area):
        self.__setlazy__('cell_volume', area)




setup = {
    'make_arguments': OrderedDict((
        ('tilt', dict(type=float, default=0., help='angle of a main axis, in pi/6 radians; from: 0= a main axis is the x-axis; to: 1= a main axis is the y-axis')),
        ('min_probability', ()),
        ('avg_probability', ()),
        ('max_probability', ()),
        ('min_distance', ()),
        ('avg_distance', ()),
        # avg_location_count allows to control avg_probability from the commandline
        ('avg_location_count', dict(args=('-c', '--location-count'), kwargs=dict(type=int, default=80, help='average number of locations per cell'), translate=True)),
        )),
    }


__all__ = ['setup', 'HexagonalMesh']

import sys
if sys.version_info[0] < 3:

    import rwa
    import tramway.core.hdf5 as rules
    hexagonal_mesh_exposes = rules.voronoi_exposes + list(HexagonalMesh.__slots__)
    rwa.hdf5_storable(
        rwa.default_storable(HexagonalMesh, exposes=hexagonal_mesh_exposes),
        agnostic=True)

    __all__.append('hexagonal_mesh_exposes')

