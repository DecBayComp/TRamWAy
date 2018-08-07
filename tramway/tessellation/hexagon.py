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

        __slots__ = ('tilt', 'hexagon_radius', 'hexagon_count')

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
                # center
                if isinstance(points, pd.DataFrame):
                        points = points.copy()
                points = self._preprocess(points)
                #points = self.descriptors(points, asarray=True)
                mean_kwargs = {}
                if not isinstance(points, pd.DataFrame):
                        mean_kwargs['keepdims'] = True
                center = self.descriptors(points).mean(axis=0, **mean_kwargs)
                points -= center
                # rotate
                theta = self.tilt * pi / 6.
                s, c = sin(-theta), cos(-theta)
                rot = np.array([[c, -s], [s, c]])
                if isinstance(points, pd.DataFrame):
                        pts = np.dot(points[self.scaler.columns].values, rot.T)
                        points.loc[:,self.scaler.columns] = pts
                        center = center.values
                else:
                        points = pts = np.dot(points, rot.T)
                #
                lower_bound = pts.min(axis=0)
                upper_bound = pts.max(axis=0)
                size = upper_bound - lower_bound
                hex_sin, hex_cos = sin(pi/6.), cos(pi/6.)
                if self.avg_probability:
                        desired_n_cells = n_cells = 1. / self.avg_probability
                        total_area = np.prod(size)
                        n_cells_prev, hex_radius_prev = [], []
                        while True:
                                hex_radius = np.sqrt(total_area / (2. * sqrt(3.) * n_cells))
                                if self.min_distance is not None:
                                        hex_radius = max(hex_radius, .5 * self.min_distance)
                                if any(np.isclose(hex_radius, r) for r in hex_radius_prev):
                                        break
                                hex_side = hex_radius / hex_cos
                                dx = 2. * hex_radius
                                dy = 1.5 * hex_side# * (2. - hex_sin)
                                m = int(ceil(size[0] / dx))
                                n = int(ceil((size[1] + 2. * (hex_side * hex_sin - hex_radius)) / dy) + 1)
                                lower_center_x = -.5 * float(m - 1) * dx
                                lower_center_y = -.5 * float(n - 1) * dy
                                centers = []
                                for k in range(n):
                                        _centers_x = lower_center_x + float(k % 2) * hex_radius + dx * np.arange(m + 1 - (k % 2))
                                        _centers_y = np.full_like(_centers_x, lower_center_y + float(k) * dy)
                                        centers.append(np.stack((_centers_x, _centers_y), axis=-1))
                                self._cell_centers = np.vstack(centers)
                                I = np.unique(self.cell_index(points))
                                assert np.all(0 <= I)
                                nc, n_cells = n_cells, I.size
                                if n_cells in n_cells_prev:
                                        break
                                #print((hex_radius, nc, m*n, n_cells, n_cells_prev))
                                n_cells_prev.append(n_cells)
                                hex_radius_prev.append(hex_radius)
                                n_cells = round(desired_n_cells * nc / float(n_cells))
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
                        lower_center_x = -.5 * float(m - 1) * dx
                        lower_center_y = -.5 * float(n - 1) * dy
                        centers = []
                        for k in range(n):
                                _centers_x = lower_center_x + float(k % 2) * hex_radius + dx * np.arange(m + 1 - (k % 2))
                                _centers_y = np.full_like(_centers_x, lower_center_y + float(k) * dy)
                                centers.append(np.stack((_centers_x, _centers_y), axis=-1))
                        self._cell_centers = np.vstack(centers)
                # rotate/center the centers back into the original space
                s, c = sin(theta), cos(theta)
                rot = np.array([[c, -s], [s, c]])
                self._cell_centers = np.dot(self._cell_centers, rot.T) + center
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
                        i = 0
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

