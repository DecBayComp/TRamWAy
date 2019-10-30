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
from collections import OrderedDict
import numpy as np
import pandas as pd


setup = {
    'make_arguments': OrderedDict((
        ('cell_count', dict(type=int, help='number of cells')),
        ('lower_bound', dict(type=float, nargs='+', help='space-separated lower coordinates: x_min y_min ...')),
        ('upper_bound', dict(type=float, nargs='+', help='space-separated upper coordinates: x_max y_max ...')),
        ('avg_probability', ()),
        # avg_location_count allows to control avg_probability from the commandline
        ('avg_location_count', dict(args=('-c', '--location-count'), kwargs=dict(type=int, default=80, help='average number of locations per cell'), translate=True)),
        )),
    }


class RandomMesh(Voronoi):

    __slots__ = ('lower_bound', 'upper_bound', 'avg_probability')

    def __init__(self, scaler=None, avg_probability=None, lower_bound=None, upper_bound=None,
        cell_count=None, **kwargs):
        Voronoi.__init__(self, scaler)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if cell_count is not None:
            avg_probability = 1. / float(cell_count)
        self.avg_probability = avg_probability

    def _preprocess(self, points):
        Voronoi._preprocess(self, points) # initialize `scaler`
        return points # ... but do not scale

    # cell_centers property
    @property
    def cell_centers(self):
        return self._cell_centers # no scaling

    @cell_centers.setter
    def cell_centers(self, centers):
        self._cell_centers = centers

    def tessellate(self, points, allow_empty_cells=False, **kwargs):
        points = self._preprocess(points)
        # find the bounding box
        if self.lower_bound is None:
            self.lower_bound = points.min(axis=0)
        elif isinstance(points, pd.DataFrame) and not isinstance(self.lower_bound, pd.Series):
            self.lower_bound = pd.Series(self.lower_bound, index=points.columns)
        if self.upper_bound is None:
            self.upper_bound = points.max(axis=0)
        elif isinstance(points, pd.DataFrame) and not isinstance(self.upper_bound, pd.Series):
            self.upper_bound = pd.Series(self.upper_bound, index=points.columns)
        # pick some centroids within the bounding box
        space_columns = self.scaler.columns
        n_centroids = int(1. / self.avg_probability)
        centroids = pd.DataFrame(np.random.rand(n_centroids, len(space_columns)), columns=space_columns)
        centroids *= self.upper_bound - self.lower_bound
        centroids += self.lower_bound
        # make the Voronoi tessellation
        Voronoi.tessellate(self, centroids)
        #
        if not allow_empty_cells:
            cells = Partition(points, self)
            #cells.cell_index = cells.cell_index(centroids, **kwargs)
            ok = 0 < cells.location_count
            self.cell_centers = None
            Voronoi.tessellate(self, centroids[ok])


__all__ = ['setup', 'RandomMesh']

import sys
if sys.version_info[0] < 3:

    import rwa
    import tramway.core.hdf5 as rules
    random_mesh_exposes = rules.voronoi_exposes + list(RandomMesh.__slots__)
    rwa.hdf5_storable(
        rwa.default_storable(RandomMesh, exposes=random_mesh_exposes),
        agnostic=True)

    __all__.append('random_mesh_exposes')

