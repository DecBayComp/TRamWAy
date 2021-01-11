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
from .base import Delaunay, Voronoi
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from collections import OrderedDict
from warnings import warn


class KohonenMesh(Voronoi):
    r"""Kohonen fitting for arbitrary initial mesh.

    Attributes:
        step_size (tuple of float):
            :math:`(\epsilon_0, \epsilon_T, \gamma)` that determine step size
            :math:`\epsilon_t = a(b+t)^{-\gamma}` at iteration :math:`t`
            with :math:`T` the size of the dataset given to given to :meth:`tessellate`;
            default is ``(.1, .01, .55)``
        neighbourhood_weight (pair of float):
            weight for the winning node (first element) and its neighbours (second element);
            default is ``(1., .1)``
        pass_count (int or float):
            number of passes over the dataset; default is 5
        initial_mesh (Delaunay):
            mesh which cell centers are used as initial nodes;
            default consists of estimating the location density and randomly sampling
            the cell centers from this distribution
        initial_mesh_tessellate_kwargs (dict):
            arguments to :meth:`initial_mesh.tessellate`

    Trailing keyword arguments are passed to :meth:`initial_mesh.__init__`.
    The default mesh initialization expects arguments: *avg_probability*, *min_distance* and
    *max_distance*.

    """

    __slots__ = ('_step_size', '_neighbourhood_weight', 'pass_count',
        'initial_mesh', 'initial_mesh_init_kwargs', 'initial_mesh_tessellate_kwargs')

    def __init__(self, scaler=None, step_size=None, neighbourhood_weight=None, pass_count=None,
        initial_mesh=None, initial_mesh_tessellate_kwargs={}, **initial_mesh_init_kwargs):
        Voronoi.__init__(self, scaler)
        if step_size is None:
            step_size = (.1, .01, .55)
        self._step_size = step_size
        if neighbourhood_weight is None:
            neighbourhood_weight = (1., .1)
        self._neighbourhood_weight = neighbourhood_weight
        if pass_count is None:
            pass_count = 5
        self.pass_count = pass_count
        self.initial_mesh = initial_mesh
        self.initial_mesh_init_kwargs = initial_mesh_init_kwargs
        self.initial_mesh_tessellate_kwargs = initial_mesh_tessellate_kwargs

    def tessellate(self, points, **kwargs):
        # build initial mesh
        if self.initial_mesh is None:
            #from tramway.tessellation.grid import RegularMesh
            #self.initial_mesh = RegularMesh
            from tramway.tessellation.gwr.gas import Gas
            avg_probability = self.initial_mesh_init_kwargs['avg_probability']
            knn = 10#round(avg_probability * points.shape[0])
            rmin = self.initial_mesh_init_kwargs['min_distance'] * .01
            rmax = self.initial_mesh_init_kwargs['max_distance']
            if rmax is None:
                rmax = self.initial_mesh_init_kwargs['avg_distance'] * 2.
            X = np.asarray(self._preprocess(points))
            #if 10000 < X.shape[0]:
            #    X = X[np.random.randint(0, X.shape[0], 10000)]
            radius = Gas(X).boxed_radius(X, knn, rmin, rmax)
            density = 1. / radius
            density /= np.sum(density)
            P = np.r_[0, np.cumsum(density)]
            p = np.sort(np.random.rand(int(round(1./avg_probability))))
            i, j = [], 0
            for q in p:
                while P[j] <= q:
                    j += 1
                i.append(j-1)
            #i = [ ((P[:-1] <= _p) & (_p < P[1:])).nonzero()[0][0].tolist() for _p in p ]
            self.initial_mesh = Voronoi(self.scaler)
            if isinstance(points, pd.DataFrame):
                X = points.loc[i]
            else:
                X = points[i]
            self.initial_mesh.tessellate(X)
        if callable(self.initial_mesh):
            self.initial_mesh = self.initial_mesh(self.scaler, **self.initial_mesh_init_kwargs)
            self.initial_mesh.tessellate(points, **self.initial_mesh_tessellate_kwargs)
            ## copy a few pointers
            self._cell_centers = self.initial_mesh._cell_centers
            self._cell_label = self.initial_mesh.cell_label
            #self._cell_adjacency = self.initial_mesh.cell_adjacency
            #self._adjacency_label = self.initial_mesh.adjacency_label
        elif isinstance(self.initial_mesh, Delaunay):
            ## copy a few pointers
            self._cell_centers = self.initial_mesh._cell_centers
            self._cell_label = self.initial_mesh.cell_label
            #self._cell_adjacency = self.initial_mesh.cell_adjacency
            #self._adjacency_label = self.initial_mesh.adjacency_label
            pass
        else:
            raise NotImplementedError
        # step size
        T = points.shape[0]
        eps_0, eps_T, gamma = self._step_size
        c = (eps_T / eps_0) ** (-1. / gamma)
        b = float(T) / (c - 1.)
        a = eps_0 / (b ** (-gamma))
        self._step_size = (a, b, gamma)
        # learn
        X = np.asarray(self.initial_mesh._preprocess(points))
        W = self._cell_centers
        t_max = T * self.pass_count
        t = 0
        while t < t_max:
            i = np.random.randint(T)
            x = X[i]
            dW = x - W
            u = np.argmin(np.sum(dW * dW, axis=1))
            W[u] += self.neighbourhood_weight(0, t) * self.step_size(t) * dW[u]
            visited = U = set([u])
            for d in range(1, len(self._neighbourhood_weight)):
                V = set()
                for u in U:
                    V |= set(self.initial_mesh.neighbours(u).tolist())
                V -= visited
                if not V:
                    break
                V_ = list(V)
                W[V_] += self.neighbourhood_weight(d, t) * self.step_size(t) * dW[V_]
                visited |= V
                U = V
            t += 1

    def step_size(self, t):
        a, b, gamma = self._step_size
        return a * (b + float(t)) ** (-gamma)

    def neighbourhood_weight(self, d, t):
        return self._neighbourhood_weight[d]

    # cell_centers property
    @property
    def cell_centers(self):
        return self.initial_mesh.cell_centers

    @cell_centers.setter
    def cell_centers(self, centers):
        self.initial_mesh.cell_centers = centers
        self._cell_centers = self.initial_mesh._cell_centers

    # cell_label
    @property
    def cell_label(self):
        return self.initial_mesh.cell_label

    @cell_label.setter
    def cell_label(self, labels):
        self.initial_mesh.cell_label = labels

    # cell_adjacency property
    @property
    def cell_adjacency(self):
        return self.initial_mesh.cell_adjacency

    @cell_adjacency.setter
    def cell_adjacency(self, matrix):
        self.initial_mesh.cell_adjacency = matrix
        self._cell_adjacency = self.initial_mesh._cell_adjacency

    # adjacency label
    @property
    def adjacency_label(self):
        return self.initial_mesh.adjacency_label

    @adjacency_label.setter
    def adjacency_label(self, labels):
        self.initial_mesh.adjacency_label = labels
        self._adjacency_label = self.initial_mesh._adjacency_label


setup = {
    'make_arguments': OrderedDict((
        ('min_distance', ()),
        ('avg_distance', ()),
        ('max_distance', ()),
        ('avg_probability', ()),
        # avg_location_count allows to control avg_probability from the commandline
        ('avg_location_count', dict(args=('-c', '--location-count'), kwargs=dict(type=int, default=80, help='average number of locations per cell'), translate=True)),
        )),
    }


__all__ = ['setup', 'KohonenMesh']

import sys
if sys.version_info[0] < 3:

    import rwa
    import tramway.core.hdf5 as rules
    kohonen_mesh_exposes = rules.voronoi_exposes + list(KohonenMesh.__slots__)
    rwa.hdf5_storable(
        rwa.default_storable(KohonenMesh, exposes=kohonen_mesh_exposes),
        agnostic=True)

    __all__.append('kohonen_mesh_exposes')

