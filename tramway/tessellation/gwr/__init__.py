# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import *
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.stats as stats
from ..base import *
from tramway.core.scaler import *
from .gas import Gas
from scipy.spatial.distance import cdist
import time
from collections import OrderedDict
from warnings import warn


class GasMesh(Voronoi):
    """GWR based tessellation.

    Attributes:
        gas (:class:`~tramway.tessellation.gwr.gas.Gas`):
            internal graph representation of the gas.
        min_probability (float):
            minimum probability of a point to be in any given cell.
        _min_distance (float):
            scaled minimum distance between adjacent cell centers.
        _avg_distance (float):
            upper bound on the average scaled distance between adjacent cell centers.
        _max_distance (float):
            scaled maximum distance between adjacent cell centers."""
    def __init__(self, scaler=None, min_distance=None, avg_distance=None, max_distance=None, \
        min_probability=None, avg_probability=None, **kwargs):
        Voronoi.__init__(self, scaler)
        self.gas = None
        self._min_distance = min_distance
        if avg_distance or max_distance is None:
            self._avg_distance = avg_distance
        else:
            self._avg_distance = max_distance * 0.25
        if max_distance or avg_distance is None:
            self._max_distance = max_distance
        else:
            self._max_distance = avg_distance * 4
        self.min_probability = min_probability
        #self.avg_probability = avg_probability

    def _preprocess(self, points, batch_size=10000, tau=333.0, trust=1.0, lifetime=50, **kwargs):
        if isinstance(points, tuple):
            points, displacements = points
        else:
            displacements = None
        init = self.scaler.init
        points = Voronoi._preprocess(self, points)
        if init:
            if self._min_distance is not None:
                self._min_distance = self.scaler.scale_distance(self._min_distance)
            if self._avg_distance is not None:
                if self._avg_distance <= 0:
                    raise ValueError('`avg_distance` is null or negative')
                self._avg_distance = self.scaler.scale_distance(self._avg_distance)
            if self._max_distance is not None:
                if self._max_distance <= 0:
                    raise ValueError('`max_distance` is null or negative')
                self._max_distance = self.scaler.scale_distance(self._max_distance)
        if self.gas is None:
            self.gas = Gas(np.asarray(points))
            if self._max_distance:
                # distances are diameters, while insertion thresholds should be radii
                self.gas.insertion_threshold = (self._avg_distance * 0.5, \
                    self._max_distance * 0.5)
                if self.min_probability:
                    self.gas.knn = int(round(self.min_probability * \
                        points.shape[0]))
                    if self.gas.knn == 0:
                        if self.min_probability<=1e-8:
                            warn('rounding min_probability down to 0', RuntimeWarning)
                            self.min_probability = 0
                            self.gas.knn = 20
                        else:
                            warn('min_probability is too low for the amount of available data', RuntimeWarning)
                            self.gas.knn = 1
                else:
                    self.gas.knn = 20
            self.gas.trust = trust
            self.gas.batch_size = batch_size
            if not isinstance(tau, tuple):
                tau = (tau, tau)
            self.gas.habituation_tau = tau
            self.gas.edge_lifetime = lifetime
            if self._min_distance:
                self.gas.collapse_below = self._min_distance# * 0.9
        return points, displacements

    def tessellate(self, points, pass_count=(), residual_factor=.7, error_count_tol=5e-3, \
        min_growth=1e-4, collapse_tol=.01, stopping_criterion=0, verbose=False, \
        plot=False, alpha_risk=1e-15, grab=None, max_frames=None, max_batches=None, axes=None, \
        complete_delaunay=False, topology='approximate density', \
        **kwargs):
        """Grow the tessellation.

        Arguments:
            points: see :meth:`~tramway.tessellation.Tessellation.tessellate`.
            pass_count (float or pair of floats):
                number of points to sample (with replacement) from data `points`, as
                a multiple of the size of the data.
                If `pass_count` is a pair of numbers, they are the lower and the upper
                bounds on the number of samples.
                If `pass_count` is a single number, it is interpreted as the lower
                bound, and the upper bound is set equal to ``2 * pass_count``.
            residual_factor (float): multiplies with `_max_distance` to determine
                `residual_max` in :meth:`~tramway.tessellation.gwr.gas.Gas.train`.
            error_count_tol (float): (see :meth:`~tramway.tessellation.gwr.gas.Gas.train`)
            min_growth (float): (see :meth:`~tramway.tessellation.gwr.gas.Gas.train`)
            collapse_tol (float): (see :meth:`~tramway.tessellation.gwr.gas.Gas.train`)
            stopping_criterion (int): (see :meth:`~tramway.tessellation.gwr.gas.Gas.train`)
            verbose (bool): verbose output.
            batch_size (int): (see :class:`~tramway.tessellation.gwr.gas.Gas`)
            tau (float): (see :class:`~tramway.tessellation.gwr.gas.Gas`)
            trust (float): (see :class:`~tramway.tessellation.gwr.gas.Gas`)
            lifetime (int): (see :class:`~tramway.tessellation.gwr.gas.Gas`)
            alpha_risk (float): location distributions of potential neighbor cells
                are compared with a t-test
            complete_delaunay (bool): complete the Delaunay graph
            topology (str): any of 'approximate density' (default), 'displacement length'

        Returns:
            See :meth:`~tramway.tessellation.Tessellation.tessellate`.

        See also:
            :class:`tramway.tessellation.gwr.gas.Gas` and
            :meth:`tramway.tessellation.gwr.gas.Gas.train`.
        """
        #np.random.seed(15894754) # to benchmark and compare between Graph implementations
        points, displacements = self._preprocess(points, **kwargs)
        if self._avg_distance:
            residual_factor *= self._avg_distance # important: do this after _preprocess!
        if pass_count is not None:
            if pass_count == (): # () has been chosen to denote default (not-None) value
                n = points.shape[0]
                p = .95
                # sample size for each point to have probability p of being chosen once
                pass_count = log(1.0 - p) / log(1.0 - 1.0 / float(n))
                # convert in number of passes
                pass_count /= float(n)
            try:
                len(pass_count) # sequence?
            except TypeError:
                pass_count = (pass_count, 2 * pass_count)
        if not (topology is None or topology == 'approximate density'):
            self.gas.topology = topology
        self.residuals = self.gas.train( \
            np.asarray(points) if displacements is None else (np.asarray(points), np.asarray(displacements)), \
            pass_count=pass_count, \
            residual_max=residual_factor, \
            error_count_tol=error_count_tol, \
            min_growth=min_growth, \
            collapse_tol=collapse_tol, \
            stopping_criterion=stopping_criterion, \
            verbose=verbose, plot=plot, \
            grab=grab, max_frames=max_frames, max_batches=max_batches, axes=axes)
        # build alternative representation of the gas (or Delaunay graph)
        [self._cell_adjacency, V, _] = self.gas.export()
        self._cell_centers = V['weight']
        self._cell_adjacency.data = np.ones_like(self._cell_adjacency.data, dtype=int)
        self.alpha_risk = alpha_risk
        self._postprocess(points, complete_delaunay, verbose, _update_cell_adjacency=True)

    def _postprocess(self, points=None, complete_delaunay=False, verbose=False, _update_cell_adjacency=False):
        # build the Voronoi graph
        voronoi = Voronoi._postprocess(self)
        if not _update_cell_adjacency:
            return voronoi # stop here

        # clean and extend the adjacency matrix with the Delaunay graph
        adjacency = self._cell_adjacency # shorter name
        # fix for issue on reload
        if 4 < adjacency.data[-1]:
            adjacency.data[:] = 1
        delaunay = sparse.csr_matrix( \
            (np.ones(2*voronoi.ridge_points.shape[0], dtype=int), \
            (voronoi.ridge_points.flatten('F'), \
            np.fliplr(voronoi.ridge_points).flatten('F'))), \
            shape=adjacency.shape)
        # if `ridge_points` includes the same ridge twice,
        # the corresponding elements in `data` are added
        if not np.all(delaunay.data == 1):
            warn('some Voronoi ridges appear twice', RuntimeWarning)
        delaunay.data[:] = 2
        A = sparse.tril(adjacency + delaunay, format='coo')
        assert delaunay.data.size/2 <= A.data.size
        self._adjacency_label = A.data # labels are: 1=gas only, 2=voronoi only, 3=both
        # edge indices for _adjacency_label
        adjacency = sparse.csr_matrix( \
            (np.tile(np.arange(0, self._adjacency_label.size), 2), \
            (np.concatenate((A.row, A.col)), np.concatenate((A.col, A.row)))), \
            shape=adjacency.shape)
        self.cell_adjacency = adjacency

        ## reintroduce Delaunay edges that do not appear in the gas
        # (turn some label-2 edges into label-4)

        if complete_delaunay:
            self._adjacency_label[self._adjacency_label==2] = 4

        elif points is not None:
            #t = time.time()
            points = np.asarray(points)
            try:
                ix = np.argmin(cdist(points, self._cell_centers), axis=1)
            except MemoryError:
                X, Y = points, self._cell_centers
                # slice X to process less rows at a time (borrowed from tessellation.base.Delaunay.cell_index)
                ix = np.zeros(X.shape[0], dtype=int)
                X2 = np.sum(X * X, axis=1, keepdims=True).astype(np.float32)
                Y2 = np.sum(Y * Y, axis=1, keepdims=True).astype(np.float32)
                X, Y = X.astype(np.float32), Y.astype(np.float32)
                n = 0
                while True:
                    n += 1
                    block = int(ceil(X.shape[0] * 2**(-n)))
                    try:
                        np.empty((block, Y.shape[0]), dtype=X.dtype)
                    except MemoryError:
                        pass # continue
                    else:
                        break
                n += 2 # safer
                block = int(ceil(X.shape[0] * 2**(-n)))
                for i in range(0, X.shape[0], block):
                    j = min(i+block, X2.size)
                    Di = np.dot(np.float32(-2.)* X[i:j], Y.T)
                    Di += X2[i:j]
                    Di += Y2.T
                    ix[i:j] = np.argmin(Di, axis=1)
            #
            ref = int( ceil(float(self.gas.knn) / 8.0) ) # int and float for PY2
            #ref = -ref # with alternative to cdist, index from the end
            ref -= 1 # with cdist, index
            A = sparse.tril(self._cell_adjacency, format='coo') # in future scipy version, check that tril does not remove explicit zeros
            assert np.any(A.data == 0)
            # compute the median distance between adjacent cell centers
            pts_i = np.stack([ self._cell_centers[i] for i in A.row ])
            pts_j = np.stack([ self._cell_centers[j] for j in A.col ])
            ref_d = np.sqrt(np.median(np.sum((pts_i - pts_j)**2, axis=1)))
            for i, j, k in zip(A.row, A.col, A.data):
                if self._adjacency_label[k] == 2: # only in Voronoi
                    xi = points[ix == i]
                    xj = points[ix == j]
                    if 1 < xi.shape[0] and 1 < xj.shape[0]:
                        dij = cdist(xi, xj)
                        #dij = np.dot(xi, xj.T)
                        #xi2 = np.sum(xi * xi, axis=1, keepdims=True)
                        #dij -= 0.5 * xi2
                        #xj2 = np.sum(xj * xj, axis=1, keepdims=True)
                        #dij -= 0.5 * xj2.T
                        dij = dij.flatten()
                        #kij = dij.argsort()
                        dij.sort()
                        if ref_d * .9 < dij[0]:
                            continue
                        try:
                            dij = dij[ref]
                        except IndexError:
                            if verbose and 1 < verbose:
                                print('skipping edge {:d} between cell {:d} (card = {:d}) and cell {:d} (card = {:d}): number of between-cell pairs = {:d} (expected: {:d})'.format(k, i, xi.shape[0], j, xj.shape[0], dij.size, ref))
                            continue
                        #dij = np.sqrt(-2.0 * dij)
                        if dij < self._min_distance:
                            self._adjacency_label[k] = 4 # mark edge as 'not congruent but valid'
                            continue
                        ci, cj = np.mean(xi, axis=0), np.mean(xj, axis=0)
                        yi = np.sort(np.dot(xi - ci, cj - ci))
                        yj = np.sort(np.dot(xj - ci, cj - ci))
                        # throttle the number of points down to control the p-value
                        n0 = 10
                        yi = yi[::-1][:min(max(n0, (ref+1)*2), yi.size)]
                        yj = yj[:min(max(n0, (ref+1)*2), yj.size)]
                        t, p = stats.ttest_ind(yi.T, yj.T, equal_var=False)
                        if self.alpha_risk < p:
                            self._adjacency_label[k] = 4 # mark edge as 'not congruent but valid'
                        continue
                        # debug candidate edges
                        # comment out the above `continue` statement
                        # and uncomment the `argsort` line
                        cell_i_k, cell_j_k = np.unravel_index(kij[ref], (xi.shape[0], xj.shape[0]))
                        if not hasattr(self, 'candidate_edges'):
                            self.candidate_edges = {}
                        self.candidate_edges[k] = (xi[cell_i_k], xj[cell_j_k])
                        #
                    elif verbose and 1 < verbose:
                        print('skipping edge {:d} between cell {:d} (card = {:d}) and cell {:d} (card = {:d})'.format(k, i, xi.shape[0], j, xj.shape[0]))
            sparsity = np.float(np.sum(self._adjacency_label==2)) / np.float(np.sum(0<self._adjacency_label))
            if .5 < sparsity:
                warn('the Delaunay-like graph is very sparse compared to the actual Delaunay graph; pass `complete_delaunay=True` to get the Delaunay graph instead', RuntimeWarning)

        new_labels = np.array([0,-1,-2,1,2])
        # before: 0=[none], 1=not congruent (gas only), 2=not congruent (voronoi only),
        #     3=congruent, 4=congruent after post-processing (initially voronoi only)
        # after: -2=not congruent (voronoi only), -1=not congruent (gas only), 0=[none],
        #     1=congruent, 2=congruent after post-processing (initially voronoi only)
        self._adjacency_label = new_labels[self._adjacency_label]

        return voronoi

    def freeze(self):
        self.gas = None



setup = {
    'name': ('gas', 'gwr'),
    'make': GasMesh,
    'make_arguments': OrderedDict((
        ('min_distance', ()),
        ('avg_distance', ()),
        ('max_distance', ()),
        ('min_probability', ()),
        ('pass_count', dict(type=float, help='fraction of the data to be sampled; can be greater than 1 (recommended)')),
        )),
    }

__all__ = ['GasMesh', 'setup']

