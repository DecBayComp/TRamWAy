# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import math
import numpy as np
from numpy.polynomial import polynomial as poly


def gradn(cells, i, X, index_map=None):
    """
    Local gradient by multi-dimensional 2 degree polynomial interpolation.

    Arguments:

        cells (tramway.inference.base.FiniteElements):
            distributed cells.

        i (int):
            cell index at which the gradient is evaluated.

        X (array):
            vector of a scalar measurement at every cell.

        index_map (array):
            index map that converts cell indices to indices in X.

    Returns:

        array:
            local gradient vector with as many elements as there are dimensions
            in the (trans-)location data.

    """
    cell = cells[i]

    # below, the measurement is renamed Y and the coordinates are X
    Y = X
    X0 = cell.center

    # cache the components related to X
    if not isinstance(cell.cache, dict):
        cell.cache = {}
    try:
        poly = cell.cache['poly']

    except KeyError: # fill in the cache
        # find neighbours
        adjacent = _adjacent = cells.neighbours(i)
        if index_map is not None:
            i = index_map[i]
            adjacent = index_map[_adjacent]
            ok = 0 <= adjacent
            if not np.all(ok):
                adjacent, _adjacent = adjacent[ok], _adjacent[ok]
        if adjacent.size:
            X0_dup = _adjacent.size
            local = np.r_[np.full(X0_dup, i), adjacent]
            # neighbour locations
            X = np.vstack([ X0 ] * X0_dup + [ cells[j].center for j in _adjacent ])
            # cache X-related terms and indices of the neighbours
            poly = cell.cache['poly'] = (_poly2(X), local)
        else:
            poly = cell.cache['poly'] = None

    if poly is None:
        return

    V, local = poly
    Y = Y[local]

    # interpolate
    W, _, _, _ = np.linalg.lstsq(V, Y, rcond=None)
    gradX = _poly2_deriv_eval(W, X0)

    return gradX


def _poly2(X):
    n, d = X.shape
    x = X.T
    return np.hstack((
        np.vstack([ x[u] * x[v] for v in range(d) for u in range(d) ]).T,
        X,
        np.ones((n, 1), dtype=X.dtype),
        ))

def _memoize1(f):
    results = {}
    def helper(x):
        try:
            y = results[x]
        except KeyError:
            y = results[x] = f(x)
        return y
    return helper

@_memoize1
def _poly2_meta_deriv(dim):
    poly = []
    for d in range(dim):
        p = []
        for v in range(dim):
            for u in range(dim):
                if u == d:
                    if v == d:
                        p.append((2., u))
                    else:
                        p.append((1., v))
                else:
                    if v == d:
                        p.append((1., u))
                    else:
                        p.append((0., None))
        for u in range(dim):
            if u == d:
                p.append((1., None))
            else:
                p.append((0., None))
        p.append((0., None))
        poly.append(p)
    return poly

def _poly2_deriv_eval(W, X):
    if not X.shape[1:]:
        X = X[np.newaxis, :]
    dim = X.shape[1]
    dP = _poly2_meta_deriv(dim)
    Q = []
    for Pd in dP:
        Qd = 0.
        i = 0
        for q, u in Pd:
            if q == 0:
                continue
            if u is None:
                Qd += q * W[i]
            else:
                Qd += q * W[i] * X[:,u]
            i += 1
        Q.append(Qd)
    return np.hstack(Q)


def neighbours_per_axis(i, cells, centers=None, eps=None, selection_angle=None, return_indices=None):
    """
    New in version `0.4.2`:
        the returned arrays can be indices in `cells` or `centers` instead of booleans in
        `cells.neighbours(i)` or `centers`, setting ``return_indices=True``.

    See also :func:`grad1`.
    """
    try:
        A = cells.spatial_adjacency
    except AttributeError:
        A = cells.adjacency

    if centers is None:
        neighbours = cells.neighbours(i)
        centers = np.vstack([ cells[j].center for j in neighbours ])
    else:
        neighbours = np.arange(len(centers))

    cell = cells[i]
    center = cell.center
    below = np.zeros((centers.shape[1], centers.shape[0]), dtype=bool)
    above = np.zeros_like(below)

    if eps is None:
        if selection_angle is None or (selection_angle == .5 and centers.shape[1] == 2):
            # InferenceMAP-compatible gradient

            if A.dtype == bool:
                # divide the space in non-overlapping region and
                # assign each neighbour to a single region;
                # assign each pair of symmetric regions to a single
                # dimension (gradient calculation)
                proj = centers - center[np.newaxis, :]
                assigned_dim = np.argmax(np.abs(proj), axis=1)
                proj_dist = proj[ np.arange(proj.shape[0]), assigned_dim ]
                for j in range(cell.dim):
                    below[ j, (assigned_dim == j) & (proj_dist < 0) ] = True
                    above[ j, (assigned_dim == j) & (0 < proj_dist) ] = True

            elif A.dtype == int:
                # neighbour cells are already classified
                # as left (-1), right (1), top (2) or bottom (-2), etc
                # in the adjacency matrix
                code = A.data[A.indptr[i]:A.indptr[i+1]] - 1
                for j in range(cell.dim):
                    below[ j, code == -j ] = True
                    above[ j, code ==  j ] = True

            else:
                raise TypeError('{} adjacency labels are not supported'.format(A.dtype))

        else:
            proj = centers - center[np.newaxis, :]
            proj /= np.sqrt(np.sum(proj * proj, axis=1, keepdims=True))
            angle = 1. - 2. * np.arccos(proj) / math.pi
            for j in range(cell.dim):
                below[ j, angle[:,j] < selection_angle - 1. ] = True
                above[ j, 1. - selection_angle < angle[:,j] ] = True

    else:
        # along each dimension, divide the space in half-spaces
        # minus margin `eps` (supposed to be positive)
        for j in range(cell.dim):
            x, x0 = centers[:,j], center[j]
            below[ j, x < x0 - eps ] = True
            above[ j, x0 + eps < x ] = True

    if return_indices:
        below = [ neighbours[b] for b in below ]
        above = [ neighbours[b] for b in above ]

    return below, above


def grad1(cells, i, X, index_map=None, eps=None, selection_angle=None, na=np.nan):
    """
    Local gradient by 2 degree polynomial interpolation along each dimension independently.

    Considering spatial coordinate :math:`x`, bin :math:`i` and its neighbour bins :math:`\\mathcal{N}_i`:

    .. math::

        \\left.X'_i\\right|_x = \\left\\{
            \\begin{array}{ll}
                \\frac{X_i - \\overline{X}_{\\mathcal{N}_i}}{x_i - \\overline{x}_{\\mathcal{N}_i}} &
                    \\textrm{ if either } \\mathcal{N}_i^- \\textrm{ or } \\mathcal{N}_i^+ \\textrm{ is } \\emptyset \\\\
                b + 2 c x_{i} & \\textrm{ with }
                    \\left[ \\begin{array}{ccc}
                        1 & \\overline{x}_{\\mathcal{N}_i^-} & \\overline{x}_{\\mathcal{N}_i^-}^2 \\\\
                        1 & x_i & x_i^2 \\\\
                        1 & \\overline{x}_{\\mathcal{N}_i^+} & \\overline{x}_{\\mathcal{N}_i^+}^2
                    \\end{array} \\right] . \\left[ \\begin{array}{c} a \\\\ b \\\\ c \\end{array} \\right] = \\left[ \\begin{array}{c}\\overline{X}_{\\mathcal{N}_i^-} \\\\ X_i \\\\ \\overline{X}_{\\mathcal{N}_i^+}\\end{array} \\right] \\textrm{ otherwise } \\\\
            \\end{array}
        \\right.

    Claims cache variable *grad1*.

    Supports the InferenceMAP way of dividing the space in non-overlapping regions (quadrants in 2D)
    so that each neighbour is involved in the calculation of a single component of the gradient.

    This is the default behaviour in cases where the `adjacent` attribute contains adequate
    labels, i.e. ``-j`` and ``j`` for each dimension ``j``, representing one side and the other
    side of cell-i's center.
    This is also the default behaviour otherwise, when `eps` is ``None``.

    If `eps` is defined, the calculation of a gradient component may recruit all the points
    minus those at a projected distance smaller than this value.

    If `selection_angle` is defined, neighbours are selected in two symmetric hypercones which
    top angle is `selection_angle` times :math:`\pi` radians.

    If :func:`get_grad_kwargs` is called (most if not all the inference modes call this function),
    the default selection behaviour is equal to `selection_angle=0.9`.
    Otherwise, :func:`grad1` defaults to `selection_angle=0.5` in 2D.

    See also:

        `neighbours_per_axis`.

    Arguments:

        cells (tramway.inference.base.FiniteElements):
            distributed cells.

        i (int):
            cell index at which the gradient is evaluated.

        X (array):
            vector of a scalar measurement at every cell.

        index_map (array):
            index map that converts cell indices to indices in X;
            for a given cell, should be constant.

        eps (float):
            minimum projected distance from cell-i's center for neighbours to be considered
            in the calculation of the gradient;
            if `eps` is ``None``, the space is instead divided in twice as many
            non-overlapping regions as dimensions, and each neighbour is assigned to
            a single region (counts against a single dimension).
            Incompatible with `selection_angle`.

        selection_angle (float):
            top angle of the neighbour selection hypercones;
            should be in the :math:`[0.5, 1.0[` range.
            Incompatible with `eps`.

        na (float):
            value for undefined components (no neighbours).

    Returns:

        array:
            local gradient vector with as many elements as there are dimensions
            in the (trans-)location data.
    """
    assert 0 < X.size
    cell = cells[i]
    # below, the measurement is renamed y and the coordinates are X
    y = X
    X0 = cell.center

    # cache neighbours (indices and center locations)
    if not isinstance(cell.cache, dict):
        cell.cache = {}
    try:
        i, adjacent, X = cell.cache['grad1']
    except KeyError:
        adjacent = _adjacent = cells.neighbours(i)
        if index_map is not None:
            adjacent = index_map[_adjacent]
            ok = 0 <= adjacent
            if not np.all(ok):
                adjacent, _adjacent = adjacent[ok], _adjacent[ok]
        if _adjacent.size:
            X = np.vstack([ cells[j].center for j in _adjacent ])
            below, above = neighbours_per_axis(i, cells, X, eps, selection_angle)

            # pre-compute the X terms for each dimension
            X_neighbours = []
            for j in range(cell.dim):
                u, v = below[j], above[j]
                if not np.any(u):
                    u = None
                if not np.any(v):
                    v = None

                if u is None:
                    if v is None:
                        Xj = None
                    else:
                        Xj = 1. / (X0[j] - np.mean(X[v,j]))
                elif v is None:
                    Xj = 1. / (X0[j] - np.mean(X[u,j]))
                else:
                    Xj = np.r_[X0[j], np.mean(X[u,j]), np.mean(X[v,j])]
                if np.isscalar(Xj):
                    try:
                        Xj = Xj.tolist()
                    except AttributeError:
                        pass
                    else:
                        if isinstance(Xj, list):
                            Xj = Xj[0]

                X_neighbours.append((u, v, Xj))

            X = X_neighbours
        else:
            X = []

        if index_map is not None:
            i = index_map[i]
        cell.cache['grad1'] = (i, adjacent, X)

    if not X:
        return None

    y0, y = y[i], y[adjacent]

    # compute the gradient for each dimension separately
    grad = []
    for u, v, Xj in X: # j= dimension index
        #u, v, Xj= below, above, X term
        if u is None:
            if v is None:
                grad_j = na
            else:
                # 1./Xj = X0[j] - np.mean(X[v,j])
                grad_j = (y0 - np.mean(y[v])) * Xj
        elif v is None:
            # 1./Xj = X0[j] - np.mean(X[u,j])
            grad_j = (y0 - np.mean(y[u])) * Xj
        else:
            # Xj = np.r_[X0[j], np.mean(X[u,j]), np.mean(X[v,j])]
            grad_j = _vander(Xj, np.r_[y0, np.mean(y[u]), np.mean(y[v])])
        grad.append(grad_j)

    return np.hstack(grad)


def _vander(x, y):
    #P = poly.polyfit(x, y, 2)
    #dP = poly.polyder(P)
    #return poly.polyval(x[0], dP)
    _, b, a = poly.polyfit(x, y, 2)
    return b + 2. * a * x[0]


def onesided_neighbours(i, cells=None, points=None, side='<>', selection_angle=.5):
    """
    2D only!

    For ``side=='>'``, `selection_angle` is:

    * ``0``: x=y positive half-line (null angle)
    * ``0.25``: top right quadrant (pi/2 actual angle)
    * ``0.5``: upper half-plane bounded by the x=-y line (pi actual angle)
    * ``0.75``: all quadrants but the bottom left one (3*pi/2 actual angle)
    * ``1``: full plane (2*pi actual angle)

    Arguments:

        i (int): cell index in `cells`.

        cells (FiniteElements): distributed (trans-)locations.

        points (numpy.ndarray): alternative relative centers;
            default centers are those in `cells` centered at `cells[i].center`.

        side (str): either '<' or '>'.

        selection_angle (float): any value in [0,1].

    Returns:

        numpy.ndarray: indices of the selected neighbours in `cells` or `points`.

    See also :func:`onesided_gradient`.
    """
    if points is None:
        center = cells[i].center
        __neighbours = cells.neighbours(i)
        points = np.vstack([ cells[j].center for j in __neighbours ]) - center[np.newaxis, :]
    else:
        __neighbours = None

    neighbours = []
    for s in side:

        # rotate the difference vectors onto the x=y axis (pi/4 or -3*pi/4)
        angle = math.pi/4
        rot = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
        if s == '<':
            rot = -rot # not exactly -3*pi/4, but further processing is symmetric

        proj = np.dot(points, rot.T)
        angle = np.arctan2(proj[:,0], proj[:,1])
        _neighbours = (selection_angle * math.pi) <= abs(angle)

        if __neighbours is not None:
            _neighbours = __neighbours[_neighbours]
        neighbours.append(_neighbours)

    if neighbours[1:]:
        return tuple(neighbours)
    else:
        return neighbours[0]


def onesided_gradient(cells, i, X, index_map=None, side='>', selection_angle=None, na=np.nan):
    cell = cells[i]

    # below, the measurement is renamed Y and the coordinates are X
    y = X
    X0 = cell.center

    # cache neighbours (indices and center locations)
    if not isinstance(cell.cache, dict):
        cell.cache = {}
    try:
        i, neighbours_lt, B_lt, neighbours_gt, B_gt = cell.cache['onesided_gradient']
    except KeyError:
        adjacent = _adjacent = cells.neighbours(i)
        if index_map is not None:
            adjacent = index_map[_adjacent]
            ok = 0 <= adjacent
            if not np.all(ok):
                adjacent, _adjacent = adjacent[ok], _adjacent[ok]
        # pre-compute the X (or B) terms
        if _adjacent.size:
            dX = cell.center[np.newaxis,:] - np.vstack([ cells[j].center for j in _adjacent ])
            dist2 = np.sum(dX * dX, axis=1, keepdims=True)
            Bs = []
            for _neighbours in onesided_neighbours(i, points=dX, side='<>', selection_angle=selection_angle):
                if np.any(_neighbours):
                    Bs.append( adjacent[_neighbours] )
                    Bs.append( dX[_neighbours] / dist2[_neighbours] )
                else:
                    Bs.append(None)
                    Bs.append(None)
        else:
            Bs = [None]*4

        if index_map is not None:
            i = index_map[i]

        neighbours_lt, B_lt, neighbours_gt, B_gt = Bs
        cell.cache['onesided_gradient'] = (i,) + tuple(Bs)

    y0 = y[i]

    if side == '<':
        neighbours, B = neighbours_lt, B_lt
    elif side == '>':
        neighbours, B = neighbours_gt, B_gt
    else:
        raise ValueError("unknown side '{}'".format(side))

    # compute the gradient
    if neighbours is None:
        return None
    else:
        return np.dot(B.T, y0 - y[neighbours])


__all__ = [ 'neighbours_per_axis', 'grad1', 'gradn', 'onesided_neighbours', 'onesided_gradient' ]

