# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
from .grad import neighbours_per_axis


def delta0(cells, i, X, index_map=None, **kwargs):
    """
    Differences with neighbour values:

    .. math::

        \\Delta X_i = \\frac{1}{\\sqrt{|\\mathcal{N}_i|}} \\left[ \\frac{X_i-X_j}{|| \\textbf{x}_i-\\textbf{x}_j ||} \\right]_{j \\in \\mathcal{N}_i}

    The above scaling is chosen so that combining the element-wise square of :meth:`~tramway.inference.base.Distributed.local_variation` with :meth:`~tramway.inference.base.Distributed.grad_sum` results in the following scalar penalty:

    .. math::

        \\Delta X_i^2 = \\frac{1}{ | \\mathcal{N}_i | } \\sum_{j \\in \\mathcal{N}_i} \\left( \\frac{X_i-X_j}{|| \\textbf{x}_i-\\textbf{x}_j ||} \\right)^2

    Claims cache variable '*delta0*'.

    Arguments:

        cells (tramway.inference.base.Distributed):
            distributed cells.

        i (int):
            cell index at which the differences are evaluated.

        X (array):
            vector of a scalar measurement at every cell.

        index_map (array):
            index map that converts cell indices to indices in X.

    Returns:

        array:
            difference vector with as many elements as there are neighbours.

    """
    cell = cells[i]
    # below, the measurement is renamed y and the coordinates are X
    y = X

    # cache neighbours (indices and center locations)
    if not isinstance(cell.cache, dict):
        cell.cache = {}
    try:
        i, adjacent, dx_norm = cell.cache['delta0']
    except KeyError:
        adjacent = _adjacent = cells.neighbours(i)
        if index_map is not None:
            adjacent = index_map[_adjacent]
            ok = 0 <= adjacent
            if not np.all(ok):
                adjacent, _adjacent = adjacent[ok], _adjacent[ok]
        if _adjacent.size:
            x0 = cell.center[np.newaxis,:]
            x = np.vstack([ cells[j].center for j in _adjacent ])
            dx_norm = x - x0
            dx_norm = np.sqrt(np.sum(dx_norm * dx_norm, axis=1))
        else:
            dx_norm = None

        if index_map is not None:
            i = index_map[i]
        cell.cache['delta0'] = (i, adjacent, dx_norm)

    if dx_norm is None:
        return None

    y0, y = y[i], y[adjacent]

    # scale by the number of differences to make the sum of the returned values be a mean value instead
    return (y - y0) / dx_norm / np.sqrt(float(len(y)))


def delta0_without_scaling(cells, i, X, index_map=None, **kwargs):
    cell = cells[i]
    # below, the measurement is renamed y and the coordinates are X
    y = X

    # cache neighbours (indices and center locations)
    if not isinstance(cell.cache, dict):
        cell.cache = {}
    try:
        i, adjacent, dx_norm = cell.cache['delta0']
    except KeyError:
        adjacent = _adjacent = cells.neighbours(i)
        if index_map is not None:
            adjacent = index_map[_adjacent]
            ok = 0 <= adjacent
            if not np.all(ok):
                adjacent, _adjacent = adjacent[ok], _adjacent[ok]
        if _adjacent.size:
            x0 = cell.center[np.newaxis,:]
            x = np.vstack([ cells[j].center for j in _adjacent ])
            dx_norm = x - x0
            dx_norm = np.sqrt(np.sum(dx_norm * dx_norm, axis=1))
        else:
            dx_norm = None

        if index_map is not None:
            i = index_map[i]
        cell.cache['delta0'] = (i, adjacent, dx_norm)

    if dx_norm is None:
        return None

    y0, y = y[i], y[adjacent]

    # scale by the number of differences to make the sum of the returned values be a mean value instead
    return (y - y0) / dx_norm


def delta1(cells, i, X, index_map=None, eps=None, selection_angle=None):
    """
    Local spatial variation.

    Similar to `grad1`.
    Considering spatial coordinate :math:`x`, bin :math:`i` and its neighbour bins :math:`\\mathcal{N}_i`:

    .. math::

        \\left.\\Delta X_i\\right|_x = \\left(
            \\begin{array}{ll}
                \\frac{X_i - \\overline{X}_{\\mathcal{N}_i^-}}{x_i - \\overline{x}_{\\mathcal{N}_i^-}} &
                    \\textrm{ or } 0 \\textrm{ if } \\mathcal{N}_i^- \\textrm{ is } \\emptyset \\\\
                \\frac{X_i - \\overline{X}_{\\mathcal{N}_i^+}}{x_i - \\overline{x}_{\\mathcal{N}_i^+}} &
                    \\textrm{ or } 0 \\textrm{ if } \\mathcal{N}_i^+ \\textrm{ is } \\emptyset \\\\
            \\end{array}
        \\right)

    Also claims cache variable *grad1* in a compatible way.

    Arguments:

        i (int):
            cell index at which the gradient is evaluated.

        X (numpy.ndarray):
            vector of a scalar measurement at every cell.

        index_map (numpy.ndarray):
            index map that converts cell indices to indices in X.

    Returns:

        numpy.ndarray:
            delta vector with as many elements as there are spatial dimensions.

    """
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
                #if np.isscalar(Xj):
                #    try:
                #        Xj = Xj.tolist()
                #    except AttributeError:
                #        pass
                #    else:
                #        if isinstance(Xj, list):
                #            Xj = Xj[0]

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

    # compute the delta for each dimension separately
    delta = []
    for u, v, Xj in X: # j= dimension index
        #u, v, Xj= below, above, X term
        if u is None:
            if v is None:
                delta_j = np.r_[0., 0.]
            else:
                # 1./Xj = X0[j] - np.mean(X[v,j])
                delta_j = np.r_[0., (y0 - np.mean(y[v])) * Xj]
        elif v is None:
            # 1./Xj = X0[j] - np.mean(X[u,j])
            delta_j = np.r_[(y0 - np.mean(y[u])) * Xj, 0.]
        else:
            # Xj = np.r_[X0[j], np.mean(X[u,j]), np.mean(X[v,j])]
            x0, xu, xv = Xj
            delta_j = np.r_[
                (y0 - np.mean(y[u])) / (x0 - xu),
                (y0 - np.mean(y[v])) / (x0 - xv),
                ]
            #delta_j = np.mean(np.abs(delta_j))
        delta.append(delta_j)

    return np.stack(delta, axis=1) # columns must represent the space dimensions



__all__ = ['delta0', 'delta0_without_scaling', 'delta1']

