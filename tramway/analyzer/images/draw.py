
import numpy as np
from skimage import draw
from math import *

def line_aa(i0, j0, i1, j1, thickness=1):
    if thickness:
        thickness = max(1., thickness)
        # grid
        i_lb = ceil(min(i0, i1) - thickness)
        j_lb = ceil(min(j0, j1) - thickness)
        i_ub = floor(max(i0, i1) + thickness)
        j_ub = floor(max(j0, j1) + thickness)
        i = np.arange(i_lb, i_ub+1)
        j = np.arange(j_lb, j_ub+1)
        i_, j_ = np.meshgrid(i, j, indexing='ij')
        i_, j_ = i_.ravel(), j_.ravel()
        ij = np.stack((i_, j_), axis=1)
        # distance to segment
        di, dj = i1 - i0, j1 - j0
        n = sqrt(di * di + dj * dj)
        n_ = np.r_[dj / n, -di / n]
        ij0 = np.array([[i0, j0]])
        d = np.dot(ij - ij0, n_)
        d2 = d * d
        d_max = 1. * thickness
        ok = d2 < d_max * d_max
        # selected indices and corresponding alpha
        i = i_[ok]
        j = j_[ok]
        d2 = d2[ok]
        d = np.sqrt(d2)
        d_cutoff = .5 * thickness
        k = 1 / (1 + np.exp(4 * (d - d_cutoff) / d_cutoff))
        # discard the corners
        w_ = np.r_[di / n, dj / n]
        w = np.dot(ij[ok] - ij0, w_)
        w0 = w<0
        w1 = n<w
        d0 = -w[w0]
        d0 = np.sqrt(d0 * d0 + d2[w0])
        d1 = w[w1] - n
        d1 = np.sqrt(d1 * d1 + d2[w1])
        k[w0] = np.minimum(k[w0],
                1 / (1 + np.exp(4 * (d0 - d_cutoff) / d_cutoff)))
        k[w1] = np.minimum(k[w1],
                1 / (1 + np.exp(4 * (d1 - d_cutoff) / d_cutoff)))
        i, j = i.astype(dtype=int), j.astype(dtype=int)
    else:
        i, j, k = draw.line_aa(i0, j0, i1, j1)
    return i, j, k
    
def multiline_aa(path_i, path_j, thickness=0):
    """
    If thickness is null, pixel indices and line thickness should be integers.
    Otherwise, floating-point values are allowed.

    Arguments:

        path_i (array-like): pixel column indices

        path_j (array-like): pixel row indices

        thickness (int): thickness in pixels

    Returns:

        (ndarray, ndarray, ndarray): column indices, row indices, pixel values

    """
    max_k = {}
    for i0, j0, i1, j1 in \
            zip(path_i[:-1], path_j[:-1], path_i[1:], path_j[1:]):
        for i, j, k in zip(*line_aa(i0, j0, i1, j1, thickness)):
            try:
                k_prev = max_k[(i,j)]
            except KeyError:
                pass
            else:
                k = max(k, k_prev)
            max_k[(i,j)] = k
    i, j, k = zip(*[ ij+(k,) for ij, k in max_k.items() ])
    return np.array(i), np.array(j), np.array(k)

__all__ = ['line_aa', 'polyline_aa']
