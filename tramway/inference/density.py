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
import pandas as pd
from scipy.spatial import ConvexHull
from collections import OrderedDict


setup = {'arguments': OrderedDict((
        ('convex_hull', dict(help="'always' if the partition was made with `knn`")),
        ('trust_volume', dict(action='store_true', help="for non-Voronoi tessellations only")),
        ))}


def infer_density(cells, convex_hull='fallback', trust_volume=False, scale=False, min_volume=None):
    """
    If ``knn`` was defined for the partition, use ``convex_hull='always'``.

    With the Voronoi-based tessellations (*random*, *kmeans*, *gwr*), do not set
    ``trust_volume`` to ``True``.
    With the other tessellations (*grid*, *hexagon*, *kdtree*), use
    ``trust_volume=True`` if ``convex_hull`` evaluates to ``True``.

    With the regular tessellations (*grid*, *hexagon*), ``scale`` can be ``True``.
    With the other tessellations, do not set ``scale`` to ``True``.
    """
    convex_hull_only = convex_hull == 'always'
    indices, densities = [], []
    for index in cells:
        cell = cells[index]
        if convex_hull_only:
            vol = np.inf
        else:
            vol = cell.volume
            if vol == 0 or np.isnan(vol):
                vol = np.inf
        if np.isinf(vol) and convex_hull and cell.dim < len(cell):
            hull = ConvexHull(cell.r)
            vol = hull.volume
            if min_volume:
                vol = max(min_volume, vol)
            elif trust_volume:
                vol = max(cell.volume, vol)
        density = len(cell) / vol
        indices.append(index)
        densities.append(density)
    if scale:
        densities = np.array(densities)
        if scale is True:
            scale = cell.volume # any cell; regular tessellations only
        densities *= scale
    return pd.DataFrame(densities, index=indices, columns=['density'])


__all__ = ['infer_density', 'setup']

