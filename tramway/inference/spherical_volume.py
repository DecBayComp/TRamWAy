# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import pi
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from collections import OrderedDict


setup = {'name': 'spherical.volume',
        'provides': 'volume'}

def ritter(points):
    if len(points) == 1:
        return points, 0.

    eps = 1e-8

    i = 0
    x = points[[i]]
    dx = points - x
    j = np.argmax(np.sum(dx * dx, axis=1))
    y = points[[j]]
    dy = points - y
    k = np.argmax(np.sum(dy * dy, axis=1))
    z = points[[k]]
    center = .5 * (y + z)
    yz = z - y
    radius = .5 * np.sqrt(np.sum(yz * yz)) + eps

    covered = set((j,k))
    while True:
        dr = points - center
        d2 = np.sum(dr * dr, axis=1)
        m = np.argmax(d2)
        if m in covered:
            return center, radius
        covered.add(m)

        d = np.sqrt(d2[m])
        if d <= radius:
            radius = d
            return center, radius
        p = points[[m]]
        alpha = radius / d
        radius = .5 * (d + radius)
        center = .5 * ((1. - alpha) * p + (1. + alpha) * center)


def infer_spherical_volume(cells, **kwargs):
    """Returns a DataFrame.

    See also `spherical_volume`.
    """
    vol = spherical_volume(cells, 'all', **kwargs)
    return None if vol is None else pd.DataFrame(vol, columns=['volume'])


def spherical_volume(cells, which='all', bounding_sphere='ritter', **kwargs):
    """
    Volume is evaluated looking for a bounding sphere.
    If the sphere does not include locations from neighbour cells as well,
    then its radius is increased until an external location is reached.

    The volume of the cell is approximated by the volume of the resulting sphere.

    For now only the Ritter's bounding sphere algorithm is implemented.

    Returns a Series, or a float if `which` is a single cell index.
    """
    if not callable(bounding_sphere):
        if bounding_sphere.lower() != 'ritter':
            raise ValueError("unsupported bounding sphere algorithm: '{}'".format(bounding_sphere))
        bounding_sphere = ritter
    indices, radii = [], []
    for index in (cells if which == 'all' else which):
        cell = cells[index]
        sphere = bounding_sphere(cell.r, **kwargs)
        if sphere is not None:
            center, radius = sphere
            neighbour_cells = cells.neighbours(index)
            if neighbour_cells.size:
                neighbour_locations = [ cells[neighbour].r for neighbour in neighbour_cells ]
                neighbour_locations = np.vstack(neighbour_locations)
                dist = neighbour_locations - center
                dist = np.sqrt(np.min(np.sum(dist * dist, axis=1)))
                if radius < dist:
                    radius = dist
            else:
                # all the neighbour cells were discarded
                continue
            indices.append(index)
            radii.append(radius)
    if not radii:
        return None
    radii = np.array(radii)
    volumes = 4 / 3 * pi * radii**3

    if which == 'all' or not np.isscalar(which):
        return pd.Series(volumes, index=indices)
    else:
        assert np.isscalar(volumes)
        return volumes.tolist()[0]


__all__ = [ 'infer_spherical_volume', 'spherical_volume', 'setup', 'ritter' ]

