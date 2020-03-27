# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
import polytope as p
from tramway.tessellation import Voronoi, Partition
from scipy.spatial import ConvexHull


setup = {'name':        'srtesseler.density',
         'provides':    'density',
         'input_type':  'Partition'}


def convex_hull(vertices=None, vertex_indices=None, points=None, point_assigned=None,
        partition=None, cell_index=None):
    # recover the vertices of the Voronoi cell
    if vertex_indices is None:
        vertex_indices = partition.tessellation.cell_vertices[cell_index]
        if vertices is None:
            vertices = partition.tessellation.vertices
    if np.any(vertex_indices < 0):
        valid_indices = vertex_indices[0<=vertex_indices]
        vertices = vertices[valid_indices]
        # for cells with missing vertices (to the infinite),
        # consider some inner points to possibly push the hull outward
        if point_assigned is None:
            point_assigned = partition.cell_index == cell_index
        if points is None:
            points = partition.points
        assigned_points = points[point_assigned]
        try:
            hull = ConvexHull(np.r_[vertices, assigned_points])
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            hull = None
    else:
        hull = ConvexHull(vertices[vertex_indices])
    return hull


def infer_srtesseler_density(cells, volume_weighted=True, **kwargs):
    """
    adapted from:

        SR-Tesseler: a method to segment and quantify localization-based super-resolution microscopy data.
        Levet, F., Hosy, E., Kechkar, A., Butler, C., Beghin, A., Choquet, C., Sibarita, J.B.
        Nature Methods 2015; 12 (11); 1065-1071.

    """
    points = cells.locations
    is_densest_tessellation = isinstance(cells.tessellation, Voronoi) and cells.number_of_cells == len(points)
    if is_densest_tessellation:
        tessellation = cells.tessellation
        partition = cells
    else:
        tessellation = Voronoi()
        tessellation.tessellate(points[['x','y']])
        partition = Partition(points, tessellation)
    #areas = tessellation.cell_volume # the default implementation for cell_volume
                                      # generates a non-infinite estimation for opened cells
    areas = np.full(tessellation.number_of_cells, np.nan)
    indices, densities, unit_polygons = [], [], []
    for i, surface_area in enumerate(areas):
        unknown_area = np.isnan(surface_area) or np.isinf(surface_area) or surface_area == 0
        polygons_required = not is_densest_tessellation and volume_weighted
        if unknown_area or polygons_required:
            hull = convex_hull(partition=partition, cell_index=i)
        if unknown_area:
            if hull is None:
                continue
            density = 1./ hull.volume # should also divide by the number of frames
        else:
            density = 1./ surface_area
        indices.append(i)
        densities.append(density)
        if polygons_required:
            unit_polygons.append(p.Polytope(hull.equations[:,[0,1]], -hull.equations[:,2]))
    unit_density = pd.Series(index=indices, data=densities)
    if is_densest_tessellation:
        density = unit_density
    else:
        indices, densities = [], []
        for i in range(cells.number_of_cells):
            assigned = cells.cell_index == i
            if not np.any(assigned):
                indices.append(i)
                densities.append(0.)
                continue
            local_polygons, = np.nonzero(assigned)
            assert 0 < local_polygons.size
            if volume_weighted:
                hull = convex_hull(partition=cells, cell_index=i, point_assigned=assigned)
                if hull is None:
                    continue
                polygon = p.Polytope(hull.equations[:,[0,1]], -hull.equations[:,2])
                weighted_sum = 0.
                total_area = 0.
                for j in local_polygons:
                    intersection = p.intersect(unit_polygons[j], polygon)
                    total_area += intersection.volume
                    weighted_sum += intersection.volume * unit_density[j]
                average_density = weighted_sum / total_area
            else:
                average_density = unit_density[local_polygons].mean()
            indices.append(i)
            densities.append(average_density)
        density = pd.Series(index=indices, data=densities)
    return pd.DataFrame(dict(density=density))


__all__ = ['infer_srtesseler_density', 'setup']

