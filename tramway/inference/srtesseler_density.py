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


setup = {"name": "srtesseler.density", "provides": "density", "input_type": "Partition"}


def convex_hull(
    vertices=None,
    vertex_indices=None,
    points=None,
    point_assigned=None,
    partition=None,
    cell_index=None,
):
    # recover the vertices of the Voronoi cell
    if vertex_indices is None:
        vertex_indices = partition.tessellation.cell_vertices[cell_index]
        if vertices is None:
            vertices = partition.tessellation.vertices
    if np.any(vertex_indices < 0):
        valid_indices = vertex_indices[0 <= vertex_indices]
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


def infer_srtesseler_density(cells, volume_weighted=True, rank=0, **kwargs):
    """
    2D particle density estimation inspired by:

        SR-Tesseler: a method to segment and quantify localization-based
        super-resolution microscopy data.
        Levet, F., Hosy, E., Kechkar, A., Butler, C., Beghin, A., Choquet, C.,
        Sibarita, J.B.
        Nature Methods 2015; 12 (11); 1065-1071.

    This (much) simplified implementation borrows from the above reference the
    same general principle:

    * a Voronoi tessellation is made out of all particle locations (one
      location = one Voronoi cell)
    * for each Voronoi cell (or particle location), the cell may be extended to
      merge with its neighbors (see argument `rank`)
    * the local density is estimated classically as the number of points (of
      merged cells) divided by the total surface area of the merged cells

    """
    points = cells.locations
    is_densest_tessellation = isinstance(
        cells.tessellation, Voronoi
    ) and cells.number_of_cells == len(points)
    if is_densest_tessellation:
        tessellation = cells.tessellation
        partition = cells
    else:
        tessellation = Voronoi()
        # TODO: try 3D
        # coords = [ col for col in 'xyz' if col in points.columns ]
        if "z" in points.columns:
            import warnings

            warnings.warn("ignoring coordinate 'z'")
        tessellation.tessellate(points[["x", "y"]])
        partition = Partition(points, tessellation)
    # estimate the density at each point
    polygons_required = not is_densest_tessellation and volume_weighted
    indices, surface_areas, polygons = [], [], []
    for i in range(tessellation.number_of_cells):
        hull = convex_hull(partition=partition, cell_index=i)
        if hull is None:
            continue
        surface_area = hull.volume
        indices.append(i)
        surface_areas.append(surface_area)
        if polygons_required:
            polygons.append(
                p.Polytope(hull.equations[:, [0, 1]], -hull.equations[:, 2])
            )
    indices, surface_areas = np.array(indices), np.array(surface_areas)
    if rank:
        if 1 < rank:
            raise NotImplementedError("rank > 1")
        index_map = np.full(tessellation.number_of_cells, len(indices))
        index_map[indices] = indices
        extended_areas = np.array(surface_areas)  # copy
        ncells = np.ones_like(surface_areas)
        regions = [[p] for p in polygons]
        for i in indices:
            for j in tessellation.neighbours(i):
                j = index_map[j]
                try:
                    extended_areas[i] += surface_areas[j]
                    ncells[i] += 1
                    if polygons_required:
                        regions[i].append(polygons[j])
                except IndexError:
                    pass
        polygons = [p.Region(ps) for ps in regions]
        local_density = ncells / extended_areas
    else:
        local_density = 1.0 / surface_areas
    local_density = pd.Series(index=indices, data=local_density)
    # sum the estimates within each spatial bin
    if is_densest_tessellation:
        density = local_density
    else:
        if isinstance(cells.cell_index, tuple):
            pt_ids, cell_ids = cells.cell_index
        indices, densities = [], []
        for i in range(cells.number_of_cells):
            if isinstance(cells.cell_index, tuple):
                polygons_i = pt_ids[cell_ids == i]
                if polygons_i.size == 0:
                    indices.append(i)
                    densities.append(0.0)
                    continue
                assigned = np.zeros(len(points), dtype=bool)
                assigned[polygons_i] = True
            else:
                assigned = cells.cell_index == i
                if not np.any(assigned):
                    indices.append(i)
                    densities.append(0.0)
                    continue
                (polygons_i,) = np.nonzero(assigned)
            average_density = local_density[polygons_i].mean()
            indices.append(i)
            densities.append(average_density)
        density = pd.Series(index=indices, data=densities)
    return pd.DataFrame(dict(density=density))


if __name__ == "__main__":
    import os.path
    from tramway.helper import load_rwa, map_plot1
    from matplotlib import pyplot as plt

    analysis_tree = load_rwa(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "tests",
            "test_commandline_py3_210816",
            "tessellation_output_gwr0.rwa",
        )
    )
    sampling = analysis_tree[0].data
    simple_map1 = infer_srtesseler_density(sampling, rank=1)
    map_plot1(simple_map1, sampling)
    plt.show()
    simple_map0 = infer_srtesseler_density(sampling, rank=0)
    map_plot1(simple_map0, sampling)
    plt.show()


__all__ = ["infer_srtesseler_density", "setup"]
