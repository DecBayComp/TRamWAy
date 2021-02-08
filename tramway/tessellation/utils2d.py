# -*- coding: utf-8 -*-

# Copyright © 2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import numpy as np
from tramway.tessellation.base import *
import scipy.spatial as spatial
from collections import namedtuple, deque, defaultdict


_Voronoi = namedtuple('BoxedVoronoi', (
        'points',
        'vertices',
        'ridge_points',
        'ridge_vertices',
        'regions',
        'point_region',
    ))

def boxed_voronoi_2d(points, bounding_box=None):
    """ moved from the *base* module with version *0.5.2*; *deprecated* """
    voronoi = spatial.Voronoi(points)
    if bounding_box is None:
        #x_min = np.minimum(np.min(points, axis=0), np.min(voronoi.vertices, axis=0))
        #x_max = np.maximum(np.max(points, axis=0), np.max(voronoi.vertices, axis=0))
        x_min, x_max = np.min(points, axis=0), np.max(points, axis=0)
        dx = x_max - x_min
        x_min -= 1e-4 * dx
        x_max += 1e-4 * dx
        bounding_box = (
            x_min,
            np.array([x_max[0], x_min[1]]),
            x_max,
            np.array([x_min[0], x_max[1]]),
            )
    t = (bounding_box[0] + bounding_box[2]) * .5
    lstsq_kwargs = dict(rcond=None)
    region_point = np.full(len(voronoi.regions), -1, dtype=int)
    region_point[voronoi.point_region] = np.arange(voronoi.point_region.size)
    extra_vertices = []
    vertex_index = n_vertices = voronoi.vertices.shape[0]
    extra_ridges = []
    n_ridges = voronoi.ridge_points.shape[0]
    _ridge_vertices, _ridge_points, _regions = \
        voronoi.ridge_vertices, voronoi.ridge_points, [[]]
    for c, u in enumerate(points):
        r = voronoi.point_region[c]
        region = voronoi.regions[r]
        #assert region_point[r] == c
        _region = []
        for k, h in enumerate(region):
            if 0 <= h:
                _region.append(h)
                continue

            # find the two "adjacent" vertices
            i, j = region[k-1], region[(k+1)%len(region)]
            assert 0<=i
            assert 0<=j
            # find the corresponding neighbour cells
            m, n = set(), set() # mutable
            for d, hs in enumerate(voronoi.regions):
                if not hs or d == r:
                    continue
                for e, cs in ((i,m), (j,n)):
                    try:
                        l = hs.index(e)
                    except ValueError:
                        continue
                    if (hs[l-1]==-1) or (hs[(l+1)%len(hs)]==-1):
                        cs.add(region_point[d])
            p, q = m.pop(), n.pop()
            assert not m
            assert not n
            # pick a distant point on the perpendicular bissector
            # of the neighbour centers, at the intersection with
            # the bounding box
            prev_edge = None
            for h, d in ((i,p), (j,q)):
                gs = []
                try:
                    gs = _ridge_vertices[n_ridges+extra_ridges.index([d,c])]
                except ValueError:
                    try:
                        gs = _ridge_vertices[n_ridges+extra_ridges.index([c,d])]
                    except ValueError:
                        pass
                if gs:
                    g, = [ g for g in gs if g != h ]
                    _region.append(g)
                    continue
                v = voronoi.vertices[h]
                w = points[d]
                if np.any(v<bounding_box[0]) or np.any(bounding_box[2]<v):
                    # vertex v stands outside the bounding box
                    # TODO: check for corners as potential intermediate vertices
                    continue
                n = np.array([u[1]-w[1], w[0]-u[0]])
                #y = (u + w) * .5
                y = v
                # determine direction: n or -n?
                if 0 < np.dot(n, t-y):
                    n = -n
                # intersection of [y,n) and [a,ab]
                v_next = None
                for l, a in enumerate(bounding_box):
                    b = bounding_box[(l+1)%len(bounding_box)]
                    M, p = np.c_[n, a-b], a-y
                    q = np.linalg.lstsq(M, p, **lstsq_kwargs)
                    q = q[0]
                    if 0<=q[0] and 0<=q[1] and q[1]<=1:
                        # intersection found
                        v_next = y + q[0] * n
                        #if 0<q[0]:
                        break
                assert v_next is not None
                if prev_edge is None or prev_edge == l:
                    h_prev = h
                else:
                    # add the corner which the previous
                    # and current edges intersect at
                    e_max = len(bounding_box)-1
                    if (0<l and prev_edge==l-1) or \
                        (l==0 and prev_edge==e_max):
                        v_prev = a
                    elif (l<e_max and prev_edge==l+1) or \
                        (l==e_max and prev_edge==0):
                        v_prev = b
                    else:
                        # two corners?
                        # better connect the intersection points and let the corners out
                        h_prev = h
                        v_prev = None
                    if v_prev is not None:
                        h_prev = vertex_index
                        vertex_index += 1
                        extra_vertices.append(v_prev)
                        extra_ridges.append([-1,c])
                        _ridge_vertices.append([h,h_prev])
                        _region.append(h_prev)
                prev_edge = l
                #
                h_next = vertex_index
                vertex_index += 1
                # insert the new vertex
                extra_vertices.append(v_next)
                k, = ((_ridge_points[:,0]==c) & (_ridge_points[:,1]==d)).nonzero()
                if k.size==0:
                    k, = ((_ridge_points[:,0]==d) & (_ridge_points[:,1]==c)).nonzero()
                k = k[0].tolist()
                _ridge_vertices = _ridge_vertices[:k] \
                    + [[h_prev,h_next]] \
                    + _ridge_vertices[k+1:]
                _region.append(h_next)
                assert len(extra_vertices) == vertex_index - n_vertices

        assert all( r in _region for r in region if 0 < r )
        _regions.append(_region)
    _points = voronoi.points
    _vertices = np.vstack([voronoi.vertices]+extra_vertices)
    _ridge_points = np.vstack([_ridge_points]+extra_ridges)
    _point_region = np.arange(1, len(_regions))#voronoi.point_region
    return _Voronoi(_points, _vertices, _ridge_points, _ridge_vertices, _regions, _point_region)


def get_exterior_cells(tessellation, relative_margin=.05, bounds=None, extended_cell_filter=None):
    """
    Looks for Voronoi cells that expand outside the 2D data bounding box.

    This function can be called in two ways:

    * pass a partition object, which bounding box is inferred from location data,
      and optionaly define a margin on this default bounding box,
    * pass a SciPy *Voronoi* or tessellation object and explicitly define the bounding box.

    Arguments:

        tessellation (*scipy.spatial.Voronoi* or *Voronoi* or *Partition*):
            SciPy *Voronoi* object,
            or spatial segmentation (*TimeLattice* objects are supported),
            or full data partition

        relative_margin (float):
            margin for the default bounds, to identify outside vertices

        bounds (pair of *numpy.ndarray*):
            lower and upper bounds on the spatial coordinates

        extended_cell_filter (callable):
            boolean function that takes the vertices of any closed cell with at least
            one out-of-bound vertex, and returns :const:`True` if the corresponding
            cell should be marked as exterior, :const:`False` otherwise

    Returns:

        list: indices of exterior cells
    
    Modified from https://github.com/DecBayComp/Stochastic_Integrals_Diffusivity/blob/master/ito-to-tramway/get_exterior_cells.py
    """
    if bounds is None:
        if isinstance(tessellation, Partition):
            partition = tessellation
            bb = partition.bounding_box
            space_cols = list('xy')
            lb, ub = bb.loc['min',space_cols].values, bb.loc['max',space_cols].values
        else:
            raise ValueError('undefined bounds')
    else:
        lb, ub = bounds

    if isinstance(tessellation, spatial.Voronoi):
        voronoi = tessellation

    else:
        if isinstance(tessellation, Partition):
            tessellation = tessellation.tessellation
        try:
            tessellation = tessellation.spatial_mesh
        except AttributeError:
            pass
        # rebuild the Voronoi diagram with explicit undefined vertices (with index -1)
        voronoi = spatial.Voronoi(tessellation.cell_centers)

    margin = relative_margin * (ub-lb)
    outside_vertices = np.any((tessellation.vertices<lb-margin)|(ub+margin<tessellation.vertices), axis=1)

    exterior_cells = set()
    ncells = len(voronoi.points)

    for cell_ix in range(ncells):

        region_ix = voronoi.point_region[cell_ix]
        if region_ix < 0:
            continue

        vertex_ids = np.asarray(voronoi.regions[region_ix])

        if np.any(vertex_ids < 0):
            # divergent cell
            exterior_cells.add(cell_ix)

        elif np.any(outside_vertices[vertex_ids]):
            # out-of-bounds vertex
            if not extended_cell_filter or extended_cell_filter(voronoi.vertices[vertex_ids]):
                exterior_cells.add(cell_ix)

    return list(exterior_cells)


class Path(object):
    __slots__ = ('fragments',)
    def __init__(self):
        self.fragments = []
    def add_edge(self, i, j):
        """
        If edge *(i,j)* closes the path, `add_edge` returns the resulting path.
        Otherwise :const:`None` is returned instead.
        """
        connected = None
        if self.fragments:
            fragments = []
            connected_twice = False
            for f, fragment in enumerate(self.fragments):
                if connected_twice:
                    fragments.append(fragment)
                    continue
                #
                ends, path = fragment
                # check if the (i, j) edge connects with the `fragment`
                if i in ends:
                    resume_from, continuation = i, j
                elif j in ends:
                    resume_from, continuation = j, i
                else:
                    # the edge does not connect;
                    # leave the fragment untouched
                    fragments.append(fragment)
                    continue
                # sanity checks
                if continuation in path:
                    if continuation in ends and not self.fragments[1:]:
                        # the path is complete
                        _, complete_path = self.fragments[0]
                        return complete_path
                    else:
                        raise ValueError('duplicate edges')
                elif connected is None:
                    # connect the (i,j) edge with the path `fragment`:
                    # replace the `resume_from` node by the `continuation` node
                    ends = ends - {resume_from} | {continuation}
                    # append the `continuation` node to the `path`
                    if path[0] == resume_from:
                        path.appendleft(continuation)
                    else: #if path[-1] == resume_from
                        path.append(continuation)
                    # update the fragment
                    fragments.append((ends, path))
                    connected = f
                else:
                    # combine fragments
                    fragments, _fragments = [], fragments
                    for f, fragment in enumerate(_fragments):
                        if f == connected:
                            ends0, path0 = fragment
                            connection_point = resume_from
                            # check `ends` and `ends0` connect
                            if not (connection_point in ends and connection_point in ends0):
                                raise RuntimeError('triangle found')
                            assert continuation in path0
                            # connect `path` with `path0`
                            new_ends = ends0 ^ ends
                            assert len(new_ends)==2
                            if path[0] != connection_point: # or path[-1] == connection_point
                                path.reverse()
                            path.popleft()
                            new_path = path0
                            if path0[0] == connection_point:
                                new_path.extendleft(path)
                            else:# if path0[-1] == connection_point
                                new_path.extend(path)
                            fragment = (new_ends, new_path)
                        fragments.append(fragment)
                    connected_twice = True
            self.fragments = fragments
        if connected is None:
            # initialize a new path fragment
            self.fragments.append((set((i,j)), deque((i,j))))
    def get_path(self):
        if len(self.fragments) == 1:
            _, path = self.fragments[0]
            path = list(path)
        else:
            path = None
        return path


def get_interior_contour(tessellation, relative_margin=None, bounds=None, return_indices=False,
        extended_cell_filter=None):
    """
    Calls `get_exterior_cells` and returns the 2D inner vertices of exterior cells
    so that these vertices make a contour around the interior cells.

    Can return vertex indices instead.
    Input arguments are similar to those of `get_interior_contour`.
    """
    if bounds is None and isinstance(tessellation, Partition):
        partition = tessellation
        bb = partition.bounding_box
        space_cols = list('xy')
        bounds = bb.loc['min',space_cols].values, bb.loc['max',space_cols].values

    if isinstance(tessellation, spatial.Voronoi):
        voronoi = tessellation

    else:
        if isinstance(tessellation, Partition):
            tessellation = tessellation.tessellation
        try:
            tessellation = tessellation.spatial_mesh
        except AttributeError:
            pass
        # rebuild the Voronoi diagram with explicit undefined vertices (with index -1)
        voronoi = spatial.Voronoi(tessellation.cell_centers)

    if relative_margin is None:
        outer_ids = get_exterior_cells(voronoi, bounds=bounds,
                extended_cell_filter=extended_cell_filter)
    else:
        outer_ids = get_exterior_cells(voronoi, relative_margin,
                bounds, extended_cell_filter)

    outer_ids = set(outer_ids)

    adjacency = defaultdict(set)
    for k, ij in enumerate(voronoi.ridge_points):
        i, j = ij
        adjacency[i].add((j, k))
        adjacency[j].add((i, k))

    contour_ridge_ids = set()
    for i in outer_ids:
        contour_ridge_ids |= { k for j, k in adjacency[i] if j not in outer_ids }

    vertex_path = Path()
    while contour_ridge_ids:
        ridge_ix = contour_ridge_ids.pop()
        if vertex_path.add_edge(*voronoi.ridge_vertices[ridge_ix]):
            assert not contour_ridge_ids
    vertex_path = vertex_path.get_path()

    if return_indices:
        return vertex_path
    else:
        return voronoi.vertices[vertex_path]


__all__ = ['_Voronoi', 'boxed_voronoi_2d', \
    'get_exterior_cells', 'get_interior_contour']

