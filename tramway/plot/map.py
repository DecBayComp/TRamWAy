# -*- coding: utf-8 -*-

# Copyright © 2017-2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import tan, atan2, degrees, radians
import sys
import numpy as np
import pandas as pd
import numpy.ma as ma
from tramway.core.exceptions import NaNWarning
from tramway.tessellation import *
from tramway.inference import FiniteElements, Maps
from matplotlib.patches import Polygon, Wedge
from matplotlib.collections import PatchCollection
import scipy.spatial
import scipy.sparse as sparse
from warnings import warn


def _bounding_box(cells, xy=None):
    # bounding box
    try:
        bounding_box = cells.bounding_box[['x', 'y']]
        xy_min, xy_max = bounding_box.values
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        xy = cells.points[['x', 'y']]
        xy_min, xy_max = xy.min(axis=0), xy.max(axis=0)
    bounding_box = [ np.array(vs) for vs in (xy_min, [xy_min[0], xy_max[1]], xy_max, [xy_max[0], xy_min[1]]) ]
    return bounding_box


def cell_to_polygon(c, X, voronoi=None, bounding_box=None, region_point=None, return_voronoi=False):
    lstsq_kwargs = {}
    if sys.version_info[0] < 3:
        lstsq_kwargs['rcond'] = None
    if bounding_box is None:
        bounding_box = _bounding_box(None, X)
    if voronoi is None:
        voronoi = scipy.spatial.Voronoi(X)
    r = voronoi.point_region[c]
    region = voronoi.regions[r]
    if region_point is None:
        region_point = np.full(len(voronoi.regions), -1, dtype=int)
        region_point[voronoi.point_region] = np.arange(voronoi.point_region.size)
    #assert region_point[r] == c
    t = (bounding_box[0] + bounding_box[2]) * .5
    u = X[c]
    vertices = []
    for k, v in enumerate(region):
        if v < 0:

            # find the two "adjacent" vertices
            i, j = region[k-1], region[(k+1)%len(region)]
            assert 0<=i
            assert 0<=j
            # find the corresponding neighbour cells
            m, n = set(), set() # mutable
            for d, vs in enumerate(voronoi.regions):
                if not vs or d == r:
                    continue
                for e, cs in ((i,m), (j,n)):
                    try:
                        l = vs.index(e)
                    except ValueError:
                        continue
                    if (vs[l-1]==-1) or (vs[(l+1)%len(vs)]==-1):
                        cs.add(region_point[d])
            p, q = m.pop(), n.pop()
            assert not m
            assert not n
            # pick a distant point on the perpendicular bissector
            # of the neighbour centers, at the intersection with
            # the bounding box
            prev_edge = None
            for v, w in ((i,p), (j,q)):
                v, w = voronoi.vertices[v], X[w]
                if np.any(v<bounding_box[0]) or np.any(bounding_box[2]<v):
                    # vertex v stands outside the bounding box
                    continue
                n = np.array([u[1]-w[1], w[0]-u[0]])
                #w = (u + w) * .5
                w = v
                # determine direction: n or -n?
                if 0 < np.dot(n, t-w):
                    n = -n
                # intersection of [w,n) and [a,ab]
                z = None
                for l, a in enumerate(bounding_box):
                    b = bounding_box[(l+1)%len(bounding_box)]
                    M, p = np.c_[n, a-b], a-w
                    q = np.linalg.lstsq(M, p, **lstsq_kwargs)
                    q = q[0]
                    if 0<=q[0] and 0<=q[1] and q[1]<=1:
                        # intersection found
                        z = w + q[0] * n
                        #if 0<q[0]:
                        break
                assert z is not None
                if not (prev_edge is None or prev_edge == l):
                    # add the corner which the previous
                    # and current edges intersect at
                    e_max = len(bounding_box)-1
                    if (0<l and prev_edge==l-1) or \
                        (l==0 and prev_edge==e_max):
                        vertices.append(a)
                    elif (l<e_max and prev_edge==l+1) or \
                        (l==e_max and prev_edge==0):
                        vertices.append(b)
                    else:
                        raise RuntimeError
                prev_edge = l
                vertices.append(z)

        else:
            vertices.append(voronoi.vertices[v])
    if return_voronoi:
        return vertices, voronoi, bounding_box, region_point
    else:
        return vertices


def box_voronoi_2d(tessellation, xlim, ylim):
    not_a_vertex = -1
    points = tessellation.cell_centers
    vertices0 = tessellation.vertices
    cell_vertices0 = tessellation.cell_vertices
    Av0 = tessellation.vertex_adjacency.tocoo()
    xy0 = tessellation.cell_centers
    c0_inner = (xlim[0]<xy0[:,0]) & (ylim[0]<xy0[:,1]) & \
               (xy0[:,0]<xlim[1]) & (xy0[:,1]<ylim[1])
    c_inner_set = set(np.nonzero(c0_inner)[0])
    x, y = points[c0_inner,[0]], points[c0_inner,[1]]
    xy1 = np.r_[ xy0,
          np.c_[ 2*xlim[0]-x, y ],
          np.c_[ x, 2*ylim[0]-y ],
          np.c_[ 2*xlim[1]-x, y ],
          np.c_[ x, 2*ylim[1]-y ] ]
    c1_inner = np.r_[ c0_inner, np.zeros(4*len(x), dtype=c0_inner.dtype) ]
    voronoi1 = scipy.spatial.Voronoi(xy1)
    ridge_vertices1 = np.array(voronoi1.ridge_vertices)
    #
    r1_border = np.array([ c1_inner[r[0]] != c1_inner[r[1]] for r in voronoi1.ridge_points ])
    v1_border_set = set(ridge_vertices1[r1_border].flatten())
    assert -1 not in v1_border_set
    #
    v1_inner_set = set()
    for c in c_inner_set:
        vs = set(voronoi1.regions[voronoi1.point_region[c]])
        v1_inner_set |= set(vs)
    try:
        v1_inner_set.remove(-1)
    except KeyError:
        pass
    v1_inner_set -= v1_border_set
    #
    corners = np.array([ [ xlim[0], ylim[0] ],
                         [ xlim[0], ylim[1] ],
                         [ xlim[1], ylim[1] ],
                         [ xlim[1], ylim[0] ] ])
    # TODO: check corners are not in vertices0
    v1_inner = np.array(list(v1_inner_set | v1_border_set))
    v1_inner_map = np.full(len(voronoi1.vertices), not_a_vertex, dtype=int)
    v1_inner_map[v1_inner] = np.arange(len(v1_inner))
    V1 = voronoi1.vertices[v1_inner]
    V12 = .5 * np.sum(V1 * V1, axis=1, keepdims=True)
    D2 = np.dot(V1, corners.T) - V12 - .5* np.sum(corners*corners, axis=1, keepdims=True).T
    v1_corners = np.argmax(D2, axis=0)
    assert np.all(np.isclose(D2[v1_corners,np.arange(len(corners))], 0, atol=1e-6))
    v1_corners = v1_inner[v1_corners]
    v1_corners_set = set(v1_corners)
    assert v1_corners_set <= v1_border_set
    #
    v1_cntr = v1_cntr0 = len(vertices0)
    vertices1 = []
    cell_vertices1 = list(cell_vertices0)
    v01_new_edges, v11_new_edges = set(), set()
    v1_map = np.full(len(voronoi1.vertices), not_a_vertex, dtype=int)
    v0_keep = np.zeros(len(vertices0), dtype=bool)
    #
    v1_border = list(v1_border_set)
    vertices1.append(voronoi1.vertices[v1_border])
    v1_map[v1_border] = np.arange(v1_cntr, v1_cntr+len(v1_border))
    v1_cntr += len(v1_border)
    #
    for c in cell_vertices0:
        v0 = cell_vertices0[c]
        if c in c_inner_set:
            V0 = vertices0[v0]
            v1 = np.array(voronoi1.regions[voronoi1.point_region[c]])
            v1 = v1[0<=v1]
            V1 = voronoi1.vertices[v1]
            V02 = .5 * np.sum(V0 * V0, axis=1, keepdims=True)
            #V12_ = .5 * np.sum(V1 * V1, axis=1, keepdims=True)
            assert np.all(0<=v1_inner_map[v1])
            V12_ = V12[v1_inner_map[v1]]
            D2 = np.dot(V0, V1.T) - V02 - V12_.T
            v0c_match = np.argmax(D2, axis=0)
            v0c_keep = np.isclose(D2[v0c_match, np.arange(len(V1))], 0, atol=1e-6)
            if not np.any(v0c_keep):
                warn('the Voronoi diagrams totally mismatch at the border', RuntimeWarning)
                #continue
                #print(D2[v0c_match, np.arange(len(V1))])
                #import matplotlib.pyplot as plt
                #plt.scatter(V0[:,0], V0[:,1], 200, c='g', marker='+')
                #plt.scatter(V1[:,0], V1[:,1], 200, c='r', marker='x')
                ##plt.scatter(corners[:,0], corners[:,1], c='b', marker='o')
                #plt.show()
            v0_match = v0[v0c_match[v0c_keep]]
            v1_match = v1[v0c_keep]
            v1_replace = v1[~v0c_keep]
            v0_kept = np.unique(v0_match)
            v0_keep[v0_kept] = True
            #assert len(v0_kept) == np.sum(v0c_keep) # not sure about that
            v1_mapped = v1_map[v1_replace]
            v1_reused = v1_mapped[v1_mapped!=not_a_vertex]
            #
            V1_new = V1[~v0c_keep][v1_mapped==not_a_vertex]
            vertices1.append(V1_new)
            v1_new = np.arange(v1_cntr, v1_cntr+len(V1_new))
            v1_cntr += len(V1_new)
            #
            v1_map[v1_replace[v1_mapped==not_a_vertex]] = v1_new
            if v0_kept.size==0 and v1_new.size==0:
                cell_vertices1[c] = v1_reused
            else:
                cell_vertices1[c] = np.r_[np.array(list(v0_kept)), v1_reused, v1_new]
            #
            v1_set = set(v1)
            for _v0, _v1 in zip(v0_match, v1_match):
                _v1_neighbours = \
                        set(ridge_vertices1[ridge_vertices1[:,0]==_v1,1]) | \
                        set(ridge_vertices1[ridge_vertices1[:,1]==_v1,0])
                for _v1_neighbour in _v1_neighbours & v1_border_set:
                    v01_new_edges.add((_v0, _v1_neighbour))
                    #
                    if _v1_neighbour in v1_corners_set:
                        continue
                    _v1_neighbour_neighbours = \
                        set(ridge_vertices1[ridge_vertices1[:,0]==_v1_neighbour,1]) | \
                        set(ridge_vertices1[ridge_vertices1[:,1]==_v1_neighbour,0])
                    _corner = _v1_neighbour_neighbours & v1_corners_set
                    while _corner:
                        v11_new_edges.add((_v1_neighbour, _corner.pop()))
        else:
            v0_keep[v0] = True
    v0_map = np.full(len(vertices0), not_a_vertex, dtype=int)
    n0 = np.sum(v0_keep)
    v0_map[v0_keep] = np.arange(n0)
    assert n0 <= v1_cntr0
    v1_map[v1_map!=not_a_vertex] += n0 - v1_cntr0
    vertices1.insert(0, vertices0[v0_keep])
    vertices1 = np.vstack(vertices1)
    for c, _vs in enumerate(cell_vertices1):
        # some elements in cell_vertices1 are scalars; convert them into arrays
        if not isinstance(_vs, np.ndarray):
            _vs = cell_vertices1[c] = np.array([_vs])
        # for all cell at the border,
        if c in c_inner_set:
            # ...map the additional vertex indices so they match the compact vertices1
            _v1 = v1_cntr0 <= _vs
            _vs[_v1] += n0 - v1_cntr0
            if not np.all(_v1):
                # ...and similarly map the pre-existing vertices that are not discarded
                _vs[~_v1] = v0_map[_vs[~_v1]]
                assert not np.any(_vs == not_a_vertex)
    v01_new0, v01_new1 = zip(*v01_new_edges)
    v01_new0, v01_new1 = v0_map[np.array(v01_new0)], v1_map[np.array(v01_new1)]
    assert not np.any(v01_new0 == not_a_vertex)
    assert not np.any(v01_new1 == not_a_vertex)
    if v11_new_edges:
        v11_new0, v11_new1 = zip(*v11_new_edges)
        v11_new0, v11_new1 = v1_map[np.array(v11_new0)], v1_map[np.array(v11_new1)]
        assert not np.any(v11_new0 == not_a_vertex)
        assert not np.any(v11_new1 == not_a_vertex)
    else:
        v11_new0, v11_new1 = [], []
    Av1_rows, Av1_cols = v0_map[Av0.row], v0_map[Av0.col]
    _ok = (0<=Av1_rows) & (0<=Av1_cols)
    Av1_rows, Av1_cols = Av1_rows[_ok], Av1_cols[_ok]
    Av1 = sparse.csr_matrix((np.ones(len(Av1_rows)+2*len(v01_new_edges)+2*len(v11_new_edges), dtype=bool),
            (np.r_[Av1_rows, v01_new0, v01_new1, v11_new0, v11_new1],
             np.r_[Av1_cols, v01_new1, v01_new0, v11_new1, v11_new0])),
            shape=(len(vertices1), len(vertices1)))
    return vertices1, cell_vertices1, Av1


def scalar_map_2d(cells, values, aspect=None, clim=None, figure=None, axes=None, linewidth=1,
        delaunay=False, colorbar=True, alpha=None, colormap=None, unit=None, clabel=None,
        xlim=None, ylim=None, try_fix_corners=True, return_patches=False, **kwargs):
    """
    Plot a 2D scalar map as a colourful image.

    Arguments:

        cells (Partition or FiniteElements): spatial description of the cells

        values (pandas.DataFrame or numpy.ndarray): feature value at each cell,
            that will be represented as a colour

        aspect (str): passed to :func:`~matplotlib.axes.Axes.set_aspect`

        clim (2-element sequence): passed to :func:`~matplotlib.cm.ScalarMappable.set_clim`

        figure (matplotlib.figure.Figure): figure handle

        axes (matplotlib.axes.Axes): axes handle

        linewidth (int): cell border line width

        delaunay (bool or dict): overlay the Delaunay graph; if ``dict``, options are passed
            to :func:`~tramway.plot.mesh.plot_delaunay`

        colorbar (bool or str or dict): add a colour bar; if ``dict``, options are passed to
            :func:`~matplotlib.pyplot.colorbar`;
            setting colorbar to '*nice*' allows to produce a colorbar close to the figure
            of the same size as the figure

        unit/clabel (str): colorbar label, usually the unit of displayed feature

        alpha (float): alpha value of the cells

        colormap (str): colormap name; see also https://matplotlib.org/users/colormaps.html

        xlim (2-element sequence): lower and upper x-axis bounds

        ylim (2-element sequence): lower and upper y-axis bounds

        return_patches (bool): returns `PatchCollection` patches and the corresponding
            bin indices.

    Extra keyword arguments are passed to :func:`~matplotlib.collections.PatchCollection`.

    """
    coords = None
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            coords = values[[ col for col in 'xyzt' if col in values.columns ]]
            values = values[[ col for col in values.columns if col not in 'xyzt' ]]
            if values.shape[1] != 1:
                warn('multiple parameters available; mapping first one only', UserWarning)
        values = values.iloc[:,0] # to Series
    #values = pd.to_numeric(values, errors='coerce')

    # parse Delaunay-related arguments
    if delaunay:
        if not isinstance(delaunay, dict):
            delaunay = {}
        if linewidth and 'linewidth' not in delaunay:
            delaunay['linewidth'] = linewidth

    # turn the cells into polygons
    ids = []
    polygons = []
    if isinstance(cells, FiniteElements):

        ix, xy, ok = zip(*[ (i, c.center, bool(c)) for i, c in cells.items() ])
        ix, xy, ok = np.array(ix), np.array(xy), np.array(ok)
        if not (coords is None or np.all(np.isclose(xy, coords))): # debug
            print('coordinates mismatch')
            print('in map:')
            print(xy)
            print('in cells:')
            print(coords)
        rp = bb = voronoi = None
        for c in range(xy.shape[0]):
            if ok[c]:
                vertices, voronoi, bb, rp = cell_to_polygon(
                    c, xy, voronoi, bb, rp, True)
                polygons.append(Polygon(vertices, True))

    elif isinstance(cells, Partition) and isinstance(cells.tessellation, Voronoi):

        xy = cells.tessellation.cell_centers
        # copy/paste from below
        if not xlim or not ylim:
            xy_min, _, xy_max, _ = _bounding_box(cells, xy)
            if not xlim:
                xlim = (xy_min[0], xy_max[0])
            if not ylim:
                ylim = (xy_min[1], xy_max[1])
        ix = np.arange(xy.shape[0])
        from tramway.tessellation.hexagon import HexagonalMesh
        from tramway.tessellation.kdtree import KDTreeMesh
        if isinstance(cells.tessellation, (RegularMesh, HexagonalMesh, KDTreeMesh)):
            vertices, cell_vertices, Av = cells.tessellation.vertices, cells.tessellation.cell_vertices, cells.tessellation.vertex_adjacency.tocsr()
        else:
            try:
                vertices, cell_vertices, Av = box_voronoi_2d(cells.tessellation, xlim, ylim)
            except AssertionError:
                warn('could not fix the borders', RuntimeWarning)
                vertices, cell_vertices, Av = cells.tessellation.vertices, cells.tessellation.cell_vertices, cells.tessellation.vertex_adjacency.tocsr()
        try:
            ok = 0 < cells.location_count
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(traceback.format_exc())
            ok = np.ones(ix.size, dtype=bool)
        if cells.tessellation.cell_label is not None:
            ok = np.logical_and(ok, 0 < cells.tessellation.cell_label)
        map_defined = np.zeros_like(ok)
        map_defined[values.index] = True
        ok[np.logical_not(map_defined)] = False
        ok[ok] = np.logical_not(np.isnan(values.loc[ix[ok]].values))
        for i in ix[ok]:
            vs = cell_vertices[i].tolist()
            # order the vertices so that they draw a polygon
            v0 = v = vs[0]
            vs = set(vs)
            _vertices = []
            #vvs = [] # debug
            while True:
                _vertices.append(vertices[v])
                #vvs.append(v)
                vs.remove(v)
                if not vs:
                    break
                ws = set(Av.indices[Av.indptr[v]:Av.indptr[v+1]]) & vs
                if not ws:
                    ws = set(Av.indices[Av.indptr[v0]:Av.indptr[v0+1]]) & vs
                    if ws:
                        _vertices = _vertices[::-1]
                    else:
                        #print((v, vs, vvs, [Av.indices[Av.indptr[v]:Av.indptr[v+1]] for v in vs]))
                        warn('cannot find a path that connects all the vertices of a cell', RuntimeWarning)
                        if try_fix_corners:
                            vs = vertices[cell_vertices[i]]
                            if len(vs) != 3:
                                _min = vs.min(axis=0) == np.r_[xlim[0],ylim[0]]
                                _max = vs.max(axis=0) == np.r_[xlim[1],ylim[1]]
                                if _min[0]:
                                    vx = v0x = xlim[0]
                                    v1x = vs[:,0].max()
                                elif _max[0]:
                                    vx = v0x = xlim[1]
                                    v1x = vs[:,0].min()
                                else:
                                    vx = None
                                if _min[1]:
                                    vy = v1y = ylim[0]
                                    v0y = vs[:,1].max()
                                elif _max[1]:
                                    vy = v1y = ylim[1]
                                    v0y = vs[:,1].min()
                                else:
                                    vy = None
                                if vx is None or vy is None:
                                    vs = None
                                else:
                                    vs = np.array([[v0x,v0y],[vx,vy],[v1x,v1y]])
                            if vs is not None:
                                polygons.append(Polygon(vs, True))
                                ids.append(i)
                        break
                v = ws.pop()
            #
            if _vertices:
                _vertices = np.vstack(_vertices)
                polygons.append(Polygon(_vertices, True))
                ids.append(i)
    else:
        _type = repr(type(cells))
        if _type.endswith("'>"):
            _type = _type.split("'")[1]
        try:
            _nested_type = repr(type(cells.tessellation))
            if _nested_type.endswith("'>"):
                _nested_type = _nested_type.split("'")[1]
            raise TypeError('wrong type for `cells`: {}<{}>'.format(_type, _nested_type))
        except AttributeError:
            raise TypeError('wrong type for `cells`: {}'.format(_type))

    if not ids:
        ids = ix[ok]
    scalar_map = values.loc[ids].values

    #print(np.nonzero(~ok)[0])

    try:
        if np.any(np.isnan(scalar_map)):
            #print(np.nonzero(np.isnan(scalar_map)))
            msg = 'NaN found'
            try:
                warn(msg, NaNWarning)
            except:
                print('warning: {}'.format(msg))
            scalar_map[np.isnan(scalar_map)] = 0
    except TypeError: # help debug
        print(scalar_map)
        print(scalar_map.dtype)
        raise

    if figure is None:
        import matplotlib.pyplot as plt
        figure = plt.gcf() # before PatchCollection
    if axes is None:
        axes = figure.gca()
    # draw patches
    patch_kwargs = kwargs
    if alpha is not False:
        if alpha is None:
            alpha = .9
        patch_kwargs['alpha'] = alpha
    if colormap is not None:
        cmap = patch_kwargs.get('cmap', None)
        if cmap is None:
            patch_kwargs['cmap'] = colormap
        elif colormap != cmap:
            warn('both cmap and colormap arguments are passed with different values', RuntimeWarning)
    patches = PatchCollection(polygons, linewidth=linewidth, **patch_kwargs)
    patches.set_array(scalar_map)
    if clim is not None:
        patches.set_clim(clim)
    axes.add_collection(patches)

    obj = None
    if delaunay or isinstance(delaunay, dict):
        if not isinstance(delaunay, dict):
            delaunay = {}
        try:
            import tramway.plot.mesh as mesh
            obj = mesh.plot_delaunay(cells, centroid_style=None,
                axes=axes, **delaunay)
        except:
            import traceback
            traceback.print_exc()

    if not xlim or not ylim:
        xy_min, _, xy_max, _ = _bounding_box(cells, xy)
        if not xlim:
            xlim = (xy_min[0], xy_max[0])
        if not ylim:
            ylim = (xy_min[1], xy_max[1])
    axes.set_xlim(*xlim)
    axes.set_ylim(*ylim)
    if aspect is not None:
        axes.set_aspect(aspect)

    if colorbar:
        if colorbar=='nice':
            # make the colorbar closer to the plot and same size
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            try:
                plt
            except NameError:
                import matplotlib.pyplot as plt
            try:
                gca_bkp = plt.gca()
                divider = make_axes_locatable(figure.gca())
                cax = divider.append_axes("right", size="5%", pad=0.05)
                _colorbar = figure.colorbar(patches, cax=cax)
                plt.sca(gca_bkp)
            except AttributeError as e:
                warn(e.args[0], RuntimeWarning)
        else:
            if not isinstance(colorbar, dict):
                colorbar = {}
            try:
                _colorbar = figure.colorbar(patches, ax=axes, **colorbar)
            except AttributeError as e:
                warn(e.args[0], RuntimeWarning)
        if clabel:
            unit = clabel
        if unit:
            _colorbar.set_label(unit)

    if return_patches:
        return patches, np.asarray(ids)
    else:
        return obj



def field_map_2d(cells, values, angular_width=30.0, overlay=False,
        aspect=None, figure=None, axes=None,
        cell_arrow_ratio=None,
        markercolor='y', markeredgecolor='k', markeralpha=0.8, markerlinewidth=None,
        transform=None, inferencemap=False,
        **kwargs):
    """
    Plot a 2D field (vector) map as arrows.

    Arguments:

        cells (Partition or FiniteElements): spatial description of the cells

        values (pandas.DataFrame or numpy.ndarray): value at each cell, represented as a colour

        angular_width (float): angle of the tip of the arrows

        overlay (bool): do not plot the amplitude as a scalar map

        aspect (str): passed to :func:`~matplotlib.axes.Axes.set_aspect`

        figure (matplotlib.figure.Figure): figure handle

        axes (matplotlib.axes.Axes): axes handle

        cell_arrow_ratio (float): size of the largest arrow relative to the median
            inter-cell distance; default is ``0.4`` if `inferencemap` is ``True``,
            else ``1``

        markercolor (str): colour of the arrows

        markeredgecolor (str): colour of the border of the arrows

        markeralpha (float): alpha value of the arrows

        markerlinewidth (float): line width of the border of the arrows

        transform ('log' or callable): if `overlay` is ``False``,
            transform applied to the amplitudes as a NumPy array

        inferencemap (bool): if ``True``, the arrow length only depends on the cell size

    Extra keyword arguments are passed to :func:`scalar_map_2d` if called.

    If `overlay` is ``True``, the *marker*-prefixed arguments can be renamed without
    the *marker* prefix.
    These arguments can only be keyworded.

    """
    try:
        force_amplitude = values.pow(2).sum(1).apply(np.sqrt)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        if not overlay:
            warn('cannot compute the amplitude; setting `overlay` to True', RuntimeWarning)
            overlay = True
    if figure is None:
        import matplotlib.pyplot as plt
        figure = plt.gcf()
    if axes is None:
        axes = figure.gca()
    if overlay:
        if markercolor is None and 'color' in kwargs:
            markercolor = kwargs['color']
        if markeredgecolor is None and 'edgecolor' in kwargs:
            markeredgecolor = kwargs['edgecolor']
        if markeralpha is None and 'alpha' in kwargs:
            markeralpha = kwargs['alpha']
        if markerlinewidth is None and 'linewidth' in kwargs:
            markerlinewidth = kwargs['linewidth']
    else:
        if transform is None:
            transform = lambda a: a
        elif transform == 'log':
            transform = np.log
        obj = scalar_map_2d(cells, transform(force_amplitude),
            figure=figure, axes=axes, **kwargs)
    if aspect is not None:
        axes.set_aspect(aspect)
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    if axes.get_aspect() == 'equal':
        aspect_ratio = 1
    else:
        aspect_ratio = (xmax - xmin) / (ymax - ymin)
    # identify the visible cell centers
    if isinstance(cells, Tessellation):
        pts = cells.cell_centers
    elif isinstance(cells, FiniteElements):
        pts = np.vstack([ cells[i].center for i in cells ])#values.index ])
    elif isinstance(cells, Partition):
        assert isinstance(cells.tessellation, Tessellation)
        pts = cells.tessellation.cell_centers#[values.index]
    inside = (xmin<=pts[:,0]) & (pts[:,0]<=xmax) & (ymin<=pts[:,1]) & (pts[:,1]<=ymax)
    # compute the distance between adjacent cell centers
    if isinstance(cells, Delaunay):
        A = cells.cell_adjacency
    elif isinstance(cells, FiniteElements):
        A = cells.adjacency
    elif isinstance(cells, Partition) and isinstance(cells.tessellation, Delaunay):
        A = cells.tessellation.cell_adjacency
    if inferencemap:
        if cell_arrow_ratio is None:
            cell_arrow_ratio = 1.
        A = A.tocsr()
        scale = np.full(A.shape[0], np.nan, dtype=pts.dtype)
        for i in values.index:
            if not inside[i]:
                continue
            js = A.indices[A.indptr[i]:A.indptr[i+1]]
            pts_i, pts_j = pts[[i]], pts[js[inside[js]]]
            if pts_j.size:
                inter_cell_distance = pts_i - pts_j
                inter_cell_distance = np.sqrt(np.sum(inter_cell_distance * inter_cell_distance, axis=1))
                scale[i] = np.nanmean(inter_cell_distance)
    else:
        if cell_arrow_ratio is None:
            cell_arrow_ratio = .4 # backward compatibility
        A = sparse.triu(A, format='coo')
        I, J = A.row, A.col
        _inside = inside[I] & inside[J]
        pts_i, pts_j = pts[I[_inside]], pts[J[_inside]]
        inter_cell_distance = pts_i - pts_j
        inter_cell_distance = np.sqrt(np.sum(inter_cell_distance * inter_cell_distance, axis=1))
        # scale force amplitude
        large_arrow_length = np.nanmax(force_amplitude[inside[values.index]]) # TODO: clipping
        scale = np.nanmedian(inter_cell_distance) / (large_arrow_length * cell_arrow_ratio)
    #
    dw = float(angular_width) / 2.0
    t = tan(radians(dw))
    t = np.array([[0.0, -t], [t, 0.0]])
    markers = []
    for i in values.index:
        center = pts[i]
        radius = force_amplitude[i]
        if inferencemap:
            f = np.asarray(values.loc[i])
            f *= scale[i] / max(1e-16, np.sqrt(np.sum(f * f)))
        else:
            f = np.asarray(values.loc[i]) * scale
        #fx, fy = f
        #angle = degrees(atan2(fy, fx))
        #markers.append(Wedge(center, radius, angle - dw, angle + dw))
        base, vertex = center + np.outer([-1./3, 2./3], f)
        ortho = np.dot(t, f)
        vertices = np.stack((vertex, base + ortho, base - ortho), axis=0)
        #vertices[:,0] = center[0] + aspect_ratio * (vertices[:,0] - center[0])
        markers.append(Polygon(vertices, True))

    patches = PatchCollection(markers, facecolor=markercolor, edgecolor=markeredgecolor,
        alpha=markeralpha, linewidth=markerlinewidth)
    axes.add_collection(patches)

    #axes.set_xlim(xmin, xmax)
    #axes.set_ylim(ymin, ymax)

    if not overlay and obj:
        return obj


def scalar_map_3d(cells, values, aspect=None, clim=None, figure=None, axes=None,
        colorbar=True, alpha=None, colormap=None, unit=None, clabel=None,
        xlim=None, ylim=None, zlim=None, triangulation_depth=2, **kwargs):
    """
    Plot a 2D scalar map as a colourful 3D surface.

    Arguments:

        cells (Tessellation or Partition): spatial description of the cells

        values (pandas.DataFrame, numpy.ndarray or Maps): feature value at each cell,
            that will be represented as a colour

        aspect (str): passed to :func:`~matplotlib.axes.Axes.set_aspect`

        clim (2-element sequence): passed to :func:`~matplotlib.cm.ScalarMappable.set_clim`;
            note `clim` affects colour, not height.

        figure (matplotlib.figure.Figure): figure handle

        axes (matplotlib.axes.Axes): axes handle

        colorbar (bool or str or dict): add a colour bar; if ``dict``, options are passed to
            :func:`~matplotlib.pyplot.colorbar`;
            setting colorbar to '*nice*' allows to produce a colorbar close to the figure
            of the same size as the figure

        unit/clabel (str): colorbar label, usually the unit of displayed feature

        alpha (float): alpha value of the cells

        colormap (str): colormap name; see also https://matplotlib.org/users/colormaps.html

        xlim (2-element sequence): lower and upper x-axis bounds

        ylim (2-element sequence): lower and upper y-axis bounds

        zlim (2-element sequence): lower and upper z-axis bounds;
            note `zlim` affects height, not colour.

    Extra keyword arguments are passed to :func:`~matplotlib.collections.PatchCollection`.

    """
    coords = None
    if isinstance(values, Maps):
        values = values.maps
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            coords = values[[ col for col in 'xyzt' if col in values.columns ]]
            values = values[[ col for col in values.columns if col not in 'xyzt' ]]
            if values.shape[1] != 1:
                warn('multiple parameters available; mapping first one only', UserWarning)
        values = values.iloc[:,0] # to Series

    try:
        mesh = cells.tessellation
    except AttributeError:
        mesh = cells
    try:
        mesh = mesh.spatial_mesh
    except AttributeError:
        pass
    xy = centers = mesh.cell_centers
    adjacency = mesh.simplified_adjacency(format='csr')

    scalar_map = np.full(centers.shape[0], np.nan)
    scalar_map[values.index] = values.values
    elevation = np.array(scalar_map)
    if zlim:
        elevation[elevation < zlim[0]] = zlim[0]
        elevation[zlim[1] < elevation] = zlim[1]
    colour = np.array(scalar_map)
    if clim:
        colour[colour < clim[0]] = clim[0]
        colour[clim[1] < colour] = clim[1]

    if figure is None:
        import matplotlib.pyplot as plt
        figure = plt.gcf() # before PatchCollection
    if axes is None:
        import mpl_toolkits.mplot3d as plt3
        axes = plt3.Axes3D(figure)

    obj = None

    if not xlim or not ylim:
        xy_min, _, xy_max, _ = _bounding_box(cells, xy)
        if not xlim:
            xlim = (xy_min[0], xy_max[0])
        if not ylim:
            ylim = (xy_min[1], xy_max[1])
    axes.set_xlim(*xlim)
    axes.set_ylim(*ylim)
    if aspect is not None:
        axes.set_aspect(aspect)

    tri = []
    ok = np.zeros(mesh.number_of_cells, dtype=bool)
    for i in range(mesh.number_of_cells):
        J = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i+1]]
        for j in J:
            if i < j:
                K = adjacency.indices[adjacency.indptr[j]:adjacency.indptr[j+1]]
                for k in K:
                    if j < k:
                        tri.append([i,j,k])
                        ok[i] = ok[j] = ok[k] = True
    tri = np.array(tri)
    #index_transform = np.full(mesh.number_of_cells, -1, dtype=int)
    #index_transform[ok] = np.arange(np.sum(ok))
    #import matplotlib.tri
    # if `plot_trisurf` could interpolate each triangle's colour:
    #tri_colour = [ np.mean(colour[_t]) for _t in tri ]
    #tri = matplotlib.tri.Triangulation(xy[ok,0], xy[ok,1], index_transform(tri))
    #axes.plot_trisurf(tri, elevation[ok], color=tri_colour, **kwargs)

    import itertools
    def split_triangle(_rdepth, _xy, *_zs):
        if _rdepth == 0:
            return tuple( [_a] for _a in (_xy,) + _zs )
        else:
            return tuple( itertools.chain(*_a) for _a in zip(
                    split_triangle(_rdepth-1,
                        np.stack((_xy[0], (_xy[0] + _xy[1]) * .5, (_xy[0] + _xy[2]) * .5), axis=0),
                        *[ np.r_[_z[0], (_z[0] + _z[1]) * .5, (_z[0] + _z[2]) * .5] for _z in _zs ]),
                    split_triangle(_rdepth-1,
                        np.stack(((_xy[0] + _xy[1]) * .5, _xy[1], (_xy[1] + _xy[2]) * .5), axis=0),
                        *[ np.r_[(_z[0] + _z[1]) * .5, _z[1], (_z[1] + _z[2]) * .5] for _z in _zs ]),
                    split_triangle(_rdepth-1,
                        np.stack(((_xy[0] + _xy[1]) * .5, (_xy[0] + _xy[2]) * .5, (_xy[1] + _xy[2]) * .5), axis=0),
                        *[ np.r_[(_z[0] + _z[1]) * .5, (_z[0] + _z[2]) * .5, (_z[1] + _z[2]) * .5] for _z in _zs ]),
                    split_triangle(_rdepth-1,
                        np.stack(((_xy[0] + _xy[2]) * .5, (_xy[1] + _xy[2]) * .5, _xy[2]), axis=0),
                        *[ np.r_[(_z[0] + _z[2]) * .5, (_z[1] + _z[2]) * .5, _z[2]] for _z in _zs ]),
                    ))

    subtri_xyz = []
    subtri_colour = []
    for ijk in tri:
        boundary_xy = xy[ijk]
        boundary_elevation = elevation[ijk]
        boundary_colour = colour[ijk]
        for _subtri_xy, _subtri_elevation, _subtri_colour in \
                zip(*split_triangle(triangulation_depth, boundary_xy, boundary_elevation, boundary_colour)):
            subtri_xyz.append(np.hstack((_subtri_xy, _subtri_elevation[:,np.newaxis])))
            subtri_colour.append(np.mean(_subtri_colour))
    subtri_xyz = np.vstack(subtri_xyz)
    #subtri = matplotlib.tri.Triangulation(
    #        subtri_xyz[:,0], subtri_xyz[:,1],
    #        np.reshape(np.arange(len(subtri_xyz)), (-1, 3), order='C'))
    #print(subtri.get_masked_triangles())
    subtri_xyz = np.stack([np.reshape(c, (-1,3)) for c in subtri_xyz.T], axis=-1)

    #obj = axes.plot_trisurf(subtri, subtri_elevation, facecolors=subtri_colour, norm=norm, **kwargs)
    import mpl_toolkits.mplot3d.art3d as art
    subtri = art.Poly3DCollection(subtri_xyz, **kwargs)
    subtri.set_array(np.asarray(subtri_colour))

    axes.add_collection(subtri)
    #axes.auto_scale_xyz(subtri_xyz[:,0], subtri_xyz[:,1], subtri_xyz[:,2])
    if xlim is not None:
        axes.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        axes.set_ylim(ylim[0], ylim[1])
    if zlim is not None:
        axes.set_zlim(zlim[0], zlim[1])

    return obj



__all__ = ['cell_to_polygon', 'scalar_map_2d', 'field_map_2d', 'scalar_map_3d']

