# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
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
from tramway.inference import Distributed
from matplotlib.patches import Polygon, Wedge
from matplotlib.collections import PatchCollection
import scipy.spatial
import scipy.sparse as sparse
from warnings import warn


def _bounding_box(cells, xy):
    # bounding box
    try:
        bounding_box = cells.bounding_box[['x', 'y']]
        xy_min, xy_max = bounding_box.values
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
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


def scalar_map_2d(cells, values, aspect=None, clim=None, figure=None, axes=None, linewidth=1,
        delaunay=False, colorbar=True, alpha=None, colormap=None, unit=None, clabel=None,
        xlim=None, ylim=None, **kwargs):
    """
    Plot a 2D scalar map as a colourful image.

    Arguments:

        cells (CellStats or Distributed): spatial description of the cells

        values (pandas.DataFrame or numpy.ndarray): feature value at each cell,
            that will be represented as a colour

        aspect (str): passed to :func:`~matplotlib.axes.Axes.set_aspect`

        clim (2-element sequence): passed to :func:`~matplotlib.cm.ScalarMappable.set_clim`

        figure (matplotlib.figure.Figure): figure handle

        axes (matplotlib.axes.Axes): axes handle

        linewidth (int): cell border line width

        delaunay (bool or dict): overlay the Delaunay graph; if ``dict``, options are passed
            to :func:`~tramway.core.plot.mesh.plot_delaunay`

        colorbar (bool or str or dict): add a colour bar; if ``dict``, options are passed to
            :func:`~matplotlib.pyplot.colorbar`;
            setting colorbar to '*nice*' allows to produce a colorbar close to the figure
            of the same size as the figure

        unit/clabel (str): colorbar label, usually the unit of displayed feature

        alpha (float): alpha value of the cells

        colormap (str): colormap name; see also https://matplotlib.org/users/colormaps.html

        xlim (2-element sequence): lower and upper x-axis bounds

        ylim (2-element sequence): lower and upper y-axis bounds

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
    polygons = []
    if isinstance(cells, Distributed):

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

    elif isinstance(cells, CellStats) and isinstance(cells.tessellation, Voronoi):

        Av = cells.tessellation.vertex_adjacency.tocsr()
        xy = cells.tessellation.cell_centers
        ix = np.arange(xy.shape[0])
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
            vs = cells.tessellation.cell_vertices[i].tolist()
            # order the vertices so that they draw a polygon
            v0 = v = vs[0]
            vs = set(vs)
            vertices = []
            #vvs = [] # debug
            while True:
                vertices.append(cells.tessellation.vertices[v])
                #vvs.append(v)
                vs.remove(v)
                if not vs:
                    break
                ws = set(Av.indices[Av.indptr[v]:Av.indptr[v+1]]) & vs
                if not ws:
                    ws = set(Av.indices[Av.indptr[v0]:Av.indptr[v0+1]]) & vs
                    if ws:
                        vertices = vertices[::-1]
                    else:
                        #print((v, vs, vvs, [Av.indices[Av.indptr[v]:Av.indptr[v+1]] for v in vs]))
                        warn('cannot find a path that connects all the vertices of a cell', RuntimeWarning)
                        break
                v = ws.pop()
            #
            if vertices:
                vertices = np.vstack(vertices)
                polygons.append(Polygon(vertices, True))
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

    scalar_map = values.loc[ix[ok]].values

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

    return obj



def field_map_2d(cells, values, angular_width=30.0, overlay=False,
        aspect=None, figure=None, axes=None,
        cell_arrow_ratio=0.4,
        markercolor='y', markeredgecolor='k', markeralpha=0.8, markerlinewidth=None,
        transform=None,
        **kwargs):
    """
    Plot a 2D field (vector) map as arrows.

    Arguments:

        cells (CellStats or Distributed): spatial description of the cells

        values (pandas.DataFrame or numpy.ndarray): value at each cell, represented as a colour

        angular_width (float): angle of the tip of the arrows

        overlay (bool): do not plot the amplitude as a scalar map

        aspect (str): passed to :func:`~matplotlib.axes.Axes.set_aspect`

        figure (matplotlib.figure.Figure): figure handle

        axes (matplotlib.axes.Axes): axes handle

        cell_arrow_ratio (float): size of the largest arrow relative to the median
            inter-cell distance

        markercolor (str): colour of the arrows

        markeredgecolor (str): colour of the border of the arrows

        markeralpha (float): alpha value of the arrows

        markerlinewidth (float): line width of the border of the arrows

        transform ('log' or callable): if `overlay` is ``False``,
            transform applied to the amplitudes as a NumPy array

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
    if isinstance(cells, Distributed):
        pts = np.vstack([ cells[i].center for i in cells ])#values.index ])
    elif isinstance(cells, CellStats):
        assert isinstance(cells.tessellation, Tessellation)
        pts = cells.tessellation.cell_centers#[values.index]
    inside = (xmin<=pts[:,0]) & (pts[:,0]<=xmax) & (ymin<=pts[:,1]) & (pts[:,1]<=ymax)
    # compute the distance between adjacent cell centers
    if isinstance(cells, Distributed):
        A = cells.adjacency
    elif isinstance(cells, CellStats) and isinstance(cells.tessellation, Delaunay):
        A = cells.tessellation.cell_adjacency
    A = sparse.triu(A, format='coo')
    I, J = A.row, A.col
    _inside = inside[I] & inside[J]
    pts_i, pts_j = pts[I[_inside]], pts[J[_inside]]
    inter_cell_distance = pts_i - pts_j
    inter_cell_distance = np.sqrt(np.sum(inter_cell_distance * inter_cell_distance, axis=1))
    # scale force amplitude
    large_arrow_length = np.max(force_amplitude[inside[values.index]]) # TODO: clipping
    scale = np.nanmedian(inter_cell_distance) / (large_arrow_length * cell_arrow_ratio)
    #
    dw = float(angular_width) / 2.0
    t = tan(radians(dw))
    t = np.array([[0.0, -t], [t, 0.0]])
    markers = []
    for i in values.index:
        center = pts[i]
        radius = force_amplitude[i]
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

__all__ = ['cell_to_polygon', 'scalar_map_2d', 'field_map_2d']

