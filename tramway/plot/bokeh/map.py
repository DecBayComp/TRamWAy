# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import bokeh.plotting as plt
from bokeh.models.ranges import Range1d
from tramway.core import *
import tramway.plot.map as mplt
import matplotlib as mpl
import numpy as np
import pandas as pd
import itertools


_long_colour_names = {
        'r':    'red',
        'g':    'green',
        'b':    'blue',
        'c':    'cyan',
        'm':    'magenta',
        'y':    'yellow',
        'k':    'black',
        }
def long_colour_name(c):
    if isinstance(c, str):
        c = _long_colour_names.get(c, c)
    return c

def plot_points(cells, style='circle', size=2, color=None, figure=None, **kwargs):
    """
    Plot molecule locations.

    If no colour is explicitly defined, colouring is based on the cell index.

    Arguments:

        cells (Partition):
            full partition

        style (str):
            point marker style

        size (float):
            point marker size (in screen units)

        color (str or numpy.ndarray):
            cell colours

        figure (bokeh.plotting.figure.Figure):
            figure handle

    returns:

        list: handles of the various clouds of points

    """
    # for compatibility with tramway.plot.mesh.plot_points
    try:
        kwargs.pop('min_count')
    except KeyError:
        pass

    if isinstance(cells, np.ndarray):
        points = cells
        label = None
    elif isinstance(cells, Partition):
        points = cells.descriptors(cells.points, asarray=True)
        label = cells.cell_index
        npts = points.shape[0]
        ncells = cells.location_count.size
        # if label is not a single index vector, convert it following
        # tessellation.base.Delaunay.cell_index with `preferred`='force index'.
        merge = nearest_cell(points, cells.tessellation.cell_centers)
        label = format_cell_index(cells.cell_index, format='array', \
            select=merge, shape=(npts, ncells))

    if isstructured(points):
        x = points['x']
        y = points['y']
    else:
        x = points[:,0]
        y = points[:,1]

    if figure is None:
        assert False
        try:
            figure = plt.curplot() # does not work
        except:
            figure = plt.figure()

    handles = []

    if label is None:
        if color is None:
            color = 'k'
        elif isinstance(color, (pd.Series, pd.DataFrame)):
            raise NotImplementedError
            color = np.asarray(color)
        if isinstance(color, np.ndarray):
            raise NotImplementedError
            cmin, cmax = np.min(color), np.max(color)
            color = (color - cmin) / (cmax - cmin)
            cmap = plt.get_cmap()
            color = [ cmap(c) for c in color ]
        else:
            color = long_colour_name(color)
        h = figure.scatter(x, y, marker=style, color=color, size=size, **kwargs)
        handles.append(h)
    else:
        L = np.unique(label)
        if color in [None, 'light']:
            if color == 'light' and 'alpha' not in kwargs:
                kwargs['alpha'] = .2
            if 2 < len(L):
                color = __colors__
                color = ['gray'] + \
                    list(itertools.islice(itertools.cycle(color), len(L)))
            elif len(L) == 2:
                color = ['gray', 'k']
            else:   color = 'k'
        elif isinstance(color, str) and L.size==1:
            color = [color]
        for i, l in enumerate(L):
            clr_i = long_colour_name(color[i])
            h = figure.scatter(x[label == l], y[label == l],
                    marker=style, color=clr_i, size=size, **kwargs)
            handles.append(h)

    ## resize window
    #try:
    #    figure.x_range = Range1d(*cells.bounding_box['x'].values)
    #    figure.y_range = Range1d(*cells.bounding_box['y'].values)
    #except AttributeError:
    #    pass
    #except ValueError:
    #    traceback.print_exc()

    return handles


def scalar_map_2d(cells, values, clim=None, figure=None, xlim=None, ylim=None, **kwargs):
    """
    Plot an interactive 2D scalar map as a colourful image.

    Arguments:

        cells (Partition): spatial description of the cells

        values (pandas.Series or pandas.DataFrame): feature value at each cell/bin,
            encoded into a colour

        clim (2-element sequence): passed to :func:`~matplotlib.cm.ScalarMappable.set_clim`

        figure (bokeh.plotting.figure.Figure): figure handle

        xlim (2-element sequence): lower and upper x-axis bounds

        ylim (2-element sequence): lower and upper y-axis bounds

    """
    if isinstance(values, pd.DataFrame):
        feature_name = values.columns[0]
        values = values.iloc[:,0] # to Series
    else:
        feature_name = None

    if figure is None:
        assert False
        figure = plt.figure()

    polygons = []

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
            polygons.append((vertices[:,0], vertices[:,1]))

    scalar_map = values.loc[ix[ok]].values
    clim = {} if clim is None else dict(vmin=clim[0], vmax=clim[1])
    scalar_map = mpl.colors.Normalize(**clim)(scalar_map)

    colors = [
            "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.viridis(scalar_map)
            ]
    patch_kwargs = dict(fill_color=colors, line_width=0)
    figure.patches(*zip(*polygons), **patch_kwargs)

    if not xlim or not ylim:
        xy_min, _, xy_max, _ = mplt._bounding_box(cells, xy)
        if not xlim:
            xlim = (xy_min[0], xy_max[0])
        if not ylim:
            ylim = (xy_min[1], xy_max[1])
    figure.x_range = Range1d(*xlim)
    figure.y_range = Range1d(*ylim)

    #plt.show(figure)


__all__ = ['plot_points', 'scalar_map_2d']
