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
from tramway.tessellation.base import Partition, format_cell_index
from tramway.plot.mesh import _graph_theme
import tramway.plot.map as mplt
import matplotlib as mpl
import numpy as np
import pandas as pd
import itertools
from warnings import warn
import scipy.sparse as sparse
from collections import defaultdict


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
    elif isinstance(cells, pd.DataFrame):
        points = cells[['x','y']].values
        label = None
    elif isinstance(cells, Partition):
        points = cells.descriptors(cells.points, asarray=True)
        label = cells.cell_index
        npts = points.shape[0]
        ncells = cells.location_count.size
        # if label is not a single index vector, convert it following
        # tessellation.base.Delaunay.cell_index with `preferred`='force index'.
        merge = mplt.nearest_cell(points, cells.tessellation.cell_centers)
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

def plot_trajectories(trajs, color=None, loc_style='circle', figure=None, **kwargs):
    """
    Plot trajectories.

    If no colour is explicitly defined, colouring is based on the trajectory index.

    Arguments:

        trajs (pandas.DataFrame):
            full partition

        loc_style (str):
            location marker style

        loc_size (float):
            location marker size (in screen units)

        color (str or numpy.ndarray):
            trajectory colours

        figure (bokeh.plotting.figure.Figure):
            figure handle

    returns:

        list: handles of the glyphs

    """
    loc_kwargs = {}
    line_kwargs = {}
    for attr in dict(kwargs):
        if attr.startswith('loc_'):
            loc_kwargs[attr[4:]] = kwargs.pop(attr)
        if attr.startswith('line_'):
            line_kwargs[attr[5:]] = kwargs.pop(attr)

    loc_kwargs.update(kwargs)
    line_kwargs.update(kwargs)

    lines_x, lines_y = [], []
    for _, df in trajs.groupby([trajs['n']]):
        lines_x.append(df['x'].values)
        lines_y.append(df['y'].values)

    if figure is None:
        assert False
        try:
            figure = plt.curplot() # does not work
        except:
            figure = plt.figure()

    handles = []

    if color is None:
        ntrajs = len(lines_x)
        if color in [None, 'light']:
            if color == 'light' and 'alpha' not in kwargs:
                kwargs['alpha'] = .2
            if 2 < ntrajs:
                color = __colors__
                color = ['gray'] + \
                    list(itertools.islice(itertools.cycle(color), ntrajs))
            elif ntrajs == 2:
                color = ['gray', 'k']
            else:   color = 'k'
        elif isinstance(color, str) and ntrajs==1:
            color = [color]
        for i, line in enumerate(zip(lines_x, lines_y)):
            line_x, line_y = line
            clr_i = long_colour_name(color[i])
            h = figure.line(line_x, line_y, color=clr_i, **line_kwargs)
            handles.append(h)
            h = figure.scatter(line_x, line_y, color=clr_i, marker=loc_style, **loc_kwargs)
            handles.append(h)
    else:
        color = long_colour_name(color)
        h = figure.multi_line(lines_x, lines_y, color=color, **line_kwargs)
        handles.append(h)
        h = figure.scatter(list(itertools.chain(*lines_x)), list(itertools.chain(*lines_y)),
                color=color, marker=loc_style, **loc_kwargs)
        handles.append(h)

    return handles



def scalar_map_2d(cells, values, clim=None, figure=None, delaunay=False, xlim=None, ylim=None, **kwargs):
    """
    Plot an interactive 2D scalar map as a colourful image.

    Arguments:

        cells (Partition): spatial description of the cells

        values (pandas.Series or pandas.DataFrame): feature value at each cell/bin,
            encoded into a colour

        clim (2-element sequence): passed to :func:`~matplotlib.cm.ScalarMappable.set_clim`

        figure (bokeh.plotting.figure.Figure): figure handle

        delaunay (bool or dict): overlay the Delaunay graph; if ``dict``, options are passed
            to :func:`~tramway.plot.bokeh.plot_delaunay`

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

    xy = cells.tessellation.cell_centers
    if not xlim or not ylim:
        xy_min, _, xy_max, _ = mplt._bounding_box(cells, xy)
        if not xlim:
            xlim = (xy_min[0], xy_max[0])
        if not ylim:
            ylim = (xy_min[1], xy_max[1])
    ix = np.arange(xy.shape[0])
    vertices, cell_vertices, Av = mplt.box_voronoi_2d(cells.tessellation, xlim, ylim)
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
                    break
            v = ws.pop()
        #
        if _vertices:
            _vertices = np.vstack(_vertices)
            polygons.append((_vertices[:,0], _vertices[:,1]))

    scalar_map = values.loc[ix[ok]].values
    clim = {} if clim is None else dict(vmin=clim[0], vmax=clim[1])
    scalar_map = mpl.colors.Normalize(**clim)(scalar_map)

    colors = [
            "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.viridis(scalar_map)
            ]
    patch_kwargs = dict(fill_color=colors, line_width=0)
    figure.patches(*zip(*polygons), **patch_kwargs)

    if delaunay or isinstance(delaunay, dict):
        if not isinstance(delaunay, dict):
            delaunay = {}
        plot_delaunay(cells, figure=figure, **delaunay)

    figure.x_range = Range1d(*xlim)
    figure.y_range = Range1d(*ylim)

    #plt.show(figure)


def plot_delaunay(cells, labels=None, color=None, style='-',
        figure=None, linewidth=1, **kwargs):
    """
    Delaunay plot.

    Arguments:

        cells (Partition):
            full partition

        labels (numpy.ndarray):
            numerical labels for cell adjacency relationship

        color (str):
            single-character colours in a string, e.g. 'rrrbgy'

        style (str):
            line style

        figure (bokeh.plotting.figure.Figure):
            figure handle

        linewidth (int):
            line width

    Returns:

        tuple: list of handles of the plotted edges,
            handle of the plotted centroids
    """
    if figure is None:
        raise NotImplementedError
    try:
        tessellation = cells.tessellation
    except AttributeError:
        tessellation = cells

    vertices = tessellation.cell_centers

    labels, color = _graph_theme(tessellation, labels, color, True)

    # if asymmetric, can be either triu or tril
    A = sparse.triu(tessellation.cell_adjacency, format='coo')
    I, J, K = A.row, A.col, A.data
    if not I.size:
        A = sparse.tril(tessellation.cell_adjacency, format='coo')
        I, J, K = A.row, A.col, A.data

    by_color = defaultdict(list)
    edge_handles = []

    # plot delaunay
    for i, j, k in zip(I, J, K):
        x, y = zip(vertices[i], vertices[j])
        if labels is None:
            c = 0
        else:
            label = tessellation.adjacency_label[k]
            try:
                c = labels.index(label)
            except ValueError:
                continue
        by_color[c].append((x, y))

    for c in by_color:
        xy = by_color[c]
        X = np.zeros((len(xy) * 3,))
        Y = np.empty((len(xy) * 3,))
        Y[:] = np.nan
        i = 0
        for x, y in xy:
            I = slice(i*3, i*3+2)
            X[I], Y[I] = x, y
            i += 1
        h = figure.line(X, Y, line_dash=_line_style_to_dash_pattern(style),
            line_color=long_colour_name(color[c if color[1:] else 0]),
            line_width=linewidth)
        edge_handles.append(h)

    return edge_handles


def _line_style_to_dash_pattern(style):
    return {
        '-':    [],
        '-.':   'dashdot',
        ':':    'dotted',
        '.-':   'dotdash',
        '--':   'dashed',
        }.get(style, style)



__all__ = ['plot_points', 'plot_trajectories', 'plot_delaunay', 'scalar_map_2d']

