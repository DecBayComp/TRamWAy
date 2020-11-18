# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import *
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
import itertools
from copy import deepcopy
import scipy.sparse as sparse
from scipy.spatial.distance import cdist
from ..core import *
from ..tessellation.base import dict_to_sparse, format_cell_index, nearest_cell, Partition, Tessellation
import traceback
from collections import defaultdict


__colors__ = ['darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkviolet', 'deeppink', 'deepskyblue', 'dodgerblue', 'firebrick', 'forestgreen', 'gold', 'goldenrod', 'hotpink', 'indianred', 'indigo', 'lightblue', 'lightcoral', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightsteelblue', 'limegreen', 'maroon', 'mediumaquamarine', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navajowhite', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', '#663399', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'skyblue', 'slateblue', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'yellowgreen']


def plot_points(cells, min_count=None, style='.', size=8, color=None, axes=None, **kwargs):
    """
    Plot 2D points coloured by associated cell.

    Arguments:

        cells (Partition or FiniteElements):
            full partition

        min_count (int):
            discard cells with less than this number of associated points

        style (str):
            point marker style

        size (int):
            point marker size

        color (str or numpy.ndarray):
            cell colours

    Returns:

        list: handles of the various clouds of points

    Extra keyword arguments are passed to *matplotlib* 's *scatter* or *plot*.
    """
    if axes is None:
        import matplotlib.pyplot as plt
        axes = plt
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
        if min_count:
            cell_mask = min_count <= cells.location_count
            label = np.array(label) # copy
            label[np.logical_not(cell_mask[cells.cell_index])] = -1
            #label = cell_mask[cells.cell_index]
    else:#if isinstance(cells, FiniteElements):

        # fully distinct implementation
        handles = []
        if color == 'light' and 'alpha' not in kwargs:
            kwargs['alpha'] = .2
        color = __colors__
        color = list(itertools.islice(itertools.cycle(color), len(cells)))
        for i in cells:
            points = cells[i].origins
            if isinstance(points, pd.DataFrame):
                points = points[['x', 'y']].values
            h = axes.plot(points[:,0], points[:,1], style, color=color[i],
                markersize=size, **kwargs)
            assert not h[1:]
            handles.append(h[0])
        return handles


    # original implementation for the not-FiniteElements case

    if isstructured(points):
        x = points['x']
        y = points['y']
    else:
        x = points[:,0]
        y = points[:,1]

    handles = []

    if label is None or (isinstance(color, str) and len(color)==1):
        if color is None:
            color = 'k'
        elif isinstance(color, (pd.Series, pd.DataFrame)):
            color = np.asarray(color)
        if isinstance(color, np.ndarray):
            cmin, cmax = np.min(color), np.max(color)
            color = (color - cmin) / (cmax - cmin)
            cmap = plt.get_cmap()
            color = [ cmap(c) for c in color ]
        handles.append(axes.scatter(x, y, color=color, marker=style, s=size, **kwargs))
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
        for i, l in enumerate(L):
            handles.append(axes.plot(x[label == l], y[label == l],
                style, color=color[i], markersize=size, **kwargs))

    # resize window
    try:
        axes.axis(cells.bounding_box[['x', 'y']].values.flatten('F'))
    except AttributeError:
        pass
    except ValueError:
        print(traceback.format_exc())

    return handles


def plot_voronoi(cells, labels=None, color=None, style='-', centroid_style='g+', negative=None,
        linewidth=1, fallback_color='gray', verbose=True, axes=None):
    """
    Voronoi plot.

    Arguments:

        cells (Partition):
            full partition

        labels (numpy.ndarray):
            numerical labels for cell adjacency relationship

        color (str):
            single-character colours in a string, e.g. 'rrrbgy'

        style (str):
            line style

        centroid_style (str):
            marker style of the cell centers

        negative (any):
            if ``None``, do not plot edges corresponding to negative adjacency labels

        linewidth (int):
            line width

        fallback_color (str):
            colour for missing edges

        verbose (bool):
            print message about missing edges

    Returns:

        tuple: list of handles of the plotted edges,
            handle of the plotted centroids
    """
    if axes is None:
        import matplotlib.pyplot as plt
        axes = plt
    vertices = cells.tessellation.vertices
    labels, color = _graph_theme(cells.tessellation, labels, color, negative)
    try:
        color += 'w'
    except TypeError:
        if not isinstance(color, list):
            color = [color]
        color.append('w')
    try:
        special_edges = cells.tessellation.candidate_edges
        #points = cells.descriptors(cells.points, asarray=True)
    except:
        special_edges = {}
    c = 0 # if cells.tessellation.adjacency_label is None
    fallback_clr = 'k'
    edge_handles, centroid_handle = [], None
    # plot voronoi
    #plt.plot(vertices[:,0], vertices[:,1], 'b+')
    if cells.tessellation.adjacency_label is not None or special_edges:
        n_cells = cells.tessellation._cell_centers.shape[0]
        n_vertices = vertices.shape[0]
        cell_vertex = dict_to_sparse(cells.tessellation.cell_vertices, \
                shape=(n_cells, n_vertices)).tocsc()
        adjacency = cells.tessellation.cell_adjacency.tocsr()
    Av = sparse.tril(cells.tessellation.vertex_adjacency, format='coo')
    d2 = np.sum((vertices[Av.row[0]] - vertices[Av.col[0]])**2)
    for u, v in zip(Av.row, Av.col):
        x, y = vertices[[u, v]].T
        if cells.tessellation.adjacency_label is not None or special_edges:
            try:
                a, b = set(cell_vertex[:,u].indices) & set(cell_vertex[:,v].indices)
                # adjacency may contain explicit zeros
                js = list(adjacency.indices[adjacency.indptr[a]:adjacency.indptr[a+1]])
                # js.index(b) will fail if a and b are not adjacent
                edge_ix = adjacency.data[adjacency.indptr[a]+js.index(b)]
            except (ValueError, IndexError):
                if verbose:
                    print("vertices {} and {} do not match with a ridge".format(u, v))
                    print(traceback.format_exc())
                #continue
                c = -1
            else:
                if cells.tessellation.adjacency_label is not None:
                    try:
                        c = labels.index(cells.tessellation.adjacency_label[edge_ix])
                    except ValueError:
                        continue
        if 0 <= c:
            try:
                _clr = color[c]
            except IndexError:
                import warnings
                warnings.warn("too few colours: '{}'; index {:d} out of range".format(color, c), RuntimeWarning)
                _clr = fallback_color
        else:
            _clr = fallback_color
        h = axes.plot(x, y, style, color=_clr, linewidth=linewidth)
        assert not h[1:]
        edge_handles.append(h[0])

        # extra debug steps
        if special_edges and edge_ix in special_edges:
            #i, j, ii, jj = special_edges[edge_ix]
            #try:
            #       i = points[cells.cell_index == i][ii]
            #       j = points[cells.cell_index == j][jj]
            #except IndexError as e:
            #       print(e)
            #       continue
            i, j = special_edges[edge_ix]
            x_, y_ = zip(i, j)
            axes.plot(x_, y_, 'c-')
            x_, y_ = (i + j) / 2
            axes.text(x_, y_, str(edge_ix), \
                horizontalalignment='center', verticalalignment='center')

    centroids = cells.tessellation.cell_centers
    # plot cell centers
    if centroid_style:
        h = axes.plot(centroids[:,0], centroids[:,1], centroid_style)
        assert not h[1:]
        centroid_handle = h[0]

    # resize window
    try:
        axes.axis(cells.bounding_box[['x', 'y']].values.flatten('F'))
    except AttributeError:
        pass
    except ValueError:
        print(traceback.format_exc())

    return edge_handles, centroid_handle


def plot_delaunay(cells, labels=None, color=None, style='-', centroid_style='g+', negative=None,
        axes=None, linewidth=1, individual=False, fallback_color='gray'):
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

        centroid_style (str):
            marker style of the cell centers

        negative (any):
            if ``None``, do not plot edges corresponding to negative adjacency labels;
            if '*voronoi*', plot the corresponding Voronoi edge instead, for edges with
            negative labels

        axes (matplotlib.axes.Axes):
            axes where to plot

        linewidth (int):
            line width

        individual (bool):
            plot each edge independently; this generates a lot of handles and takes time

        fallback_color (str):
            colour for unexpected labels

    Returns:

        tuple: list of handles of the plotted edges,
            handle of the plotted centroids
    """
    if axes is None:
        import matplotlib.pyplot as plt
        axes = plt
    try:
        tessellation = cells.tessellation
    except AttributeError:
        tessellation = cells

    vertices = tessellation.cell_centers
    if negative == 'voronoi':
        voronoi = tessellation.cell_vertices

    labels, color = _graph_theme(tessellation, labels, color, negative)

    # if asymetric, can be either triu or tril
    A = sparse.triu(tessellation.cell_adjacency, format='coo')
    I, J, K = A.row, A.col, A.data
    if not I.size:
        A = sparse.tril(tessellation.cell_adjacency, format='coo')
        I, J, K = A.row, A.col, A.data

    if not individual:
        by_color = defaultdict(list)
    edge_handles, centroid_handle = [], None # handles

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
            if label <= 0:
                if negative == 'voronoi':
                    try:
                        vert_ids = set(tessellation.cell_vertices.get(i, [])) & set(tessellation.cell_vertices.get(j, []))
                        x, y = voronoi[vert_ids].T
                    except ValueError:
                        continue
        if individual:
            h = axes.plot(x, y, style, color=color[c], linewidth=linewidth)
            assert not h[1:]
            edge_handles.append(h)
        else:
            by_color[c].append((x, y))

    if not individual:
        if not color[1:]:
            _clr = color[0]
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
            if color[1:]:
                try:
                    _clr = color[c]
                except IndexError:
                    import warnings
                    warnings.warn('too few specified colours; at least {:d} needed'.format(c), RuntimeWarning)
                    _clr = fallback_color
            h = axes.plot(X, Y, style, color=_clr, linewidth=linewidth)
            assert not h[1:]
            edge_handles.append(h[0])

    # plot cell centers
    if centroid_style:
        h = axes.plot(vertices[:,0], vertices[:,1], centroid_style)
        assert not h[1:]
        centroid_handle = h[0]

    # resize window
    try:
        axes.axis(cells.bounding_box[['x', 'y']].values.flatten('F'))
    except AttributeError:
        pass
    except ValueError:
        print(traceback.format_exc())

    return edge_handles, centroid_handle


def _graph_theme(tess, labels, color, negative):
    if tess.adjacency_label is None:
        if not color:
            color = 'r'
    else:
        if labels is None:
            labels = np.unique(tess.adjacency_label).tolist()
    if labels is not None:
        if negative is None:
            labels = [ l for l in labels if 0 < l ]
            nnp = 0
        else:
            nnp = len([ l for l in labels if l <= 0 ]) # number of non-positive labels
    if not color:
        neg_color = 'cym'
        pos_color = 'rgb'
        labels.sort()
        color = neg_color[:nnp] + pos_color
    return (labels, color)


def plot_distributed(cells, vertex_color='g', vertex_style='x', edge_color='r',
        arrow_size=5, arrow_color='y'):
    """
    Plot a :class:`~tramway.inference.base.FiniteElements` object as a mesh.

    Arguments:

        cells (tramway.inference.base.FiniteElements):
            mesh prepared for the inference

        vertex_color (str): colour for the cell centers

        vertex_style (str): marker style for the cell centers

        edge_color (str): colour of the edge between adjacent cells

        arrow_size (int): size of the arrows along the edges

        arrow_color (str): colour of the arrows along the edges

    Returns:

        tuple: list of handles of the edges,
            handle of the cell centers (vertices),
            list of handles of the arrow segments

    `plot_distributed` is similar to `plot_delaunay` but takes a :class:`~tramway.inference.base.FiniteElements` object instead.
    """
    import matplotlib.pyplot as plt

    centers = np.vstack([ cells[i].center for i in cells ])
    _min, _max = np.min(centers), np.max(centers)

    plot_arrows = arrow_size is not None and 0 < arrow_size and arrow_color is not None
    if plot_arrows:
        arrow_size = float(arrow_size) * 1e-3 * (_max - _min)
        half_base = arrow_size / sqrt(5.)

    edges, arrows, vertices = [], [], None
    plotted = defaultdict(list)
    for i in cells:
        x = cells[i].center
        for j in cells.adjacency[i].indices:
            y = cells[j].center

            # edge
            if j not in plotted[i]:
                h = plt.plot([x[0], y[0]], [x[1], y[1]], edge_color+'-')
                assert not h[1:]
                edges.append(h[0])
                plotted[i].append(j)

            # arrow
            if plot_arrows:
                dr = y - x
                top = x + .6667 * dr
                dr /= sqrt(np.sum(dr * dr))
                bottom = top - arrow_size * dr
                n = np.array([-dr[1], dr[0]])
                left, right = bottom + half_base * n, bottom - half_base * n
                h = plt.plot([left[0], top[0], right[0]], [left[1], top[1], right[1]], arrow_color+'-')
                assert not h[1:]
                arrows.append(h[0])

    # cell centers
    plot_vertices = vertex_color and vertex_style
    if plot_vertices:
        h = plt.plot(centers[:,0], centers[:,1], vertex_color+vertex_style)
        assert not h[1:]
        vertices = h[0]

    return edges, vertices, arrows


def plot_cell_indices(cells, font_size=12, shift_indices=False, **kwargs):
    """
    Plot cell indices at the cell centers.

    Arguments:

        cells (Partition or Tessellation or FiniteElements):
            tessellation

        font_size (int):
            alias for `fontsize`

        shift_indices (bool):
            first cell is numbered 1 instead of 0

    Returns:

        list: handles of the individual text elements

    Trailing keyword arguments are passed to :func:`~matplotlib.pyplot.text`.
    """
    import matplotlib.pyplot as plt
    kwargs['fontsize'] = kwargs.get('fontsize', font_size)
    handles = []
    if isinstance(cells, Partition):
        cells = cells.tessellation
    # common plotting logic
    def text(x, y, i):
        if np.isnan(x) or np.isnan(y):
            import warnings
            warnings.warn('nan coordinate', RuntimeWarning)
            return
        elif np.isinf(x) or np.isinf(y):
            import warnings
            warnings.warn('inf coordinate', RuntimeWarning)
            return
        h = plt.text(x, y, str(i+1 if shift_indices else i), **kwargs)
        handles.append(h)
    # Partition and Tessellation
    if isinstance(cells, Tessellation):
        i = 0
        for x,y in cells.cell_centers:
            text(x, y, i)
            i += 1
    # FiniteElements
    else:#if isinstance(cells, FiniteElements):
        for i in cells:
            x,y = cells[i].center
            text(x, y, i)
    return handles


def plot_indices(*args, **kwargs):
    """
    Alias for :func:`plot_cell_indices`.

    *plot_cell_indices* was formerly named *plot_indices*.
    """
    return plot_cell_indices(*args, **kwargs)


__all__ = ['__colors__', 'plot_points', 'plot_voronoi', 'plot_delaunay', 'plot_distributed', 'plot_indices', 'plot_cell_indices']

