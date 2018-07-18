# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
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
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
import scipy.sparse as sparse
from scipy.spatial.distance import cdist
from ..core import *
from ..tessellation.base import dict_to_sparse, format_cell_index, nearest_cell
import traceback
from collections import defaultdict


def plot_points(cells, min_count=None, style='.', size=8, color=None, tess=None, **kwargs):
        if isinstance(cells, np.ndarray):
                points = cells
                label = None
        else:
                points = cells.descriptors(cells.points, asarray=True)
                label = cells.cell_index
                npts = points.shape[0]
                ncells = cells.location_count.size
                # if label is not a single index vector, convert it following 
                # tessellation.base.Delaunay.cell_index with `preferred`='force index'.
                merge = nearest_cell(points, cells.tessellation.cell_centers)
                label = format_cell_index(cells.cell_index, format='array', \
                        select=merge, shape=(npts, ncells))
                if min_count and ('knn' not in cells.param or min_count < cells.param['knn']):
                        cell_mask = min_count <= cells.location_count
                        label[np.logical_not(cell_mask[cells.cell_index])] = -1
                        #label = cell_mask[cells.cell_index]


        if isstructured(points):
                x = points['x']
                y = points['y']
        else:
                x = points[:,0]
                y = points[:,1]

        if label is None:
                if color is None:
                        color = 'k'
                elif isinstance(color, (pd.Series, pd.DataFrame)):
                        color = np.asarray(color)
                if isinstance(color, np.ndarray):
                        cmin, cmax = np.min(color), np.max(color)
                        color = (color - cmin) / (cmax - cmin)
                        cmap = plt.get_cmap()
                        color = [ cmap(c) for c in color ]
                plt.scatter(x, y, color=color, marker=style, s=size, **kwargs)
        else:
                L = np.unique(label)
                if color in [None, 'light']:
                        if color == 'light' and 'alpha' not in kwargs:
                                kwargs['alpha'] = .2
                        if 2 < len(L):
                                color = ['darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkviolet', 'deeppink', 'deepskyblue', 'dodgerblue', 'firebrick', 'forestgreen', 'gold', 'goldenrod', 'hotpink', 'indianred', 'indigo', 'lightblue', 'lightcoral', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightsteelblue', 'limegreen', 'maroon', 'mediumaquamarine', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navajowhite', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', '#663399', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'skyblue', 'slateblue', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'yellowgreen']
                                color = ['gray'] + \
                                        list(itertools.islice(itertools.cycle(color), len(L)))
                        elif len(L) == 2:
                                color = ['gray', 'k']
                        else:   color = 'k'
                for i, l in enumerate(L):
                        plt.plot(x[label == l], y[label == l], 
                                style, color=color[i], markersize=size, **kwargs)

        # resize window
        try:
                plt.axis(cells.bounding_box[['x', 'y']].values.flatten('F'))
        except AttributeError:
                pass
        except ValueError:
                print(traceback.format_exc())


def plot_voronoi(cells, labels=None, color=None, style='-', centroid_style='g+', negative=None,
                linewidth=1):
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
                                print(traceback.format_exc())
                                print("vertices {} and {} do not match with a ridge".format(u, v))
                                continue
                if cells.tessellation.adjacency_label is not None:
                        try:
                                c = labels.index(cells.tessellation.adjacency_label[edge_ix])
                        except ValueError:
                                continue
                plt.plot(x, y, style, color=color[c], linewidth=linewidth)

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
                        plt.plot(x_, y_, 'c-')
                        x_, y_ = (i + j) / 2
                        plt.text(x_, y_, str(edge_ix), \
                                horizontalalignment='center', verticalalignment='center')

        centroids = cells.tessellation.cell_centers
        # plot cell centers
        if centroid_style:
                plt.plot(centroids[:,0], centroids[:,1], centroid_style)

        # resize window
        try:
                plt.axis(cells.bounding_box[['x', 'y']].values.flatten('F'))
        except AttributeError:
                pass
        except ValueError:
                print(traceback.format_exc())


def plot_delaunay(cells, labels=None, color=None, style='-', centroid_style='g+', negative=None,
                axes=None, linewidth=1, individual=False):
        if axes is None:
                axes = plt
        try:
                tessellation = cells.tessellation
        except AttributeError:
                tessellation = cells

        vertices = tessellation.cell_centers
        if negative is 'voronoi':
                voronoi = tessellation.cell_vertices

        labels, color = _graph_theme(tessellation, labels, color, negative)

        # if asymetric, can be either triu or tril
        A = sparse.triu(tessellation.cell_adjacency, format='coo')
        I, J, K = A.row, A.col, A.data
        if not I.size:
                A = sparse.tril(tessellation.cell_adjacency, format='coo')
                I, J, K = A.row, A.col, A.data

        # plot delaunay
        if not individual:
                by_color = defaultdict(list)
        obj = []
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
                                if negative is 'voronoi':
                                        try:
                                                vert_ids = set(tessellation.cell_vertices.get(i, [])) & set(tessellation.cell_vertices.get(j, []))
                                                x, y = voronoi[vert_ids].T
                                        except ValueError:
                                                continue
                if individual:
                        obj.append(axes.plot(x, y, style, color=color[c], linewidth=linewidth))
                else:
                        by_color[c].append((x, y))

        if not individual:
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
                        obj.append(axes.plot(X, Y, style,
                                color=color[c if color[1:] else 0],
                                linewidth=linewidth))

        # plot cell centers
        if centroid_style:
                obj.append(axes.plot(vertices[:,0], vertices[:,1], centroid_style))

        # resize window
        try:
                axes.axis(cells.bounding_box[['x', 'y']].values.flatten('F'))
        except AttributeError:
                pass
        except ValueError:
                print(traceback.format_exc())

        if obj:
                return list(itertools.chain(*obj))
        else:
                return []


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
                neg_color = 'cymw'
                pos_color = 'rgbk'
                labels.sort()
                color = neg_color[:nnp] + pos_color
        return (labels, color)


def plot_distributed(cells, vertex_color='g', vertex_style='x', edge_color='r',
                arrow_size=5, arrow_color='y', font_size=12, shift_indices=False):
        centers = np.vstack([ cells[i].center for i in cells ])
        _min, _max = np.min(centers), np.max(centers)
        arrow_size = float(arrow_size) * 1e-3 * (_max - _min)
        half_base = arrow_size / sqrt(5.)
        plotted = defaultdict(list)
        for i in cells:
                x = cells[i].center
                for j in cells.adjacency[i].indices:
                        y = cells[j].center
                        if j not in plotted[i]:
                                plt.plot([x[0], y[0]], [x[1], y[1]], edge_color+'-')
                                plotted[i].append(j)
                        dr = y - x
                        top = x + .6667 * dr
                        dr /= sqrt(np.sum(dr * dr))
                        bottom = top - arrow_size * dr
                        n = np.array([-dr[1], dr[0]])
                        left, right = bottom + half_base * n, bottom - half_base * n
                        plt.plot([left[0], top[0], right[0]], [left[1], top[1], right[1]], arrow_color+'-')
                plt.text(x[0], x[1], str(i+1 if shift_indices else i), fontsize=font_size)
        plt.plot(centers[:,0], centers[:,1], vertex_color+vertex_style)


def plot_indices(cells, **kwargs):
        try:
                cells = cells.tessellation
        except (KeyboardInterrupt, SystemExit):
                raise
        except:
                pass
        i = 0
        for x,y in cells.cell_centers:
                plt.text(x, y, str(i), **kwargs)
                i += 1

