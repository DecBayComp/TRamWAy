# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
import scipy.sparse as sparse
from scipy.spatial.distance import cdist
from ..core import *
from ..tessellation import dict_to_sparse
import traceback


def plot_points(cells, min_count=None, style='.', size=8, color=None, tess=None):
	if isinstance(cells, np.ndarray):
		points = cells
		label = None
	else:
		points = cells.descriptors(cells.points, asarray=True)
		label = cells.cell_index
		npts = points.shape[0]
		ncells = cells.location_count.size
		# if label is not a single index vector, convert it following 
		# tessellation.base.Delaunay.cellIndex with `prefered`='force index'.
		if isinstance(label, tuple):
			label = sparse.csr_matrix((np.ones_like(label[0], dtype=bool), \
				label), shape=(npts, ncells))
		if sparse.issparse(label):
			cell_count_per_point = np.diff(label.tocsr().indptr)
			if all(cell_count_per_point < 2): # no overlap
				I, J = label.nonzero()
				label = -np.ones(npts, dtype=int)
				label[I] = J
			else: # in the following, "affected" refers to overlap
				affected_points = 1 < cell_count_per_point
				affected_cells = label[affected_points].indices
				#_, affected_cells = label[affected_points,:].nonzero()
				affected_cells = np.unique(affected_cells) # unique indices
				map_affected_cells = np.zeros(ncells, dtype=affected_cells.dtype)
				map_affected_cells[affected_cells] = affected_cells
				affected_points, = affected_points.nonzero() # indices
				allright_points = 1 == cell_count_per_point # bool
				allright_cells = label[allright_points].indices # indices
				#_, allright_cells = label[allright_points,:].nonzero()
				if cells.tessellation is None:
					# to compute again the point-center distance matrix first estimate
					# cell centers as the centers of gravity of the associated points
					cell_centers = np.zeros((ncells, points.shape[1]),
						dtype=points.dtype)
					label = label.tocsc()
					for i in affected_cells:
						j = label.indices[label.indptr[i]:label.indptr[i+1]]
						if j.size:
							jj = affected_points[j]
							cell_centers[i] = np.mean(points[jj], axis=0)
				else:
					cell_centers = cells.tessellation.cell_centers
				spmat = label
				label = -np.ones(npts, dtype=int)
				label[allright_points] = allright_cells
				for p in affected_points:
					cells = spmat[p].indices
					D = cell_centers[cells] - points[p]#[np.newaxis,:] # broadcast seems to also work without newaxis
					label[p] = cells[np.argmin(np.sum(D * D, axis=1))]
		#
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
		plt.plot(x, y, style, color=color, markersize=size)
	else:
		L = np.unique(label)
		kwargs = {}
		if color in [None, 'light']:
			if color == 'light':
				kwargs = {'alpha': .2}
			if 2 < len(L):
				color = ['darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkviolet', 'deeppink', 'deepskyblue', 'dodgerblue', 'firebrick', 'forestgreen', 'gold', 'goldenrod', 'hotpink', 'indianred', 'indigo', 'lightblue', 'lightcoral', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightsteelblue', 'limegreen', 'maroon', 'mediumaquamarine', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navajowhite', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', '#663399', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'skyblue', 'slateblue', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'yellowgreen']
				color = ['gray'] + \
					list(itertools.islice(itertools.cycle(color), len(L)))
			elif len(L) == 2:
				color = ['gray', 'k']
			else:	color = 'k'
		for i, l in enumerate(L):
			plt.plot(x[label == l], y[label == l], 
				style, color=color[i], markersize=size, **kwargs)

	# resize window
	try:
		plt.axis(cells.descriptors(cells.bounding_box, asarray=True).flatten('F'))
	except AttributeError:
		pass
	except ValueError:
		print(traceback.format_exc())


def plot_voronoi(cells, labels=None, color=None, style='-', centroid_style='g+', negative=None,
		linewidth=1):
	vertices = cells.tessellation.vertices
	labels, color = _graph_theme(cells.tessellation, labels, color, negative)
	color += 'w'
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
			#	i = points[cells.cell_index == i][ii]
			#	j = points[cells.cell_index == j][jj]
			#except IndexError as e:
			#	print(e)
			#	continue
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
		plt.axis(cells.descriptors(cells.bounding_box, asarray=True).flatten('F'))
	except AttributeError:
		pass
	except ValueError:
		print(traceback.format_exc())


def plot_delaunay(cells, labels=None, color=None, style='-', centroid_style='g+', negative=None,
		axes=None, linewidth=1):
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
		obj.append(axes.plot(x, y, style, color=color[c], linewidth=linewidth))

	# plot cell centers
	if centroid_style:
		obj.append(axes.plot(vertices[:,0], vertices[:,1], centroid_style))

	# resize window
	try:
		axes.axis(cells.descriptors(cells.bounding_box, asarray=True).flatten('F'))
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


