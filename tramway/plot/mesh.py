
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
import scipy.sparse as sparse
from scipy.spatial.distance import cdist
from ..core import *


def plot_points(cells, min_count=None, style='.', size=8, color=None, tess=None):
	if isinstance(cells, np.ndarray):
		points = cells
		label = None
	else:
		points = cells.descriptors(cells.points, asarray=True)
		label = cells.cell_index
		npts = points.shape[0]
		ncells = cells.cell_count.size
		# if label is not a single index vector, convert it following 
		# tesselation.base.Delaunay.cellIndex with `prefered`='force index'.
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
				if cells.tesselation is None:
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
					cell_centers = cells.tesselation.cell_centers
				spmat = label
				label = -np.ones(npts, dtype=int)
				label[allright_points] = allright_cells
				for p in affected_points:
					cells = spmat[p].indices
					D = cell_centers[cells] - points[p]#[np.newaxis,:] # broadcast seems to also work without newaxis
					label[p] = cells[np.argmin(np.sum(D * D, axis=1))]
		#
		if min_count and ('knn' not in cells.param or min_count < cells.param['knn']):
			cell_mask = min_count <= cells.cell_count
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
		if color is None:
			if 2 < len(L):
				color = ['darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkviolet', 'deeppink', 'deepskyblue', 'dodgerblue', 'firebrick', 'forestgreen', 'gold', 'goldenrod', 'hotpink', 'indianred', 'indigo', 'lightblue', 'lightcoral', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightsteelblue', 'limegreen', 'maroon', 'mediumaquamarine', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navajowhite', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', '#663399', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'skyblue', 'slateblue', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'yellowgreen']
				color = ['gray'] + \
					list(itertools.islice(itertools.cycle(color), len(L)))
			elif len(L) == 2:
				color = ['gray', 'k']
			else:	color = 'k'
		for i, l in enumerate(L):
			plt.plot(x[label == l], y[label == l], 
				style, color=color[i], markersize=size)


def plot_voronoi(cells, labels=None, color=None, style='-', centroid_style='g+', negative=None):
	vertices = cells.tesselation.cell_vertices
	labels, color = _graph_theme(cells.tesselation, labels, color, negative)
	try:
		special_edges = cells.tesselation.candidate_edges
		#points = cells.descriptors(cells.points, asarray=True)
	except:
		special_edges = {}
	c = 0 # if cells.tesselation.adjacency_label is None
	# plot voronoi
	for edge_ix, vert_ids in enumerate(cells.tesselation.ridge_vertices):
		if all(0 <= vert_ids):
			x, y = vertices[vert_ids].T
			if cells.tesselation.adjacency_label is not None:
				try:
					c = labels.index(cells.tesselation.adjacency_label[edge_ix])
				except ValueError:
					continue
			plt.plot(x, y, style, color=color[c], linewidth=1)

			# extra debug steps
			if edge_ix in special_edges:
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

	centroids = cells.tesselation.cell_centers
	# plot cell centers
	if centroid_style:
		plt.plot(centroids[:,0], centroids[:,1], centroid_style)

	# resize window
	plt.axis(cells.descriptors(cells.bounding_box, asarray=True).flatten('F'))


def plot_delaunay(cells, labels=None, color=None, style='-', centroid_style='g+', negative=None):
	vertices = cells.tesselation.cell_centers
	if negative is 'voronoi':
		voronoi = cells.tesselation.cell_vertices

	labels, color = _graph_theme(cells.tesselation, labels, color, negative)

	# if asymetric, can be either triu or tril
	A = sparse.triu(cells.tesselation.cell_adjacency, format='coo')
	I, J, K = A.row, A.col, A.data
	if not I.size:
		A = sparse.tril(cells.tesselation.cell_adjacency, format='coo')
		I, J, K = A.row, A.col, A.data

	# plot delaunay
	for i, j, k in zip(I, J, K):
		x, y = zip(vertices[i], vertices[j])
		if labels is None:
			c = 0
		else:
			label = cells.tesselation.adjacency_label[k]
			try:
				c = labels.index(label)
			except ValueError:
				continue
			if label <= 0:
				if negative is 'voronoi':
					vert_ids = cells.tesselation.ridge_vertices[k]
					if any(vert_ids < 0):
						continue
					x, y = voronoi[vert_ids].T
		plt.plot(x, y, style, color=color[c], linewidth=1)

	# plot cell centers
	if centroid_style:
		plt.plot(vertices[:,0], vertices[:,1], centroid_style)

	# resize window
	plt.axis(cells.descriptors(cells.bounding_box, asarray=True).flatten('F'))


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


