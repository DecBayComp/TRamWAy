
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
import scipy.sparse as sparse
from scipy.spatial.distance import cdist


def plot_points(stats, min_count=None, style='.', size=8, color=None):
	if isinstance(stats, np.ndarray):
		points = stats
		label = None
	else:
		points = np.asarray(stats.coordinates)
		label = stats.cell_index
		npts = stats.coordinates.shape[0]
		ncells = stats.cell_count.size
		# if label is not a single index vector, convert it following 
		# tesselation.base.Delaunay.cellIndex with `prefered`='force index'.
		if isinstance(label, tuple):
			label = sparse.coo_matrix((np.ones_like(label[0], dtype=bool), \
				label), shape=(npts, ncells))
		if sparse.issparse(label):
			I, J = label.nonzero()
			cell_count_per_point = np.diff(label.tocsr().indptr)
			if all(cell_count_per_point < 2): # (much) faster
				label = -np.ones(npts, dtype=int)
				label[I] = J
			else: # more generic
				# to compute again the point-center distance matrix first estimate
				# cell centers as the centers of gravity of the associated points
				cell_centers = np.zeros((ncells, points.shape[1]), dtype=points.dtype)
				label = label.tocsc()
				for i in range(ncells):
					j = label.indices[label.indptr[i]:label.indptr[i+1]]
					if j.size:
						cell_centers[i] = np.mean(points[j], axis=0)
				D = cdist(points, cell_centers)
				label = np.argmin(D, axis=1)
				label[cell_count_per_point == 0] = -1
		#
		if min_count and ('knn' not in stats.param or min_count < stats.param['knn']):
			cell_mask = min_count <= stats.cell_count
			label[np.logical_not(cell_mask[stats.cell_index])] = -1
			#label = cell_mask[stats.cell_index]


	if isinstance(points, pd.DataFrame):
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
				color = ['gray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkviolet', 'deeppink', 'deepskyblue', 'dodgerblue', 'firebrick', 'forestgreen', 'gold', 'goldenrod', 'hotpink', 'indianred', 'indigo', 'lightblue', 'lightcoral', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightsteelblue', 'limegreen', 'maroon', 'mediumaquamarine', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navajowhite', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', '#663399', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'skyblue', 'slateblue', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'yellowgreen']
				color = list(itertools.islice(itertools.cycle(color), len(L)))
			elif len(L) == 2:
				color = ['gray', 'k']
			else:	color = 'k'
		for i, l in enumerate(L):
			plt.plot(x[label == l], y[label == l], 
				style, color=color[i], markersize=size)


def plot_voronoi(tess, stats, labels=None, color=None, style='-', centroid_style='g+', negative=None):
	vertices = np.asarray(tess.cell_vertices)
	labels, color = _graph_theme(tess, labels, color, negative)
	# plot voronoi
	for edge_ix, vert_ids in enumerate(tess.ridge_vertices):
		if all(0 <= vert_ids):
			x, y = zip(*vertices[vert_ids])
			if tess.adjacency_label is None:
				c = 0
			else:
				try:
					c = labels.index(tess.adjacency_label[edge_ix])
				except ValueError:
					continue
			plt.plot(x, y, style, color=color[c], linewidth=1)

	centroids = np.asarray(tess.cell_centers)
	# plot cell centers
	if centroid_style:
		plt.plot(centroids[:,0], centroids[:,1], centroid_style)

	# resize window
	if isinstance(stats.bounding_box, pd.DataFrame):
		plt.axis(np.asarray(stats.bounding_box.T).flatten())
	else:
		plt.axis(stats.bounding_box)


def plot_delaunay(tess, stats, labels=None, color=None, style='-', centroid_style='g+', negative=None):
	vertices = np.asarray(tess.cell_centers)
	if negative is 'voronoi':
		voronoi = np.asarray(tess.cell_vertices)

	labels, color = _graph_theme(tess, labels, color, negative)

	A = sparse.tril(tess.cell_adjacency, format='coo')
	I, J, K = A.row, A.col, A.data

	# plot delaunay
	for i, j, k in zip(I, J, K):
		x, y = zip(vertices[i], vertices[j])
		if labels is None:
			c = 0
		else:
			label = tess.adjacency_label[k]
			try:
				c = labels.index(label)
			except ValueError:
				continue
			if label <= 0:
				if negative is 'voronoi':
					vert_ids = tess.ridge_vertices[k]
					if any(vert_ids < 0):
						continue
					x, y = zip(*voronoi[vert_ids])
		plt.plot(x, y, style, color=color[c], linewidth=1)

	# plot cell centers
	if centroid_style:
		plt.plot(vertices[:,0], vertices[:,1], centroid_style)

	# resize window
	if isinstance(stats.bounding_box, pd.DataFrame):
		plt.axis(np.asarray(stats.bounding_box.T).flatten())
	else:
		plt.axis(stats.bounding_box)


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


