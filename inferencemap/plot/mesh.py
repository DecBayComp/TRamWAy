
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
import scipy.sparse as sparse


def plot_points(stats, min_count=None, style='.', size=8, color=None):
	if isinstance(stats, np.ndarray):
		points = stats
		label = None
	else:
		label = stats.cell_index
		npts = stats.coordinates.shape[0]
		ncells = stats.cell_count.size
		# if label is not a single index vector, convert it following 
		# tesselation.base.Delaunay.cellIndex with `prefered`='force index'.
		if isinstance(label, tuple):
			label = sparse.csc_matrix((np.ones_like(label[0], dtype=bool), \
				label), shape=(npts, ncells))
		if sparse.issparse(label):
			I, J = label.nonzero()
			cell_count_per_point = np.diff(label.tocsr().indptr)
			if all(cell_count_per_point < 2): # (much) faster
				label = -np.ones(npts, dtype=int)
				label[I] = J
			else: # more generic
				raise NotImplementedError('will be fixed very soon; for display purposes, please tesselate without `--overlap` for now')
				# to compute again the point-center distance matrix...
				cell_centers = np.zeros((ncells, stats.coordinates.shape[1]), \
					dtype=stats.coordinates.dtype)
				# (to do:) ...first estimate cell centers as the centers of gravity of 
				# the associated points
				D = cdist(np.asarray(stats.coordinates), cell_centers)
				label = np.argmin(D, axis=1)
				label[cell_count_per_point == 0] = -1
		#
		if min_count and ('knn' not in stats.param or min_count < stats.param['knn']):
			cell_mask = min_count <= stats.cell_count
			label[np.logical_not(cell_mask[stats.cell_index])] = -1
			#label = cell_mask[stats.cell_index]

		points = stats.coordinates


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


def plot_voronoi(tess, stats, color=None, style='-', centroid_style='g+'):
	vertices = np.asarray(tess.cell_vertices)
	if not color:
		if tess.adjacency_label is None:
			color = 'r'
		else:
			neg_color = 'yg'
			pos_color = 'rb'
			labels = np.unique(tess.adjacency_label).tolist()
			neg_labels = [ l for l in labels if float(l) <= 0 ]
			color = neg_color[:len(neg_labels)] + pos_color
	# plot voronoi
	for edge_ix, vert_ids in enumerate(tess.ridge_vertices):
		if all(0 <= vert_ids):
			x, y = zip(*vertices[vert_ids])
			if tess.adjacency_label is None:
				c = 0
			else:
				c = labels.index(int(tess.adjacency_label[edge_ix]))
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


def plot_delaunay(tess, stats, color=None, style='-', centroid_style='g+', negative='voronoi'):
	vertices = np.asarray(tess.cell_centers)
	if not color:
		if tess.adjacency_label is None:
			color = 'r'
		else:
			neg_color = 'cymw'
			pos_color = 'rgbk'
			labels = np.unique(tess.adjacency_label).tolist()
			neg_labels = [ l for l in labels if float(l) <= 0 ]
			color = neg_color[:len(neg_labels)] + pos_color
	if negative is 'voronoi':
		voronoi = np.asarray(tess.cell_vertices)

	directed = tess.cell_adjacency.data[-1] + 1 == tess.cell_adjacency.data.size
	if directed:
		tess.cell_adjacency.data[0] += 1
		I, J = tess.cell_adjacency.nonzero()
		tess.cell_adjacency.data[0] -= 1
		ok = True
	else:
		symetric = True
		uncrawled = deepcopy(tess.cell_adjacency) # `copy` is not deep enough
		uncrawled.data[::] = True
		I, J = uncrawled.nonzero()

	# plot delaunay
	for k, edge_ix in enumerate(tess.cell_adjacency.data):
		if not directed:
			# each edge is supposed to be reported twice in `cell_adjacency`;
			# check whether it hasn't already been crawled
			ok = uncrawled.data[k]
			if ok:
				if uncrawled[J[k], I[k]]:
					uncrawled[J[k], I[k]] = False
				elif symetric:
					print('warning: adjacency matrix is not symetric')
					symetric = False
		if ok:
			x, y = zip(vertices[I[k]], vertices[J[k]])
			if tess.adjacency_label is None:
				c = 0
			else:
				c = labels.index(int(tess.adjacency_label[edge_ix]))
				if c < len(neg_labels):
					if negative is 'none':
						continue
					elif negative is 'voronoi':
						vert_ids = tess.ridge_vertices[edge_ix]
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


