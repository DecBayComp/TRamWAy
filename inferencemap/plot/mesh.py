
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from ..mesh.tesselation import *
import itertools


def plot_points(points, stats=None, tess=None, min_count=None, style='.', size=8, color=None):
	if tess:
		if tess.cell_label:
			label = tess.cell_label[stats.cell_to_point]
		else:
			if stats is None:
				stats = tess.cellStats(points)
			label = stats.ce
	elif min_count:
		cell_mask = min_count <= stats.cell_count
		label = cell_mask[stats.cell_to_point]
	elif stats:
		label = stats.cell_index
	else:	label = None

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
				color = 'bgrcmyk'
				color = list(itertools.islice(itertools.cycle(color), len(L)))
			elif len(L) == 2:
				color = ['gray', 'k']
			else:	color = 'k'
		for i, l in enumerate(L):
			plt.plot(x[label == l], y[label == l], 
				style, color=color[i], markersize=size)


def plot_voronoi(tess, stats, color=None, style='-', centroid_style='r+'):
	vertices = np.asarray(tess.cell_vertices)
	if not color:
		if tess.adjacency_label is None:
			color = 'r'
		else:
			color = 'wgyr'
	# plot voronoi
	for edge_ix, vert_ids in enumerate(tess.ridge_vertices):
		if all(0 <= vert_ids):
			x, y = zip(*vertices[vert_ids])
			if tess.adjacency_label is None:
				c = 0
			else:
				c = int(tess.adjacency_label[edge_ix])
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


