
import os
import numpy as np
import pandas as pd
import warnings
import scipy.sparse as sparse

class IOWarning(Warning):
	pass
class FileNotFoundWarning(IOWarning):
	pass

def load_xyt(path, columns=['n', 'x', 'y', 't'], concat=True):
	if os.path.isdir(path):
		path = [ os.path.join(path, f) for f in os.listdir(path) ]
	else:
		path = [path]
	index_max = 0
	df = []
	for f in path:
		try:
			dff = pd.read_table(f, names=columns)
			if dff['n'].min() < index_max:
				dff['n'] += index_max
				index_max = dff['n'].max()
			df.append(dff)
		except:
			warnings.warn(f, FileNotFoundWarning)
	if concat:
		df = pd.concat(df)
	return df

def save_delaunay(path, points, adjacency, cell_labels=None, edge_labels=None, vertices=None, ridge_vertices=None):
	args = dict(points=points, adjacency=adjacency)
	if cell_labels:
		args['cell_labels'] = cell_labels
	if edge_labels:
		args['edge_labels'] = edge_labels
	if vertices:
		args['vertices'] = vertices
	if ridge_vertices:
		args['ridge_vertices'] = ridge_vertices
	np.savez(path, **args)

def load_delaunay(path):
	return np.load(path)

