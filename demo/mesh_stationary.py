#python -m inferencemap.demo.mesh_stationary

import numpy as np
import pandas as pd
import numpy.linalg as la
import matplotlib.pyplot as plt
from ..tesselation import *
from ..plot.mesh import *
from ..spatial.scaler import *
from ..motion.jumps import *

#################### Script parameters #########################

xyt_file = 'glycine_receptor.trxyt'

subset_size = None # glycine_receptor is actually not so big...
n_pts_per_cell = 80.0 # from InferenceMAP v1, useful for k-means
min_n_pts_per_cell = 10.0 # should try 20.0 instead
#method_name = 'kmeans'
method_name = 'gwr'

################################################################

# load the data
df = pd.read_table(xyt_file, names=['n', 'x', 'y', 't'])

# distance between adjacent cell centers
if xyt_file == 'glycine_receptor.trxyt':
	avg_distance = 0.18431692306401384
else:
	avg_distance = np.nanmean(la.norm(np.asarray(jumps(df)[['x','y']])), axis=1) # slow
min_distance = 0.8 * avg_distance
max_distance = 2.0 * avg_distance

method = dict(kmeans=KMeansMesh, gwr=GasMesh)

if subset_size:
	# take a subset of points instead
	subset = np.random.permutation(df.shape[0])
	df = df.iloc[subset[0:subset_size],]
n_pts = df.shape[0]

# initialize a Tesselation object
tess = method[method_name](whiten, min_distance=min_distance, max_distance=max_distance, \
	min_probability=min_n_pts_per_cell / n_pts, avg_probability=n_pts_per_cell / n_pts)
# grow the tesselation
tess.tesselate(df[['x', 'y']])
# partition the dataset into the cells of the tesselation
stats = tess.cellStats(df[['x', 'y']])

# plot the data points together with the tesselation
plot_points(df[['x', 'y']], stats, min_count=min_n_pts_per_cell)
plot_voronoi(tess, stats)
plt.title(method_name + '-based voronoi')
plt.show()

# plot a histogram of the number of points per cell
plt.hist(stats.cell_count, range=(0,600), bins=20)
plt.title(method_name + '-based tesselation')
plt.xlabel('cell count')
plt.show()

# plot a histogram of the distance between adjacent cell centers
A = tess.cell_adjacency.tocoo()
i, j, k = A.row, A.col, A.data
if tess.adjacency_label is not None:
	i = i[tess.adjacency_label[k] == 3]
	j = j[tess.adjacency_label[k] == 3]
pts = np.asarray(tess.cell_centers)
dist = la.norm(pts[i,:] - pts[j,:], axis=1)
dmin = np.log(min_distance)
dmax = np.log(max_distance)
plt.hist(np.log(dist), bins=20)
plt.plot((dmin, dmin), plt.ylim(), 'r-')
plt.plot((dmax, dmax), plt.ylim(), 'r-')
plt.title(method_name + '-based tesselation')
plt.xlabel('inter-centroid distance (log)')
plt.show()

