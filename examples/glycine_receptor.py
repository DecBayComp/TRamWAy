
import os
#from pathlib import Path
#import urllib.request # only in PY3
from inferencemap.helper.tesselation import *

#demo_name = 'glycine-receptor'

short_description = 'various tesselations of glycine receptor data'

#data_server = 'http://157.99.149.144/'
data_server = 'http://dl.pasteur.fr/fop/jod5vvB5/'
data_file = 'glycine_receptor.trxyt'
data_dir = ''

if __name__ == '__main__':
	main()

def main():
	local = os.path.join(data_dir, data_file)

	## download data file if missing
	#if not os.path.exists(local):
	#	urllib.request.urlretrieve(data_server + data_file, local)

	output_basename, _ = os.path.splitext(local)
	def out(method, extension):
		return '{}.{}{}'.format(output_basename, method, extension)

	# grid + knn
	print("\nfirst approach: regular grid + (knn <= 40)")
	method = 'grid'
	grid = tesselate(local, method, output_file=out(method, '.h5'), knn=(None, 40), verbose=True)
	print('overlaying Voronoi graph')
	cell_plot(grid, output_file=out(method, '.png'), verbose=True)

	# kd-tree
	print("\nsecond approach: kd-tree")
	method = 'kdtree'
	kdtree = tesselate(local, method, output_file=out(method, '.h5'), verbose=True)
	print('overlaying Voronoi graph')
	cell_plot(kdtree, output_file=out(method, '.png'), verbose=True)

	# k-means
	print("\nthird approach: k-means")
	method = 'kmeans'
	kmeans = tesselate(local, method, output_file=out(method, '.h5'), verbose=True)
	print('overlaying Delaunay graph')
	cell_plot(kmeans, output_file=out(method, '.png'), verbose=True, xy_layer='delaunay')

	# gwr
	print("\nfourth approach: GWR + (40 <= knn <= 60)")
	method = 'gwr'
	gwr = tesselate(local, method, output_file=out(method, '.h5'), knn=(40, 60), overlap=True, \
		verbose=True, pass_count=1)
	# `pass_count` above is a gwr-specific and controls the convergence (both accuracy and 
	# calculation time) by defining bounds on the number of passes over the data.
	# Note that the data are sampled with replacement. As a consequence ``pass_count=1`` is not
	# enough for the algorithm to visit every point once.
	# For higher accuracy, set ``pass_count=3`` for example.
	print('overlaying Delaunay graph')
	cell_plot(gwr, output_file=out(method, '.png'), verbose=True, xy_layer='delaunay')


