#!/usr/bin/env python

import argparse
import sys
import os
import numpy as np
import pandas as pd
import numpy.linalg as la
import matplotlib.pyplot as plt
from .tesselation import *
from .plot.mesh import *
from .spatial.scaler import *
from .motion.jumps import *
from .io import *


hdf_extensions = ['.h5', '.hdf', '.hdf5']
imt_extensions = [ '.imt' + ext for ext in hdf_extensions ]
fig_formats = ['png', 'pdf', 'ps', 'eps', 'svg']


def render(args):
	if args.input is None:
		args.input = ['./']
	if args.input[1:]:
		print('ignoring the extra input files')
	imt_path = args.input[0]
	if os.path.isdir(imt_path):
		imt_path = os.listdir(imt_path)
		files, exts = zip(*os.path.splitext(imt_path))
		for ext in imt_extensions:
			if ext in exts:
				imt_path = imt_path[exts.index(ext)]
				break
		if isinstance(imt_path, list):
			imt_path = imt_path[0]
		auto_select = True
	elif os.path.isfile(imt_path):
		auto_select = False
	else:
		candidates = [ imt_path + ext for ext in imt_extensions ]
		candidates = [ f for f in candidates if os.path.isfile(f) ]
		if candidates:
			imt_path = candidates[0]
		else:
			raise IOError('no tesselation file found in {}'.format(imt_path))
		auto_select = True
	if auto_select and args.verbose:
		print('selecting {} as a tesselation file'.format(imt_path))

	# load the data
	hdf = HDF5Store(args.input[0], 'r')
	stats = hdf.peek('points')
	tess = hdf.peek('mesh')
	# guess back some input parameters
	method_name = {RegularMesh: ('grid', 'regular grid'), \
		KMeansMesh: ('kmeans', 'k-means-based tesselation'), \
		GasMesh: ('gwr', 'gwr-based tesselation')}
	method_name, method_title = method_name[type(tess)]
	min_distance = stats.param.get('min_distance', 0)
	max_distance = stats.param.get('max_distance', None)
	min_cell_count = stats.param.get('min_cell_count', 0)

	# plot the data points together with the tesselation
	plot_points(stats.coordinates, stats, min_count=min_cell_count)
	plot_voronoi(tess, stats)
	plt.title(method_name + '-based voronoi')
	if args.output or args.print:
		filename, figext = io.path.splitext(args.output)
		if figext and figext in fig_formats:
			pass
		elif args.print:
			figext = args.print
		else:
			figext = 'svg'
		if not filename:
			filename, _ = io.path.splitext(args.input[0])
			filename, _ = io.path.splitext(filename)
		plt.savefig(filename + 'vor' + figext)

	if args.histogram and 'c' in args.histogram:
		# plot a histogram of the number of points per cell
		plt.hist(stats.cell_count, range=(0,600), bins=20)
		plt.title(method_title)
		plt.xlabel('cell count')
		if args.output or args.print:
			plt.savefig(filename + 'cnt' + figext)

	if args.histogram and 'd' in args.histogram:
		# plot a histogram of the distance between adjacent cell centers
		A = tess.cell_adjacency.tocoo()
		i, j, k = A.row, A.col, A.data
		if tess.adjacency_label is not None:
			i = i[tess.adjacency_label[k] == 3]
			j = j[tess.adjacency_label[k] == 3]
		pts = np.asarray(tess.cell_centers)
		dist = la.norm(pts[i,:] - pts[j,:], axis=1)
		plt.hist(np.log(dist), bins=20)
		if max_distance:
			dmin = np.log(min_distance)
			dmax = np.log(max_distance)
			plt.plot((dmin, dmin), plt.ylim(), 'r-')
			plt.plot((dmax, dmax), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('inter-centroid distance (log)')
		if args.output or args.print:
			plt.savefig(filename + 'icd' + figext)

	if args.histogram and 'p' in args.histogram:
		adj = point_adjacency_matrix(tess, stats, symetric=False)
		dist = adj.data
		plt.hist(np.log(dist), bins=50)
		if max_distance:
			dmin = np.log(min_distance)
			dmax = np.log(max_distance)
			plt.plot((dmin, dmin), plt.ylim(), 'r-')
			plt.plot((dmax, dmax), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('inter-point distance (log)')
		if args.output or args.print:
			plt.savefig(filename + 'pwd' + figext)

	if not args.print:
		plt.show()




def tesselate(args):
	if args.input:
		xyt_path = args.input
	else:
		xyt_path = './'
	df, xyt_path = load_xyt(xyt_path, return_paths=True, verbose=args.verbose)

	if args.method:
		if args.distance is None:
			b = np.asarray(jumps(df)[['x','y']]) # slow
			avg_distance = np.nanmean(np.sqrt(np.sum(b * b, axis=1)))
			if args.verbose:
				print('average jump distance: {}'.format(avg_distance))
		else:
			avg_distance = args.distance
		min_distance = 0.8 * avg_distance
		max_distance = 2.0 * avg_distance

		methods = dict(grid=RegularMesh, kmeans=KMeansMesh, gwr=GasMesh)
		method = methods[args.method]
		if args.w:
			args.scaling = 'whiten'
		elif args.scaling is None:
			args.scaling = 'none'
		scalers = dict(none=Scaler(), whiten=whiten, unit=unitrange)
		scaler = scalers[args.scaling]
		n_pts = df.shape[0]

		# initialize a Tesselation object
		tess = method(scaler, min_distance=min_distance, max_distance=max_distance, \
			min_probability=args.cell_count / n_pts, \
			avg_probability=args.min_cell_count / n_pts, \
			verbose=args.verbose)

		# grow the tesselation
		tess.tesselate(df[['x', 'y']])

	else:
		raise NotImplementedError

	# partition the dataset into the cells of the tesselation
	stats = tess.cellStats(df[['x', 'y']])
	stats.param['min_distance'] = min_distance
	stats.param['avg_distance'] = avg_distance
	stats.param['max_distance'] = max_distance
	stats.param['min_cell_count'] = args.min_cell_count
	stats.param['avg_cell_count'] = args.cell_count
	stats.param['method'] = args.method

	# save `stats` and `tess`
	if args.output is None:
		xyt_file, _ = os.path.splitext(xyt_path[0])
		imt_path = xyt_file + imt_extensions[0]
	else:
		imt_path, imt_ext = os.path.splitext(args.output)
		if imt_ext in hdf_extensions:
			imt_path = args.output
		else:
			imt_path += imt_extensions[0]

	store = HDF5Store(imt_path, 'w')
	store.poke('points', stats)
	store.poke('mesh', tess)
	store.close()
	if args.verbose:
		print('{} written'.format(imt_path))

	sys.exit(0)




if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog='inferencemap', \
		description='InferenceMAP central command.', \
		epilog='See also https://github.com/influencecell/inferencemap')
	parser.add_argument('-v', '--verbose', action='store_true', help='increase verbosity')
	parser.add_argument('-i', '--input', action='append', \
		help='path to input file or directory')
	parser.add_argument('-o', '--output', help='path to output file')
	sub = parser.add_subparsers(title='commands', \
		description="type '%(prog)s command --help' for additional help")

	# render
	render_parser = sub.add_parser('render')
	render_parser.set_defaults(func=render)
	render_parser.add_argument('-s', '--min-cell-count', type=int, default=20, \
		help='minimum number of points per cell')
	render_parser.add_argument('-H', '--histogram', help="plot/print additional histogram(s); any combination of 'c' (cell count histogram), 'd' (distance between neighboring centers) and 'p' (distance between any pair of points from distinct neighboring centers)")
	render_parser.add_argument('-p', '--print', choices=fig_formats, help='print figure(s) on disk instead of plotting')

	# tesselate
	tesselate_parser = sub.add_parser('tesselate')
	tesselate_parser.set_defaults(func=tesselate)
	tesselate_group1 = tesselate_parser.add_mutually_exclusive_group(required=True)
	tesselate_group1.add_argument('-m', '--method', choices=['grid', 'qtree', 'kmeans', 'gwr'])
	tesselate_group1.add_argument('-r', '--reuse', \
		#nargs='?', type=argparse.FileType('r'), default=sys.stdin, \
		help='apply precomputed tesselation from file')
	tesselate_parser.add_argument('-n', '--knn', type=int, \
		help='maximum number of nearest neighbors')
	tesselate_parser.add_argument('-d', '--distance', type=float, help='average jump distance')
	tesselate_group2 = tesselate_parser.add_mutually_exclusive_group()
	tesselate_group2.add_argument('-w', action='store_true', help='whiten the input data')
	tesselate_group2.add_argument('--scaling', choices=['whiten', 'unit'])
	tesselate_parser.add_argument('-s', '--min-cell-count', type=int, default=20, \
		help='minimum number of points per cell')
	grid_parser = tesselate_parser
	grid_parser.add_argument('-S', '--cell-count', type=int, default=80, \
		help='average number of points per cell [grid|kmeans]')
	#gwr_parser = tesselate_parser
	#gwr_parser.add_argument('-S', '--cell-count', type=int, default=80, \
	#	help='average number of points per cell [gwr]')
	
	# parse
	args = parser.parse_args()
	args.func(args)

