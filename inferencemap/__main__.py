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
sub_extensions = ['imt', 'vor', 'cnt', 'icd', 'pwd']


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
		KDTreeMesh: ('kdtree', 'kD-tree-based tesselation'), \
		KMeansMesh: ('kmeans', 'k-means-based tesselation'), \
		GasMesh: ('gwr', 'gwr-based tesselation')}
	method_name, method_title = method_name[type(tess)]
	min_distance = stats.param.get('min_distance', 0)
	avg_distance = stats.param.get('avg_distance', None)
	min_cell_count = stats.param.get('min_cell_count', 0)

	# plot the data points together with the tesselation
	fig0 = plt.figure()
	if 'knn' in stats.param: # if knn <= min_count, min_count is actually ignored
		plot_points(stats)
	else:
		plot_points(stats, min_count=min_cell_count)
	if args.delaunay:
		plot_delaunay(tess, stats)
	else:
		plot_voronoi(tess, stats)
	plt.title(method_name + '-based voronoi')

	if args.output or args.__dict__['print']:
		if args.output:
			filename, figext = os.path.splitext(args.output)
			if args.__dict__['print']:
				figext = args.__dict__['print']
			elif figext and figext[1:] in fig_formats:
				figext = figext[1:]
			else:
				figext = fig_formats[0]
		else:
			figext = args.__dict__['print']
			filename, _ = os.path.splitext(args.input[0])
		subname, subext = os.path.splitext(filename)
		if subext and subext[1:] in sub_extensions:
			filename = subname
		vor_file = filename + '.vor.' + figext
		if args.verbose:
			print('writing file: {}'.format(vor_file))
		fig0.savefig(vor_file)

	if args.histogram and 'c' in args.histogram:
		# plot a histogram of the number of points per cell
		fig1 = plt.figure()
		plt.hist(stats.cell_count, bins=np.arange(0, min_cell_count*20, min_cell_count))
		plt.plot((min_cell_count, min_cell_count), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('cell count')
		if args.output or args.__dict__['print']:
			cnt_file = filename + '.cnt.' + figext
			if args.verbose:
				print('writing file: {}'.format(cnt_file))
			fig1.savefig(cnt_file)

	if args.histogram and 'd' in args.histogram:
		# plot a histogram of the distance between adjacent cell centers
		A = tess.cell_adjacency.tocoo()
		i, j, k = A.row, A.col, A.data
		if tess.adjacency_label is not None:
			i = i[0 < tess.adjacency_label[k]]
			j = j[0 < tess.adjacency_label[k]]
		pts = np.asarray(tess.cell_centers)
		dist = la.norm(pts[i,:] - pts[j,:], axis=1)
		fig2 = plt.figure()
		plt.hist(np.log(dist), bins=50)
		if avg_distance:
			dmin = np.log(min_distance)
			dmax = np.log(avg_distance)
			plt.plot((dmin, dmin), plt.ylim(), 'r-')
			plt.plot((dmax, dmax), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('inter-centroid distance (log)')
		if args.output or args.__dict__['print']:
			icd_file = filename + '.icd.' + figext
			if args.verbose:
				print('writing file: {}'.format(icd_file))
			fig2.savefig(icd_file)

	if args.histogram and 'p' in args.histogram:
		adj = point_adjacency_matrix(tess, stats, symetric=False)
		dist = adj.data
		fig3 = plt.figure()
		plt.hist(np.log(dist), bins=100)
		if avg_distance:
			dmin = np.log(min_distance)
			dmax = np.log(avg_distance)
			plt.plot((dmin, dmin), plt.ylim(), 'r-')
			plt.plot((dmax, dmax), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('inter-point distance (log)')
		if args.output or args.__dict__['print']:
			pwd_file = filename + '.pwd.' + figext
			if args.verbose:
				print('writing file: {}'.format(pwd_file))
			fig3.savefig(pwd_file)

	if not args.__dict__['print']:
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
			jump_length = np.nanmean(np.sqrt(np.sum(b * b, axis=1)))
			if args.verbose:
				print('average jump distance: {}'.format(jump_length))
		else:
			jump_length = args.distance
		min_distance = 0.8 * jump_length
		avg_distance = 1.0 * jump_length
		#max_distance = 2.0 * jump_length

		methods = dict(grid=RegularMesh, kdtree=KDTreeMesh, kmeans=KMeansMesh, gwr=GasMesh)
		method = methods[args.method]
		if args.w:
			args.scaling = 'whiten'
		elif args.scaling is None:
			args.scaling = 'none'
		scalers = dict(none=Scaler, whiten=whiten, unit=unitrange)
		scaler = scalers[args.scaling]()
		n_pts = float(df.shape[0])
		if args.max_cell_count:
			max_probability = float(args.max_cell_count) / n_pts
		else:
			max_probability = None

		# initialize a Tesselation object
		tess = method(scaler, min_distance=min_distance, avg_distance=avg_distance, \
			#max_distance=max_distance, \
			min_probability=float(args.min_cell_count) / n_pts, \
			avg_probability=float(args.cell_count) / n_pts, \
			max_probability=max_probability, \
			#lower_levels=args.lower_levels, \ # in args
			**args.__dict__)

		# grow the tesselation
		tess.tesselate(df[['x', 'y']], verbose=args.verbose)

	else:
		raise NotImplementedError

	# partition the dataset into the cells of the tesselation
	if args.overlap:
		stats = tess.cellStats(df[['x', 'y']], knn=args.knn)
	else:
		stats = tess.cellStats(df[['x', 'y']], knn=args.knn, prefered='force index')
	stats.param['jump_length'] = jump_length
	stats.param['min_distance'] = min_distance
	stats.param['avg_distance'] = avg_distance
	#stats.param['max_distance'] = max_distance
	stats.param['avg_cell_count'] = args.cell_count
	args_ = ['min_cell_count', 'max_cell_count', 'method', 'knn']
	if args.method == 'kdtree':
		args_.append('lower_levels')
	for arg in args_:
		if args.__dict__[arg] is not None:
			stats.param[arg] = args.__dict__[arg]

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

	store = HDF5Store(imt_path, 'w', args.verbose)
	if args.verbose:
		print('writing file: {}'.format(imt_path))
	store.poke('points', stats)
	store.poke('mesh', tess)
	store.close()

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
	render_parser.add_argument('-D', '--delaunay', action='store_true', help='plot the Delaunay graph instead of the Voronoi')
	render_parser.add_argument('-H', '--histogram', help="plot/print additional histogram(s); any combination of 'c' (cell count histogram), 'd' (distance between neighboring centers) and 'p' (distance between any pair of points from distinct neighboring centers)")
	render_parser.add_argument('-p', '--print', choices=fig_formats, help='print figure(s) on disk instead of plotting')

	# tesselate
	tesselate_parser = sub.add_parser('tesselate')
	tesselate_parser.set_defaults(func=tesselate)
	tesselate_group1 = tesselate_parser.add_mutually_exclusive_group(required=True)
	tesselate_group1.add_argument('-m', '--method', choices=['grid', 'kdtree', 'kmeans', 'gwr'])
	tesselate_group1.add_argument('-r', '--reuse', \
		#nargs='?', type=argparse.FileType('r'), default=sys.stdin, \
		help='apply precomputed tesselation from file')
	tesselate_parser.add_argument('-n', '--knn', type=int, \
		help='maximum number of nearest neighbors')
	tesselate_parser.add_argument('--overlap', action='store_true', help='allow cells to overlap (useful with knn)')
	tesselate_parser.add_argument('-d', '--distance', type=float, help='average jump distance')
	#tesselate_parser.add_argument('-r', '--frame-rate', type=float, help='frame rate')
	#tesselate_parser.add_argument('-t', '--time-regularization', 
	tesselate_group2 = tesselate_parser.add_mutually_exclusive_group()
	tesselate_group2.add_argument('-w', action='store_true', help='whiten the input data')
	tesselate_group2.add_argument('--scaling', choices=['whiten', 'unit'])
	tesselate_parser.add_argument('-s', '--min-cell-count', type=int, default=20, \
		help='minimum number of points per cell')
	kdtree_parser = tesselate_parser
	kdtree_parser.add_argument('-S', '--max-cell-count', type=int, \
		help='maximum number of points per cell [kdtree]')
	kdtree_parser.add_argument('-l', '--lower-levels', type=int, \
		help='number of levels below the smallest one [kdtree]')
	grid_parser = tesselate_parser
	grid_parser.add_argument('-c', '--cell-count', type=int, default=80, \
		help='average number of points per cell [grid|kmeans]')
	#gwr_parser = tesselate_parser
	#gwr_parser.add_argument('-c', '--cell-count', type=int, default=80, \
	#	help='average number of points per cell [gwr]')
	
	# parse
	args = parser.parse_args()
	args.func(args)

