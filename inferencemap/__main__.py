#!/usr/bin/env python

import argparse
import sys
from .helper.tesselation import *


def _render(args):
	kwargs = dict(args.__dict__)
	input_file = kwargs.pop('input', None)
	output_file = kwargs.pop('output', None)
	fig_format = kwargs.pop('print', None)
	delaunay = kwargs.pop('delaunay', False)
	hist = kwargs.pop('histogram', '')
	if hist is None: hist = ''
	if delaunay:
		kwargs['xy_layer'] = 'delaunay'
	del kwargs['func']
	del kwargs['min_cell_count']
	cell_plot(input_file, output_file=output_file, fig_format=fig_format, \
		point_count_hist='c' in hist, cell_dist_hist='d' in hist, \
		point_dist_hist='p' in hist, **kwargs)
	sys.exit(0)


def _tesselate(args):
	kwargs = dict(args.__dict__)
	input_file = kwargs.pop('input', None)
	output_file = kwargs.pop('output', None)
	scaling = kwargs.pop('w', None)
	if scaling and not kwargs['scaling']:
		kwargs['scaling'] = 'whiten'
	avg_cell_count = kwargs.pop('cell_count', None)
	max_level = kwargs.pop('lower_levels', None)
	knn = kwargs.pop('knn', None)
	min_nn, max_nn = knn, knn
	min_nn = kwargs.pop('min_nn', min_nn)
	max_nn = kwargs.pop('max_nn', max_nn)
	if not (min_nn is None and max_nn is None):
		knn = (min_nn, max_nn)
	del kwargs['func']
	del kwargs['reuse']
	tesselate(input_file, output_file=output_file, \
		avg_cell_count=avg_cell_count, max_level=max_level, \
		knn=knn, **kwargs)
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
	render_parser = sub.add_parser('show-cells')
	render_parser.set_defaults(func=_render)
	render_parser.add_argument('-s', '--min-cell-count', type=int, default=20, \
		help='minimum number of points per cell')
	render_parser.add_argument('-D', '--delaunay', action='store_true', help='plot the Delaunay graph instead of the Voronoi')
	render_parser.add_argument('-H', '--histogram', help="plot/print additional histogram(s); any combination of 'c' (cell count histogram), 'd' (distance between neighboring centers) and 'p' (distance between any pair of points from distinct neighboring centers)")
	render_parser.add_argument('-p', '--print', choices=fig_formats, help='print figure(s) on disk instead of plotting')

	# tesselate
	tesselate_parser = sub.add_parser('tesselate')
	tesselate_parser.set_defaults(func=_tesselate)
	tesselate_group1 = tesselate_parser.add_mutually_exclusive_group(required=True)
	tesselate_group1.add_argument('-m', '--method', choices=['grid', 'kdtree', 'kmeans', 'gwr'])
	tesselate_group1.add_argument('-r', '--reuse', \
		#nargs='?', type=argparse.FileType('r'), default=sys.stdin, \
		help='apply precomputed tesselation from file')
	tesselate_parser.add_argument('--knn', type=int, \
		help='number of nearest neighbors per cell center. This is equivalent to --min-nn N --max-nn N')
	tesselate_parser.add_argument('--max-nn', '--max-knn', '--knn-max', type=int, \
		help='maximum number of nearest neighbors')
	tesselate_parser.add_argument('-n', '--min-nn', '--min-knn', '--knn-min', type=int, \
		help='minimum number of nearest neighbors. Cells can overlap')
	#tesselate_parser.add_argument('--overlap', action='store_true', help='allow cells to overlap (useful with knn)')
	tesselate_parser.add_argument('-d', '--distance', type=float, help='average jump distance')
	#tesselate_parser.add_argument('-r', '--frame-rate', type=float, help='frame rate')
	#tesselate_parser.add_argument('-t', '--time-scale', type=float, nargs='+', help='time multiplier(s) for tesselation(s) 2/3d + t')
	tesselate_parser.add_argument('-t', '--time-scale', type=float, help='time multiplier for 2/3d+t tesselation')
	tesselate_group2 = tesselate_parser.add_mutually_exclusive_group()
	tesselate_group2.add_argument('-w', action='store_true', help='whiten the input data')
	tesselate_group2.add_argument('--scaling', choices=['whiten', 'unit'])
	tesselate_parser.add_argument('-s', '--min-cell-count', type=int, default=20, \
		help='minimum number of points per cell. This affects the tesselation only and not directly the partition. See --min-knn for a partition-related parameter')
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

