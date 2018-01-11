# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import argparse
try:
	from ConfigParser import ConfigParser
except ImportError:
	from configparser import ConfigParser
import sys
from .helper import *
from .feature import *
import tramway.tesselation.plugins as tesselation


def _parse_args(args):
	kwargs = dict(args.__dict__)
	del kwargs['func']
	try:
		input_file = kwargs.pop('input')
		input_file[0]
	except (KeyError, IndexError):
		print('please specify input file(s) with -i')
		sys.exit(1)
	return input_file, kwargs


def _render_cells(args):
	input_file, kwargs = _parse_args(args)
	output_file = kwargs.pop('output', None)
	fig_format = kwargs.pop('print', None)
	delaunay = kwargs.pop('delaunay', False)
	hist = kwargs.pop('histogram', '')
	if hist is None: hist = ''
	if delaunay:
		kwargs['xy_layer'] = 'delaunay'
	del kwargs['min_location_count']
	cell_plot(input_file, output_file=output_file, fig_format=fig_format, \
		location_count_hist='c' in hist, cell_dist_hist='d' in hist, \
		location_dist_hist='p' in hist, **kwargs)
	sys.exit(0)


def _tesselate(args):
	input_file, kwargs = _parse_args(args)
	output_file = kwargs.pop('output', None)
	scaling = kwargs.pop('w', None)
	if scaling and not kwargs['scaling']:
		kwargs['scaling'] = 'whiten'
	avg_location_count = kwargs.pop('location_count', None)
	max_level = kwargs.pop('lower_levels', None)
	min_nn = kwargs.pop('knn', None)
	max_nn = kwargs.pop('max_nn', None)
	if not (min_nn is None and max_nn is None):
		knn = (min_nn, max_nn)
	else:
		knn = None
	if kwargs.get('inplace', not None) is None:
		del kwargs['inplace']
	if kwargs['method'] is None:
		del kwargs['method']
	elif kwargs['method'] == 'kdtree' and min_nn is not None:
		kwargs['metric'] = 'euclidean'
	tesselate(input_file, output_file=output_file, \
		avg_location_count=avg_location_count, max_level=max_level, \
		knn=knn, **kwargs)
	sys.exit(0)

def _infer(mode):
	def __infer(args):
		input_file, kwargs = _parse_args(args)
		output_file = kwargs.pop('output', None)
		kwargs['mode'] = mode
		infer(input_file[0], output_file=output_file, **kwargs)
		# kwargs: mode, localization_error, diffusivity_prior, potential_prior, jeffreys_prior
		sys.exit(0)
	return __infer

def _render_map(args):
	input_file, kwargs = _parse_args(args)
	output_file = kwargs.pop('output', None)
	fig_format = kwargs.pop('print', None)
	map_plot(input_file[0], output_file, fig_format, **kwargs)
	sys.exit(0)

def _dump_rwa(args):
	input_files, kwargs = _parse_args(args)
	verbose = kwargs.get('verbose', False)
	for input_file in input_files:
		print('in {}:'.format(input_file))
		store = HDF5Store(input_file, 'r', verbose)
		try:
			analyses = store.peek('analyses')
		except EnvironmentError:
			print(traceback.format_exc())
			raise OSError('HDF5 libraries may not be installed')
		except KeyError:
			print('no analyses found')
		else:
			print(format_analyses(analyses, global_prefix='\t'))
		finally:
			store.close()

def _curl(args):
	input_file, kwargs = _parse_args(args)
	input_label = kwargs.get('input_label', kwargs.get('label', None))
	if input_file[1:]:
		raise NotImplementedError('cannot handle multiple input files')
	input_file = input_file[0]
	analyses = load_rwa(input_file)
	cells, maps = find_artefacts(analyses, (CellStats, Maps), input_label)
	curl = Curl(cells, maps)
	vector_fields = { f: vs for f, vs in curl.variables.items() if len(vs) == 2 }
	curl_name = kwargs.get('output_label', None)
	if not curl_name:
		curl_name = 'curl'
	if 1 < len(vector_fields):
		raise NotImplementedError('multiple vector fields')
	distance = kwargs.get('distance', 1)
	for f in vector_fields:
		_name = '{} {} {}'.format(curl_name, f, distance)
		curl.extract(_name, f, distance)
	output_file = kwargs.get('output_file', input_file)
	save_rwa(output_file, analyses, force=output_file == input_file)



def main():
	parser = argparse.ArgumentParser(prog='tramway',
		description='TRamWAy central command.',
		epilog='See also https://github.com/DecBayComp/TRamWAy',
		conflict_handler='resolve')
	global_arguments = [
		('-v', '--verbose', dict(action='count', help='increase verbosity')),
		('-i', '--input', dict(action='append', default=[],
			metavar='INPUT_FILE', help='path to input file or directory')),
		('-o', '--output', dict(metavar='OUTPUT_FILE', help='path to output file'))]
	for arg1, arg2, kwargs in global_arguments:
		parser.add_argument(arg1, arg2, dest=arg1[1]+'pre', **kwargs)
	sub = parser.add_subparsers(title='commands', \
		description="type '%(prog)s command --help' for additional help")


	# tesselate
	try:
		tesselate_parser = sub.add_parser('sample', aliases=['tesselate'])
	except TypeError: # Py2
		tesselate_parser = sub.add_parser('sample')
	tesselate_parser.set_defaults(func=_tesselate)
	for arg1, arg2, kwargs in global_arguments:
		tesselate_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
	tesselate_parser.add_argument('-L', '--input-label', help='input label')
	tesselate_parser.add_argument('--inplace', action='store_true', \
		help='replace the input sampling by the output one (only when --input-label is defined)')
	tesselate_parser.add_argument('-l', '--label', '--output-label', help='output label')
	tesselate_parser.add_argument('--comment', help='description message for the output artefact')
	tesselate_parser.add_argument('-m', '--method', choices=['grid', 'kdtree', 'kmeans', 'gwr'] + \
		list(tesselation.all_plugins.keys()))
	tesselate_parser.add_argument('-n', '--knn', '--min-nn', '--min-knn', '--knn-min', type=int, \
		help='minimum number of nearest neighbors. Cells can overlap')
	tesselate_parser.add_argument('--max-nn', '--max-knn', '--knn-max', type=int, \
		help='maximum number of nearest neighbors')
	tesselate_parser.add_argument('-d', '--distance', type=float, help='average jump distance')
	#tesselate_parser.add_argument('-r', '--frame-rate', type=float, help='frame rate')
	#tesselate_parser.add_argument('-t', '--time-scale', type=float, nargs='+', help='time multiplier(s) for tesselation(s) 2/3d + t')
	tesselate_parser.add_argument('-t', '--time-scale', type=float, help='time multiplier for 2/3d+t tesselation')
	tesselate_group2 = tesselate_parser.add_mutually_exclusive_group()
	tesselate_group2.add_argument('-w', action='store_true', help='whiten the input data')
	tesselate_group2.add_argument('--scaling', choices=['whiten', 'unit'])
	tesselate_parser.add_argument('-s', '--min-location-count', type=int, default=20, \
		help='minimum number of locations per cell. This affects the tesselation only and not directly the partition. See --min-knn for a partition-related parameter')
	kdtree_parser = tesselate_parser
	kdtree_parser.add_argument('-S', '--max-location-count', type=int, \
		help='maximum number of locations per cell [kdtree]')
	kdtree_parser.add_argument('-ll', '--lower-levels', type=int, \
		help='number of levels below the smallest one [kdtree]')
	grid_parser = tesselate_parser
	grid_parser.add_argument('-c', '--location-count', type=int, default=80, \
		help='average number of locations per cell [grid|kmeans]')
	#gwr_parser = tesselate_parser
	#gwr_parser.add_argument('-c', '--location-count', type=int, default=80, \
	#	help='average number of locations per cell [gwr]')
	plugin_parser = tesselate_parser
	for plugin in tesselation.all_plugins:
		setup, _ = tesselation.all_plugins[plugin]
		if 'make_arguments' in setup:
			for arg in setup['make_arguments']:
				arg_kwargs = setup['make_arguments'][arg]
				if arg_kwargs:
					try:
						arg_args = ('--'+arg.replace('_','-'),)
						if isinstance(arg_kwargs, (tuple, list)):
							arg_short, arg_kwargs = arg_kwargs
							arg_args = (arg_short,)+arg_args
						plugin_parser.add_argument(*arg_args, **arg_kwargs)
					except:
						print("in plugin '{}': error inserting argument '{}':".format(plugin, arg))
						print(traceback.format_exc())


	# infer
	infer_parser = sub.add_parser('infer') #, conflict_handler='resolve'
	isub = infer_parser.add_subparsers(title='modes', \
		description="type '%(prog)s infer mode --help' for additional help about mode")
	for mode in all_modes:
		mode_parser = isub.add_parser(mode)
		setup, _ = all_modes[mode]
		short_args = [ args[0] for args in setup.get('arguments', {}).values()
				if isinstance(args, (tuple, list)) ]
		mode_parser.set_defaults(func=_infer(mode))
		for short_arg, long_arg, kwargs in global_arguments:
			dest = short_arg[1:] + 'post'
			if short_arg in short_args:
				mode_parser.add_argument(long_arg, dest=dest, **kwargs)
			else:
				mode_parser.add_argument(short_arg, long_arg, dest=dest, **kwargs)
		mode_parser.add_argument('-L', '--input-label', help='comma-separated list of input labels')
		mode_parser.add_argument('-l', '--output-label', help='output label')
		mode_parser.add_argument('--comment', help='description message for the output artefact')
		#infer_parser.add_argument('-m', '--mode', \
		#	choices=['D', 'DF', 'DD', 'DV'] + list(all_modes.keys()),
		#	help='inference mode') #, metavar='INFERENCE_MODE'
		args = setup['arguments']
		for arg in args:
			long_arg = '--' + arg.replace('_', '-')
			parser_args = args[arg]
			if isinstance(parser_args, (tuple, list)):
				short_arg, parser_args = parser_args
				mode_parser.add_argument(short_arg, long_arg, **parser_args)
			else:
				mode_parser.add_argument(long_arg, **parser_args)


	# dump analysis tree
	dump_parser = sub.add_parser('dump')
	dump_parser.set_defaults(func=_dump_rwa)
	for arg1, arg2, kwargs in global_arguments:
		dump_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)


	# extract features
	feature_parser = sub.add_parser('extract')
	fsub = feature_parser.add_subparsers(title='features', \
		description="type '%(prog)s extract feature --help' for additional help")

	# extract curl
	curl_parser = fsub.add_parser('curl')
	curl_parser.set_defaults(func=_curl)
	for arg1, arg2, kwargs in global_arguments:
		if arg1 in ['-v']:
			curl_parser.add_argument(arg2, dest=arg1[1]+'post', **kwargs)
		else:
			curl_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
	curl_parser.add_argument('-L', '--label', '--input-label', help='comma-separated list of input labels')
	curl_parser.add_argument('-l', '--output-label', help='output label')
	curl_parser.add_argument('-d', '--distance', type=int, default=1, help='radius in number of cells')


	# plot artefacts
	try:
		plot_parser = sub.add_parser('draw', aliases=['show'])
	except TypeError: # Py2
		plot_parser = sub.add_parser('draw')
	psub = plot_parser.add_subparsers(title='show artefacts', \
		description="type %(prog)s draw artefact --help for additional help")

	# plot cells
	cells_parser = psub.add_parser('cells')
	cells_parser.set_defaults(func=_render_cells)
	for arg1, arg2, kwargs in global_arguments:
		cells_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
	cells_parser.add_argument('-L', '--input-label', help='comma-separated list of input labels')
	cells_parser.add_argument('-s', '--min-location-count', type=int, default=20, \
		help='minimum number of locations per cell')
	cells_parser.add_argument('-D', '--delaunay', action='store_true', help='plot the Delaunay graph instead of the Voronoi')
	cells_parser.add_argument('-H', '--histogram', help="plot/print additional histogram(s); any combination of 'c' (cell count histogram), 'd' (distance between neighboring centers) and 'p' (distance between any pair of locations from distinct neighboring centers)")
	cells_parser.add_argument('-p', '--print', choices=fig_formats, help='print figure(s) on disk instead of plotting')

	# plot map(s)
	map_parser = psub.add_parser('map')
	map_parser.set_defaults(func=_render_map)
	for arg1, arg2, kwargs in global_arguments:
		map_parser.add_argument(arg1, arg2, dest=arg1[1]+'post', **kwargs)
	map_parser.add_argument('-L', '--input-label', help='comma-separated list of input labels')
	map_parser.add_argument('-p', '--print', choices=fig_formats, help='print figure(s) on disk instead of plotting')



	# parse
	args = parser.parse_args()
	args.verbose = args.vpre
	try:
		args.vpost
	except AttributeError:
		args.input = args.ipre
		args.output = args.opre
	else:
		if args.vpost:
			if args.verbose is None:
				args.verbose = args.vpost
			else:
				args.verbose += args.vpost
		args.input = args.ipre + args.ipost
		args.output = args.opre if args.opre else args.opost
		del args.vpost
		del args.ipost
		del args.opost
	del args.vpre
	del args.ipre
	del args.opre
	if args.verbose is None:
		args.verbose = False
	labels = None
	try:
		labels = args.input_label
	except AttributeError:
		try:
			labels = args.label
		except AttributeError:
			pass
	if labels:
		if labels[0] in "'\"" and labels[0] == labels[-1]:
			labels = labels[1:-1]
		labels = labels.split(',')
		args.input_label = []
		for label in labels:
			try:
				label = int(label)
			except (TypeError, ValueError):
				pass
			args.input_label.append(label)
	try:
		args.output_label = int(args.output_label)
	except (AttributeError, TypeError, ValueError):
		pass
	try:
		args.func(args)
	except AttributeError as e:
		if e.args and 'Namespace' in e.args[0]:
			parser.print_help()
		else:
			raise



if __name__ == '__main__':
	main()

