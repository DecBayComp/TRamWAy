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
import tramway.tessellation as tessellation
import tramway.inference as inference
from .feature import *
from .helper import *
import tramway.core.hdf5.compat


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


def _tessellate(args):
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
	tessellate(input_file, output_file=output_file, \
		avg_location_count=avg_location_count, max_level=max_level, \
		knn=knn, **kwargs)
	sys.exit(0)


def _sample(method, parse_extra=None):
	def sample(args):
		input_file, kwargs = _parse_args(args)
		output_file = kwargs.pop('output', None)
		scaling = kwargs.pop('w', None)
		if scaling and not kwargs['scaling']:
			kwargs['scaling'] = 'whiten'
		min_nn = kwargs.pop('knn', None)
		max_nn = kwargs.pop('max_nn', None)
		if not (min_nn is None and max_nn is None):
			kwargs['knn'] = (min_nn, max_nn)
		if parse_extra:
			for extra_arg, parse_arg in parse_extra:
				kwargs[extra_arg] = parse_arg(**kwargs)
		kwargs = { kw: arg for kw, arg in kwargs.items() if arg is not None }
		tessellate(input_file, output_file=output_file, method=method, **kwargs)
		sys.exit(0)
	return sample

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
		store.lazy = True
		try:
			analyses = lazyvalue(store.peek('analyses'))
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
			metavar='INPUT_FILE', help='path to input file')),
		('-o', '--output', dict(metavar='OUTPUT_FILE', help='path to output file'))]
	for arg1, arg2, kwargs in global_arguments:
		parser.add_argument(arg1, arg2, dest=arg1[1]+'pre', **kwargs)
	sub = parser.add_subparsers(title='commands', \
		description="type '%(prog)s command --help' for additional help")


	# tessellate
	try:
		tessellate_parser = sub.add_parser('tessellate', aliases=['sample'])
	except TypeError: # Py2
		tessellate_parser = sub.add_parser('tessellate')
	tsub = tessellate_parser.add_subparsers(title='methods', \
		description="type '%(prog)s sample method --help' for additional help about method")
	for method in tessellation.plugins:
		method_parser = tsub.add_parser(method)
		setup, _ = tessellation.plugins[method]
		short_args = short_options(setup.get('make_arguments', {}))
		for short_arg, long_arg, kwargs in global_arguments:
			dest = short_arg[1:] + 'post'
			if short_arg in short_args:
				method_parser.add_argument(long_arg, dest=dest, **kwargs)
			else:
				method_parser.add_argument(short_arg, long_arg, dest=dest, **kwargs)
		method_parser.add_argument('-l', '--label', '--output-label', help='output label')
		method_parser.add_argument('--comment', help='description message')
		method_parser.add_argument('-L', '--input-label', help='comma-separated list of input labels')
		method_parser.add_argument('--inplace', action='store_true', \
			help='replace the input sampling by the output one (only when --input-label is defined)')
		method_parser.add_argument('-n', '--knn', '--min-nn', '--knn-min', type=int, \
			help='minimum number of nearest neighbors; cells can overlap')
		method_parser.add_argument('-N', '--max-nn', '--knn-max', type=int, \
			help='maximum number of nearest neighbors')
		method_parser.add_argument('-d', '--distance', type=float, help='average jump distance')
		method_group = method_parser.add_mutually_exclusive_group()
		method_group.add_argument('-w', action='store_true', help='whiten the input data')
		method_group.add_argument('--scaling', choices=['whiten', 'unit'])
		method_parser.add_argument('-s', '--min-location-count', type=int, default=20, \
			help='minimum number of locations per cell; this affects the tessellation only and not directly the partition; see --knn for a partition-related parameter')
		translations = add_arguments(method_parser, setup.get('make_arguments', {}), name=method)
		method_parser.set_defaults(func=_sample(method, translations))


	# infer
	infer_parser = sub.add_parser('infer') #, conflict_handler='resolve'
	isub = infer_parser.add_subparsers(title='modes', \
		description="type '%(prog)s infer mode --help' for additional help about mode")
	for mode in inference.plugins:
		mode_parser = isub.add_parser(mode)
		setup, _ = inference.plugins[mode]
		short_args = short_options(setup.get('arguments', {}))
		for short_arg, long_arg, kwargs in global_arguments:
			dest = short_arg[1:] + 'post'
			if short_arg in short_args:
				mode_parser.add_argument(long_arg, dest=dest, **kwargs)
			else:
				mode_parser.add_argument(short_arg, long_arg, dest=dest, **kwargs)
		mode_parser.add_argument('-L', '--input-label', help='comma-separated list of input labels')
		mode_parser.add_argument('-l', '--output-label', help='output label')
		mode_parser.add_argument('--comment', help='description message for the output artefact')
		try:
			add_arguments(mode_parser, setup['arguments'], name=mode)
		except KeyError:
			pass
		mode_parser.set_defaults(func=_infer(mode))


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

