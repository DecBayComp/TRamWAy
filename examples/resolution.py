
import os
import sys
from rwa import lazyvalue
from tramway.core import find_artefacts, format_analyses, load_rwa, save_rwa
from tramway.tessellation import CellStats
from tramway.inference import Maps
from tramway.helper import tessellate, infer, map_plot, cell_plot
from pandas import DataFrame
import warnings
import traceback

short_description = 'infer and plot diffusivity and potential maps with varying spatial resolution'

data_server = 'http://dl.pasteur.fr/fop/jod5vvB5/'
data_file = 'glycine_receptor.trxyt'
data_dir = ''

verbose = True


methods = ['grid', 'kdtree', 'kmeans', 'gwr']
default_localization_error = 0.001
default_prior_diffusivity = 0.01
default_prior_potential = 0.1

default_ref_distances = [.36, .27, .18, .09]
min_location_count = 20


arguments = [
	('--grid',	dict(help='regular grid mesh', action='store_true')),
	('--kdtree',	dict(help='kd-tree mesh', action='store_true')),
	('--kmeans',	dict(help='k-means mesh', action='store_true')),
	('--gwr',	dict(help='growing-when-required-based mesh (default)', action='store_true', default=True)),
	('-d',	dict(help='D inference mode', action='store_true')),
	('-dd',	dict(help='DD inference mode', action='store_true')),
	('-df',	dict(help='DF inference mode (default)', action='store_true')),
	('-dv',	dict(help='DV inference mode', action='store_true')),
	(('--localization-error', '-le'), dict(help='localization error', type=float, default=default_localization_error)),
	(('--prior-diffusivity', '-pd'), dict(help='prior diffusivity', type=float, default=default_prior_diffusivity)),
	(('--prior-potential', '-pv'), dict(help='prior potential', type=float, default=default_prior_potential)),
	('-nc', dict(help='do NOT plot colorbars', action='store_true'))]


def artefact_size(a):
	a = lazyvalue(a, deep=True)
	if isinstance(a, DataFrame):
		return a.shape[0]
	elif isinstance(a, CellStats):
		return a.tessellation.cell_centers.shape[0]
	elif isinstance(a, Maps):
		return a.maps.shape[0]


def main(**kwargs):

	#warnings.simplefilter('error')

	xyt_file = os.path.join(data_dir, data_file)
	if not os.path.isfile(xyt_file):
		print("cannot find file: {}".format(xyt_file))
		sys.exit(1)

	output_basename, _ = os.path.splitext(xyt_file)
	output_basename = 'resolution'

	# determine which tessellation method
	method = None
	for m in methods:
		if kwargs.get(m, False):
			method = m
			break
	if not method:
		method = methods[-1] # gwr
	def label(res):
		return '{}@{}'.format(method, res)

	# tessellate
	rwa_file = output_basename + '.rwa'
	analyses = None
	analyses_or_file = xyt_file
	_modified = False
	for _i, _ref_distance in enumerate(default_ref_distances):
		_label = label(_ref_distance)
		if analyses is None and os.path.isfile(rwa_file):
			try:
				analyses_or_file = analyses = load_rwa(rwa_file)
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				print(traceback.format_exc())
		_tessellate_i = True
		if analyses is not None:
			try:
				_, = find_artefacts(analyses, CellStats, labels=_label)
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				print("couldn't find {}".format(_label))
				pass
			else:
				print("{} found".format(_label))
				_tessellate_i = False
		if _tessellate_i:
			_modified = True
			print('tessellating {}...'.format(_label))
			tessellate(analyses_or_file, method, label=_label, ref_distance=_ref_distance,
				min_location_count=5, knn=min_location_count, strict_rel_max_size=10,
				output_file=rwa_file if analyses is None else None, verbose=True)
	if analyses is None:
		analyses = load_rwa(rwa_file, verbose=True)
	elif _modified:
		save_rwa(rwa_file, analyses, force=True)
	#print(format_analyses(analyses, node=artefact_size))

	# determine which inference modes
	_d  = kwargs.get('d',  False)
	_dd = kwargs.get('dd', False)
	_df = kwargs.get('df', False)
	_dv = kwargs.get('dv', False)
	if not any((_d, _dd, _df, _dv)):
		_df = True

	localization_error = kwargs.get('locatization_error', default_localization_error)
	priorD = kwargs.get('prior_diffusivity', default_prior_diffusivity)
	priorV = kwargs.get('prior_potential', default_prior_potential)

	def img(mode, res):
		return '{}.{}.{}.{}.png'.format(output_basename, method, int(res * 100), mode)

	# infer and plot maps
	map_plot_args = dict(show=True, clip=4., point_style=dict(alpha=.01),
			colorbar=not kwargs.get('nc', False))
	if _d:
		for _ref_distance in default_ref_distances:
			_label = label(_ref_distance)
			try:
				D = analyses[_label]['D'].data
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				_modified = True
				D = infer(analyses, mode='d', input_label=_label,
					output_label='D', localization_error=localization_error)
			map_plot(D, output_file=img('d', _ref_distance),
				cells=analyses[_label].data, **map_plot_args)

	if _df:
		_map_plot_args = dict(map_plot_args) # copy
		_map_plot_args['clip'] = dict(diffusivity=map_plot_args['clip'], force=2.)
		for _ref_distance in default_ref_distances:
			_label = label(_ref_distance)
			try:
				DF = analyses[_label]['DF'].data
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				_modified = True
				DF = infer(analyses, mode='df', input_label=_label,
					output_label='DF', localization_error=localization_error)
			map_plot(DF, output_file=img('df', _ref_distance),
				cells=analyses[_label].data, **_map_plot_args)

	if _dd:
		for _ref_distance in default_ref_distances:
			_label = label(_ref_distance)
			try:
				DD = analyses[_label]['DD'].data
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				_modified = True
				DD = infer(analyses, mode='dd', input_label=_label,
					output_label='DD', localization_error=localization_error,
					priorD=priorD)
			map_plot(DD, output_file=img('dd', _ref_distance),
				cells=analyses[_label].data, **map_plot_args)

	if _dv:
		_map_plot_args = dict(map_plot_args) # copy
		_map_plot_args['clip'] = dict(diffusivity=map_plot_args['clip'], potential=2.)
		for _ref_distance in default_ref_distances:
			_label = label(_ref_distance)
			try:
				DV = analyses[_label]['DV'].data
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				_modified = True
				DV = infer(analyses, mode='dv', input_label=_label,
					output_label='DV', localization_error=localization_error,
					priorD=priorD, priorV=priorV, output_file=rwa_file)
			map_plot(DV, output_file=img('dv', _ref_distance),
				cells=analyses[_label].data, **_map_plot_args)

	if _modified:
		#print(format_analyses(analyses, node=artefact_size))
		save_rwa(rwa_file, analyses, force=True, verbose=verbose)

	sys.exit(0)


if __name__ == '__main__':
	main()

