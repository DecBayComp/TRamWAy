# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import *
from tramway.inference import *
import tramway.inference as inference # inference.plugins
from tramway.plot.map import *
from tramway.helper.tessellation import *
import matplotlib.pyplot as plt
from warnings import warn
import os
from time import time
import collections
import traceback


#sub_extensions = dict([(ext.upper(), ext) for ext in ['d', 'df', 'dd', 'dv', 'dx']])


def infer(cells, mode='D', output_file=None, partition={}, verbose=False, \
	localization_error=None, diffusivity_prior=None, potential_prior=None, jeffreys_prior=None, \
	max_cell_count=None, dilation=None, worker_count=None, min_diffusivity=0, \
	store_distributed=False, constructor=None, cell_sampling=None, \
	priorD=None, priorV=None, input_label=None, output_label=None, comment=None, \
	return_cells=None, profile=None, force=False, **kwargs):
	"""
	Inference helper.

	Arguments:

		cells (str or CellStats or Analyses): data partition or path to partition file

		mode (str or callable): plugin name; see for example 
			:mod:`~tramway.inference.d` (``'d'``), 
			:mod:`~tramway.inference.df` (``'df'``), 
			:mod:`~tramway.inference.dd` (``'dd'``), 
			:mod:`~tramway.inference.dv` (``'dv'``); 
			can be also a function suitable for :meth:`Distributed.run`

		output_file (str): desired path for the output map file

		partition (dict): keyword arguments for :func:`~tramway.helper.tessellation.find_partition`
			if `cells` is a path; deprecated

		verbose (bool or int): verbosity level

		localization_error (float): localization error

		prior_diffusivity (float): prior diffusivity

		prior_potential (float): prior potential

		jeffreys_prior (float): Jeffreys' prior

		max_cell_count (int): if defined, divide the mesh into convex subsets of cells

		dilation (int): overlap of side cells if `max_cell_count` is defined

		worker_count (int): number of parallel processes to be spawned

		min_diffusivity (float): (possibly negative) lower bound on local diffusivities

		store_distributed (bool): store the :class:`~tramway.inference.base.Distributed` object 
			in the map file

		constructor (callable): see also :func:`~tramway.inference.base.distributed`

		cell_sampling (str): either ``None``, ``'individual'`` or ``'group'``; may ignore 
			`max_cell_count` and `dilation`

		input_label (list): label path to the input :class:`~tramway.tessellation.base.Tessellation`
			object in `cells` if the latter is an `Analyses` or filepath

		output_label (str): label for the resulting analysis instance

		comment (str): description message for the resulting analysis

		return_cells (bool): return a tuple with a :class:`~tramway.tessellation.base.CellStats` 
			object as extra element

		profile (bool or str): profile each child job if any;
			if `str`, dump the output stats into *.prof* files;
			if `tuple`, print a report with :func:`~pstats.Stats.print_stats` and
			tuple elements as input arguments.

	Returns:

		Maps or pandas.DataFrame or tuple:

	`priorD` and `priorV` are legacy arguments. 
	They are deprecated and `diffusivity_prior` and `potential_prior` should be used instead
	respectively.
	"""

	input_file = None
	all_analyses = analysis = None
	if isinstance(cells, str):
		try:
			input_file = cells
			all_analyses = load_rwa(input_file)
			if output_file and output_file == input_file:
				all_analyses = extract_analysis(all_analyses, input_label)
			cells = None
		except KeyError:
			# legacy format
			input_file, cells = find_partition(cells, **partition)
			if cells is None:
				raise ValueError('no cells found')
		if verbose:
			print('loading file: {}'.format(input_file))
	elif isinstance(cells, Analyses):
		all_analyses, cells = cells, None
	elif not isinstance(cells, CellStats):
		raise TypeError('wrong type for argument `cells`')

	if cells is None:
		if not all_analyses:
			raise ValueError('no cells found')
		if not input_label:
			labels = tuple(all_analyses.labels)
			if labels[1:]:
				raise ValueError('multiple instances; input_label is required')
			input_label = labels[-1]
		if isinstance(input_label, (tuple, list)):
			if input_label[1:]:
				analysis = all_analyses
				for label in input_label[:-1]:
					analysis = analysis[label]
				cells = analysis.data
				analysis = analysis[input_label[-1]]
				if not isinstance(cells, CellStats):
					cells = analysis.data
			else:
				input_label = input_label[0]
		if cells is None:
			analysis = all_analyses[input_label]
			cells = analysis.data
		if not isinstance(cells, CellStats):
			raise ValueError('cannot find cells at the specified label')
	elif all_analyses is None:
		all_analyses = Analyses(cells.points)
		assert analysis is None
		analysis = Analyses(cells)
		all_analyses.add(analysis)
		assert input_label is None
		input_label = tuple(all_analyses.labels)

	if mode in ('D', 'DF', 'DD', 'DV'):
		mode = mode.lower()
	setup, module = inference.plugins[mode]

	if isinstance(analysis.data, Distributed):
		_map = analysis.data
	else:

		if cells is None or cells.tessellation is None:
			raise ValueError('no cells found')

		# prepare the data for the inference
		if constructor is None:
			constructor = Distributed
		detailled_map = distributed(cells, new=constructor)

		if cell_sampling is None:
			try:
				cell_sampling = setup['cell_sampling']
			except KeyError:
				pass
		multiscale = cell_sampling in ['individual', 'group']
		if multiscale:
			if max_cell_count is None:
				if cell_sampling == 'individual':
					max_cell_count = 1
				else:
					max_cell_count = 20
			if dilation is None:
				if cell_sampling == 'individual':
					dilation = 0
				else:
					dilation = 2
			multiscale_map = detailled_map.group(max_cell_count=max_cell_count, \
				adjacency_margin=dilation)
			_map = multiscale_map
		else:
			_map = detailled_map

		if store_distributed:
			if output_label is None:
				output_label = analysis.autoindex()
			analysis.add(Analysis(_map), label=output_label)
			analysis = analysis[output_label]
			output_label = None

	runtime = time()

	if mode in inference.plugins:

		args = setup.get('arguments', {})
		for arg in ('localization_error', 'diffusivity_prior', 'potential_prior',
				'jeffreys_prior', 'min_diffusivity', 'worker_count'):
			try:
				args[arg]
			except KeyError:
				pass
			else:
				val = eval(arg)
				if val is not None:
					kwargs[arg] = val
		if profile:
			kwargs['profile'] = profile
		x = _map.run(getattr(module, setup['infer']), **kwargs)

	else:
		raise ValueError('unknown ''{}'' mode'.format(mode))

	maps = Maps(x, mode=mode)
	for p in kwargs:
		if p not in ['worker_count']:
			setattr(maps, p, kwargs[p])
	analysis.add(Analyses(maps), label=output_label, comment=comment)

	runtime = time() - runtime
	if verbose:
		print('{} mode: elapsed time: {}ms'.format(mode, int(round(runtime*1e3))))
	maps.runtime = runtime

	if input_file and not output_file:
		output_file = input_file

	if output_file:
		# store the result
		save_rwa(output_file, all_analyses, verbose, force=input_file == output_file or force)

	if return_cells == True: # NOT `is`
		return (maps, cells)
	elif return_cells == False:
		return maps
	elif input_file:
		if return_cells is not None:
			warn("3-element return value will no longer be the default; pass return_cells='first' to maintain this behavior", FutureWarning)
		return (cells, mode, x)
	else:
		return x


def map_plot(maps, cells=None, clip=None, output_file=None, fig_format=None, \
	figsize=(24.0, 18.0), dpi=None, aspect=None, show=None, verbose=False, \
	alpha=None, point_style=None, \
	label=None, input_label=None, mode=None, \
	**kwargs):
	"""
	Plot scalar/vector 2D maps.

	Arguments:

		maps (str or Analyses or pandas.DataFrame or Maps): maps as a path to a rwa map file, 
			an analysis tree, a dataframe or a :class:`Maps`;
			filepaths and analysis trees may require `label` (or equivalently `input_label`)
			to be defined; dataframes and encapsulated maps require `cells` to be defined

		cells (CellStats or Tessellation): mesh with optional partition

		clip (float): clips map values by absolute values;
			if ``clip < 1``, it is the quantile at which to clip absolute values of the map;
			otherwise it defines: ``threshold = median + clip * (third_quartile - first_quartile)``

		output_file (str): path to output file

		fig_format (str): for example '*.png*'

		figsize ((float, float)): figure size

		dpi (int): dot per inch

		aspect (float or str): aspect ratio or '*equal*'

		show (bool or str): call :func:`~matplotlib.pyplot.show`; if ``show='draw'``, call
			:func:`~matplotlib.pyplot.draw` instead

		verbose (bool): verbosity level

		alpha (float): alpha value for scalar maps; useful in combination with `point_style`

		point_style (dict): if defined, points are overlaid

		label/input_label (int or str): analysis instance label

		mode (bool or str): inference mode; can be ``False`` so that mode information from
			files, analysis trees and encapsulated maps are not displayed
	"""

	# get cells and maps objects from the first input argument
	input_file = None
	if isinstance(maps, tuple):
		warn('`maps` as (CellStats, str, DataFrame) tuple are deprecated', DeprecationWarning)
		cells, mode, maps = maps
	elif isinstance(maps, (pd.DataFrame, Maps)):
		if cells is None:
			raise ValueError('`cells` is not defined')
	elif isinstance(maps, Analyses):
		analyses = maps
		if label is None:
			label = input_label
		cells, maps = find_artefacts(analyses, (CellStats, Maps), label)
	else: # `maps` is a file path
		input_file = maps
		if label is None:
			label = input_label
		try:
			analyses = load_rwa(input_file)
			#if label:
			#	analyses = extract_analysis(analyses, label)
		except KeyError:
			print(traceback.format_exc())
			try:
				# old format
				store = HDF5Store(input_file, 'r')
				store.lazy = False
				maps = peek_maps(store, store.store)
			finally:
				store.close()
			try:
				tess_file = maps.rwa_file
			except AttributeError:
				# even older
				tess_file = maps.imt_file
			if not isinstance(tess_file, str):
				tess_file = tess_file.decode('utf-8')
			tess_file = os.path.join(os.path.dirname(input_file), tess_file)
			store = HDF5Store(tess_file, 'r')
			store.lazy = False
			try:
				cells = store.peek('cells')
				if cells.tessellation is None:
					cells._tessellation = store.peek('_tesselation', store.store['cells'])
			finally:
				store.close()
		except ImportError:
			warn('HDF5 libraries may not be installed', ImportWarning)
		else:
			cells, maps = find_artefacts(analyses, (CellStats, Maps), label)
	if isinstance(maps, Maps):
		if mode != False:
			mode = maps.mode
		maps = maps.maps

	# `mode` type may be inadequate because of loading a Py2-generated rwa file in Py3 or conversely
	if mode and not isinstance(mode, str):
		try: # Py2
			mode = mode.encode('utf-8')
		except AttributeError: # Py3
			mode = mode.decode('utf-8')

	# output filenames
	print_figs = output_file or (input_file and fig_format)

	if print_figs:
		if output_file:
			filename, figext = os.path.splitext(output_file)
			if fig_format:
				figext = fig_format
			elif figext and figext[1:] in fig_formats:
				figext = figext[1:]
			else:
				figext = fig_formats[0]
		else:
			figext = fig_format
			filename, _ = os.path.splitext(input_file)

	# identify and plot the possibly various maps
	figs = []

	all_vars = splitcoord(maps.columns)
	scalar_vars = {'diffusivity': 'D', 'potential': 'V'}
	scalar_vars = [ (v, scalar_vars.get(v, None)) for v in all_vars if len(all_vars[v]) == 1 ]

	for col, short_name in scalar_vars:

		col_kwargs = {}
		for a in kwargs:
			if isinstance(kwargs[a], (dict, pd.Series, pd.DataFrame)) and col in kwargs[a]:
				col_kwargs[a] = kwargs[a][col]
			else:
				col_kwargs[a] = kwargs[a]

		if figsize:
			fig = plt.figure(figsize=figsize)
		else:
			fig = plt.gcf()
		figs.append(fig)

		_map = maps[col]
		if isinstance(clip, (dict, pd.Series, pd.DataFrame)):
			try:
				__clip = clip[col]
			except:
				__clip = None
		else:
			__clip = clip
		if __clip:
			_map = _clip(_map, __clip)
		scalar_map_2d(cells, _map, aspect=aspect, alpha=alpha, **col_kwargs)

		if point_style is not None:
			points = cells.descriptors(cells.points, asarray=True) # `cells` should be a `CellStats`
			if 'color' not in point_style:
				point_style['color'] = None
			plot_points(points, **point_style)

		if mode:
			if short_name:
				title = '{} ({} - {} mode)'.format(short_name, col, mode)
			else:
				title = '{} ({} mode)'.format(col, mode)
		elif short_name:
			title = '{} ({})'.format(short_name, col)
		else:
			title = '{}'.format(col)
		plt.title(title)

		if print_figs:
			if maps.shape[1] == 1:
				figfile = '{}.{}'.format(filename, figext)
			else:
				figfile = '{}_{}.{}'.format(filename, short_name.lower(), figext)
			if verbose:
				print('writing file: {}'.format(figfile))
			fig.savefig(figfile, dpi=dpi)

	vector_vars = {'force': 'F'}
	vector_vars = [ (v, vector_vars.get(v, None)) for v in all_vars if len(all_vars[v]) == 2 ]
	for name, short_name in vector_vars:
		cols = all_vars[name]

		var_kwargs = {}
		for a in kwargs:
			if isinstance(kwargs[a], (dict, pd.Series, pd.DataFrame)) and name in kwargs[a]:
				var_kwargs[a] = kwargs[a][name]
			else:
				var_kwargs[a] = kwargs[a]
		
		if figsize:
			fig = plt.figure(figsize=figsize)
		else:
			fig = plt.gcf()
		figs.append(fig)

		_vector_map = maps[cols]
		if isinstance(clip, (dict, pd.Series, pd.DataFrame)):
			try:
				__clip = clip[name]
			except:
				__clip = None
		else:
			__clip = clip
		if __clip:
			_vector_map = _clip(_vector_map, __clip)
		if point_style is None:
			field_map_2d(cells, _vector_map, aspect=aspect, **var_kwargs)
		else:
			_scalar_map = _vector_map.pow(2).sum(1).apply(np.sqrt)
			scalar_map_2d(cells, _scalar_map, aspect=aspect, alpha=alpha, **var_kwargs)
			points = cells.descriptors(cells.points, asarray=True) # `cells` should be a `CellStats`
			if 'color' not in point_style:
				point_style['color'] = None
			plot_points(points, **point_style)
			field_map_2d(cells, _vector_map, aspect=aspect, overlay=True, **var_kwargs)

		extra = None
		if short_name:
			main = short_name
			extra = name
		else:
			main = name
		if mode:
			if extra:
				extra += ' - {} mode'.format(mode)
			else:
				extra = '{} mode'.format(mode)
		if extra:
			title = '{} ({})'.format(main, extra)
		else:
			title = main
		plt.title(title)

		if print_figs:
			if maps.shape[1] == 1:
				figfile = '{}.{}'.format(filename, figext)
			else:
				if short_name:
					ext = short_name.lower()
				else:
					ext = keyword
				figfile = '{}_{}.{}'.format(filename, ext, figext)
			if verbose:
				print('writing file: {}'.format(figfile))
			fig.savefig(figfile, dpi=dpi)

	if show or not print_figs:
		if show == 'draw':
			plt.draw()
		elif show is not False:
			plt.show()
	elif print_figs:
		for fig in figs:
			plt.close(fig)


def _clip(m, q):
	if q <= 0:
		return m
	amplitude = m.pow(2)
	if m.shape[1:]:
		amplitude = amplitude.sum(1)
		columns = m.columns
	amplitude = amplitude.apply(np.sqrt)
	if q < 1:
		amax = amplitude.quantile(q)
	else:
		amax = amplitude.quantile(.5) + q * (amplitude.quantile(.75) - amplitude.quantile(.25))
	amplitude = amplitude.values
	exceed = amplitude > amax
	factor = amax / amplitude[exceed]
	M, index, m = type(m), m.index, m.values
	if m.shape[1:]:
		m[exceed, :] = m[exceed, :] * factor[:, np.newaxis]
		m = M(m, columns=columns, index=index)
	else:
		m[exceed] = m[exceed] * factor
		m = M(m, index=index)
	return m

