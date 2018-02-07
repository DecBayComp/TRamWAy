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
from tramway.helper.analysis import *
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
	**kwargs):
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

	Returns:

		pandas.DataFrame or tuple:

	`priorD` and `priorV` are legacy arguments. 
	They are deprecated and `diffusivity_prior` and `potential_prior` should be used instead
	respectively.
	"""

	input_file = None
	all_analyses = analysis = None
	if isinstance(cells, str):
		try:
			input_file = cells
			if output_file and output_file == input_file:
				all_analyses = find_analysis(input_file)
			else:
				all_analyses = find_analysis(input_file, input_label)
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
					dilation = 1
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
				kwargs[arg] = eval(arg)
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
		save_rwa(output_file, all_analyses, verbose, force=input_file == output_file)

	if input_file:
		return (cells, mode, x)
	else:
		return x


def map_plot(maps, output_file=None, fig_format=None, \
	show=False, verbose=False, figsize=(24.0, 18.0), dpi=None, aspect=None, \
	cells=None, mode=None, clip=None, label=None, input_label=None, \
	**kwargs):
	"""
	Plot scalar/vector 2D maps.

	Arguments:

		maps (str or pandas.DataFrame or tuple): maps as a path to a rwa map file, a dataframe 
			(`cells` must be defined) or a (:class:`CellStats`, :class:`str`, :class:`pandas.DataFrame`)
			tuple

		output_file (str): path to output file

		fig_format (str): for example *'.png'*

		show (bool): call ``matplotlib.pyplot.show()``

		verbose (bool): verbosity level

		figsize ((float, float)): figure size

		dpi (int): dot per inch

		aspect (float or str): aspect ratio or *'equal'*

		cells (CellStats or Tessellation): mesh

		mode (str): inference mode

		clip (float): quantile at which to clip absolute values of the map

		label/input_label (int or str): analysis instance label
	"""

	if isinstance(maps, tuple):
		cells, mode, maps = maps
		input_file = None
	elif isinstance(maps, pd.DataFrame):
		if cells is None:
			raise ValueError('`cells` is not defined')
	else:
		input_file = maps
		if label is None:
			label = input_label
		try:
			analyses = find_analysis(input_file, label)
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
			finally:
				store.close()
		except ImportError:
			warn('HDF5 libraries may not be installed', ImportWarning)
		else:
			cells, maps = find_artefacts(analyses, (CellStats, Maps), label)
		mode = maps.mode
		maps = maps.maps

	if mode and not isinstance(mode, str):
		try: # Py2
			mode = mode.encode('utf-8')
		except AttributeError: # Py3
			mode = mode.decode('utf-8')

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

	figs = []

	all_vars = splitcoord(maps.columns)
	scalar_vars = {'diffusivity': 'D', 'potential': 'V'}
	scalar_vars = [ (v, scalar_vars.get(v, None)) for v in all_vars if len(all_vars[v]) == 1 ]

	for col, short_name in scalar_vars:

		col_kwargs = {}
		for a in kwargs:
			if isinstance(kwargs[a], (dict, pd.DataFrame)) and col in kwargs[a]:
				col_kwargs[a] = kwargs[a][col]
			else:
				col_kwargs[a] = kwargs[a]

		fig = plt.figure(figsize=figsize)
		figs.append(fig)

		scalar_map_2d(cells, _clip(maps[col], clip), aspect=aspect, **col_kwargs)

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
			if isinstance(kwargs[a], (dict, pd.DataFrame)) and name in kwargs[a]:
				var_kwargs[a] = kwargs[a][name]
			else:
				var_kwargs[a] = kwargs[a]
		
		fig = plt.figure(figsize=figsize)
		figs.append(fig)

		field_map_2d(cells, _clip(maps[cols], clip), aspect=aspect, **var_kwargs)

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
		plt.show()
	else:
		for fig in figs:
			plt.close(fig)


def _clip(m, q):
	if q:
		amplitude = m.pow(2)
		if 1 < len(m.shape):
			amplitude = amplitude.sum(1)
		amplitude = amplitude.apply(np.sqrt)
		amax = amplitude.quantile(q)
		m = m.copy()
		factor = amplitude[amplitude > amax].rdiv(amax)
		if 1 < len(m.shape):
			m.loc[amplitude > amax, :] *= factor
		else:
			m.loc[amplitude > amax] *= factor
	return m

