# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.io import *
from tramway.core import lightcopy
from tramway.inference import *
from tramway.plot.map import *
from tramway.helper.tesselation import *
import matplotlib.pyplot as plt
from warnings import warn
import os
from time import time


sub_extensions = dict([(ext.upper(), ext) for ext in ['d', 'df', 'dd', 'dv', 'dx']])


def infer(cells, mode='D', output_file=None, imt_selectors={}, verbose=False, \
	localization_error=None, priorD=None, priorV=None, jeffreys_prior=None, \
	max_cell_count=20, dilation=1, worker_count=None, positive_diffusivity=True, \
	store_distributed=False, **kwargs):

	input_file = None
	if isinstance(cells, str):
		input_file, cells = find_imt(cells, **imt_selectors)
		if verbose:
			print('loading file: {}'.format(input_file))
	if cells is None:
		raise ValueError('no cells found')

	# prepare the data for the inference
	detailled_map = distributed(cells)

	if mode == 'DD' or mode == 'DV':
		multiscale_map = detailled_map.group(max_cell_count=max_cell_count, \
			adjacency_margin=dilation)
		_map = multiscale_map
	else:
		_map = detailled_map

	runtime = time()

	if mode == 'D':

		# infer diffusivity (D mode)
		diffusivity = _map.run(inferD, \
			localization_error=localization_error, jeffreys_prior=jeffreys_prior, \
			positive_diffusivity=positive_diffusivity, **kwargs)
		x = diffusivity

	elif mode == 'DF':
		
		# infer diffusivity and force (DF mode)
		df = _map.run(inferDF, \
			localization_error=localization_error, jeffreys_prior=jeffreys_prior, \
			positive_diffusivity=positive_diffusivity, **kwargs)
		x = df

	elif mode == 'DD':

		dd = _map.run(inferDD, \
			localization_error, priorD, jeffreys_prior, \
			positive_diffusivity=positive_diffusivity, worker_count=worker_count, \
			**kwargs)
		x = dd

	elif mode == 'DV':

		dv = _map.run(inferDV, \
			localization_error, priorD, priorV, jeffreys_prior, \
			positive_diffusivity=positive_diffusivity, worker_count=worker_count, \
			**kwargs)
		x = dv

	else:
		raise ValueError('unknown ''{}'' mode'.format(mode))

	runtime = time() - runtime
	if verbose:
		print('{} mode: elapsed time: {}ms'.format(mode, int(round(runtime*1e3))))

	if input_file and not output_file:
		output_file = os.path.splitext(input_file)[0]
		_file, subext = os.path.splitext(output_file)
		if subext == '.imt':
			output_file = _file
		output_file = '{}.{}.rwa'.format(output_file, \
			sub_extensions[mode])

	if output_file:
		# store the result
		if verbose:
			print('writing file: {}'.format(output_file))
		try:
			store = HDF5Store(output_file, 'w')
			store.poke('mode', mode)
			store.poke(mode, x)
			store.poke('positive_diffusivity', positive_diffusivity)
			if localization_error is not None:
				store.poke('localization_error', localization_error)
			if priorD is not None:
				store.poke('priorD', priorD)
			if priorV is not None:
				store.poke('priorV', priorV)
			if jeffreys_prior is not None:
				store.poke('jeffreys_prior', jeffreys_prior)
			if kwargs:
				store.poke('extra_args', kwargs)
			if store_distributed:
				store.poke('distributed_translocations', lightcopy(_map))
			elif input_file:
				store.poke('imt_file', input_file)
			else:
				store.poke('tesselation_param', cells.param)
			store.poke('version', 1.0)
			store.poke('runtime', runtime)
			store.close()
		except:
			warn('HDF5 libraries may not be installed', ImportWarning)

	if input_file:
		return (cells, mode, x)
	else:
		return x


def map_plot(maps, output_file=None, fig_format=None, \
	show=False, verbose=False, figsize=(24.0, 18.0), dpi=None):

	if isinstance(maps, tuple):
		cells, mode, maps = maps
		input_file = None
	else:
		input_file = maps
		try:
			hdf = HDF5Store(input_file, 'r')
			mode = hdf.peek('mode')
			maps = hdf.peek(mode)
			try:
				cells = hdf.peek('imt_file')
				_, cells = find_imt(cells)
			except KeyError:
				cells = hdf.peek('distributed_translocations')
			hdf.close()
		except ImportError:
			warn('HDF5 libraries may not be installed', ImportWarning)

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


	if 'diffusivity' in maps:
		fig0 = plt.figure(figsize=figsize)

		scalar_map_2d(cells, maps['diffusivity'])
		plt.title('D ({} mode)'.format(mode))

		if print_figs:
			if maps.shape[1] == 1:
				diffusivity_file = '{}.{}'.format(filename, figext)
			else:
				diffusivity_file = '{}_{}.{}'.format(filename, 'd', figext)
			if verbose:
				print('writing file: {}'.format(diffusivity_file))
			fig0.savefig(diffusivity_file, dpi=dpi)


	if 'potential' in maps:
		fig1 = plt.figure(figsize=figsize)

		scalar_map_2d(cells, maps['potential'])
		plt.title('V ({} mode)'.format(mode))

		if print_figs:
			if maps.shape[1] == 1:
				potential_file = '{}.{}'.format(filename, figext)
			else:
				potential_file = '{}_{}.{}'.format(filename, 'v', figext)
			if verbose:
				print('writing file: {}'.format(potential_file))
			fig1.savefig(potential_file, dpi=dpi)


	cols = [ col for col in maps.columns if col.startswith('force') ]
	if cols: # 'force'
		fig2 = plt.figure(figsize=figsize)

		field_map_2d(cells, maps[cols])
		plt.title('F ({} mode)'.format(mode))

		if print_figs:
			if maps.shape[1] == 1:
				force_file = '{}.{}'.format(filename, figext)
			else:
				force_file = '{}_{}.{}'.format(filename, 'f', figext)
			if verbose:
				print('writing file: {}'.format(force_file))
			fig2.savefig(force_file, dpi=dpi)


	if show or not print_figs:
		plt.show()


