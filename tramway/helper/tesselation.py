# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import os
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from ..core import *
from ..tesselation import *
from ..tesselation.plugins import *
from ..spatial.scaler import *
from ..spatial.translocation import *
from ..plot.mesh import *
from ..io import *
from .analysis import *
from warnings import warn
import six
import traceback


hdf_extensions = ['.rwa', '.h5', '.hdf', '.hdf5']
fig_formats = ['png', 'pdf', 'ps', 'eps', 'svg']
sub_extensions = dict([ (a, a) for a in ['imt', 'vor', 'hpc', 'hcd', 'hpd'] ])


class UseCaseWarning(UserWarning):
	pass
class IgnoredInputWarning(UserWarning):
	pass


def tesselate(xyt_data, method='gwr', output_file=None, verbose=False, \
	scaling=False, time_scale=None, \
	knn=None, distance=None, ref_distance=None, \
	rel_min_distance=0.8, rel_avg_distance=2.0, rel_max_distance=None, \
	min_location_count=20, avg_location_count=None, max_location_count=None, \
	strict_min_location_count=None, \
	compress=False, label=None, output_label=None, comment=None, input_label=None, \
	**kwargs):
	"""
	Tesselation from points series and partitioning.

	This helper routine is a high-level interface to the various tesselation techniques 
	implemented in TRamWAy.

	Arguments:
		xyt_data (str or matrix):
			Path to a *.trxyt* or *.rwa* file or raw data in the shape of 
			:class:`pandas.DataFrame` 
			(or any data format documented in :mod:`tramway.spatial.descriptor`).


		method (str):
			Tesselation method or plugin name.
			See also 
			:class:`~tramway.tesselation.RegularMesh` (``'grid'``), 
			:class:`~tramway.tesselation.KDTreeMesh` (``'kdtree'``), 
			:class:`~tramway.tesselation.KMeansMesh` (``'kmeans'``) and 
			:class:`~tramway.tesselation.GasMesh` (``'gas'``, ``'gng'`` or ``'gwr'``).

		output_file (str):
			Path to a *.rwa* file. The resulting tesselation and data partition will be 
			stored in this file. If `xyt_data` is a path to a file and `output_file` is not 
			defined, then `output_file` will be adapted from `xyt_data` with extension 
			*.rwa* and possibly overwrite the input file.

		verbose (bool or int): Verbose output.

		scaling (bool or str):
			Normalization of the data.
			Any of 'unitrange', 'whiten' or other methods defined in 
			:mod:`tramway.spatial.scaler`.

		time_scale (bool or float): 
			If this argument is defined and intepretable as ``True``, the time axis is 
			scaled by this factor and used as a space variable for the tesselation (2D+T or 
			3D+T, for example).
			This is equivalent to manually scaling the ``t`` column and passing 
			``scaling=True``.

		knn (int or pair of ints):
			After growing the tesselation, a minimum and maximum numbers of nearest 
			neighbors of each cell center can be used instead of the entire cell 
			population. Let's denote ``min_nn, max_nn = knn``. Any of ``min_nn`` and 
			``max_nn`` can be ``None``.
			If a single `int` is supplied instead of a pair, then `knn` becomes ``min_nn``.
			``min_nn`` enables cell overlap and any point may be associated with several
			cells.

		distance/ref_distance (float):
			Supposed to be the average translocation distance. Can be modified so that the 
			cells are smaller or larger.

		rel_min_distance (float):
			Multiplies with `ref_distance` to define the minimum inter-cell distance.

		rel_avg_distance (float):
			Multiplies with `ref_distance` to define an upper on the average inter-cell 
			distance.

		rel_max_distance (float):
			Multiplies with `ref_distance` to define the maximum inter-cell distance.

		min_location_count (int):
			Minimum number of points per cell. Depending on the method, can be strictly
			enforced or regarded as a recommendation.

		avg_location_count (int):
			Average number of points per cell. For non-plugin method, per default, it is
			set to four times `min_location_count`.

		max_location_count (int):
			Maximum number of points per cell. This is used only by *kdtree*.

		input_label (str):
			Label for the input tesselation for nesting tesselations.

		label/output_label (int or str):
			Label for the resulting analysis instance.

		comment (str):
			Description message for the resulting analysis.

	Returns:
		tramway.tesselation.CellStats: A partition of the data with 
			:attr:`~tramway.tesselation.CellStats.tesselation` attribute set.


	Apart from the parameters defined above, extra input arguments are admitted and passed to the
	initializer of the selected tesselation method. See the individual documentation of these 
	methods for more information.

	"""
	no_nesting_error = ValueError('nesting tesselations does not apply to translocation data')
	if isinstance(xyt_data, six.string_types) or isinstance(xyt_data, (tuple, list, frozenset, set)):
		xyt_path = list_rwa(xyt_data)
		if xyt_path:
			if xyt_path[1:]:
				raise ValueError('too many files match')
			analyses = find_analysis(xyt_path[0])
			if input_label:
				input_partition, = find_artefacts(analyses, CellStats, input_label)
				raise NotImplementedError
			else:
				xyt_data = analyses.data
		else:
			xyt_data, xyt_path = load_xyt(xyt_data, return_paths=True, verbose=verbose)
			analyses = Analyses(xyt_data)
			if input_label is not None:
				raise no_nesting_error
	else:
		xyt_path = []
		if isinstance(xyt_data, Analyses):
			analyses = xyt_data
			xyt_data = analyses.data
		else:
			analyses = Analyses(xyt_data)
		#warn('TODO: test direct data input', UseCaseWarning)
		if input_label is not None:
			raise no_nesting_error
	input_files = xyt_path

	try:
		setup, module = all_plugins[method]
		constructor = getattr(module, setup['make'])
	except KeyError: # former code
		plugin = False
		methods = dict(grid=RegularMesh, kdtree=KDTreeMesh, kmeans=KMeansMesh, gwr=GasMesh)
		constructor = methods[method]
	else:
		plugin = True
	
	if distance:
		ref_distance = distance
	if ref_distance:
		transloc_length = None
	else:
		transloc_xy = np.asarray(translocations(xyt_data))
		if transloc_xy.shape[0] == 0:
			raise ValueError('no translocation found')
		transloc_length = np.nanmean(np.sqrt(np.sum(transloc_xy * transloc_xy, axis=1)))
		if verbose:
			print('average translocation distance: {}'.format(transloc_length))
		ref_distance = transloc_length
	min_distance = rel_min_distance * ref_distance
	avg_distance = rel_avg_distance * ref_distance
	if rel_max_distance:
		# applies only to KDTreeMesh
		max_distance = rel_max_distance * ref_distance
		if method != 'kdtree':
			warn('`rel_max_distance` is relevant only with `kdtree`', IgnoredInputWarning)
	else:
		max_distance = None

	if not scaling:
		scaling = 'none'
	elif scaling is True:
		scaling = 'whiten'
	scalers = dict(none=Scaler, whiten=whiten, unit=unitrange)
	scaler = scalers[scaling]()

	n_pts = float(xyt_data.shape[0])
	if min_location_count:
		min_probability = float(min_location_count) / n_pts
	else:
		min_probability = None
		if not plugin:
			warn('undefined `min_location_count`; not tested', UseCaseWarning)
	if not plugin and not avg_location_count:
		avg_location_count = 4 * min_location_count
	if avg_location_count:
		avg_probability = float(avg_location_count) / n_pts
	else:
		avg_probability = None
		if not plugin:
			warn('undefined `avg_location_count`; not tested', UseCaseWarning)
	if max_location_count:
		# applies only to KDTreeMesh
		max_probability = float(max_location_count) / n_pts
		if not plugin and method != 'kdtree':
			warn('`max_location_count` is relevant only with `kdtree`', IgnoredInputWarning)
	else:
		max_probability = None

	colnames = ['x', 'y']
	if 'z' in xyt_data:
		colnames.append('z')
	if time_scale:
		colnames.append('t')
		scaler.factor = [('t', time_scale)]

	# initialize a Tesselation object
	params = dict( \
		min_distance=min_distance, \
		avg_distance=avg_distance, \
		max_distance=max_distance, \
		min_probability=min_probability, \
		avg_probability=avg_probability, \
		max_probability=max_probability, \
		)
	if plugin:
		for ignored in ['max_level']:
			try:
				if kwargs[ignored] is None:
					del kwargs[ignored]
			except KeyError:
				pass
		params.update(dict( \
			min_location_count=min_location_count, \
			avg_location_count=avg_location_count, \
			max_location_count=max_location_count, \
			))
		params.update(kwargs)
		for key in setup.get('make_arguments', {}):
			try:
				param = params[key]
			except KeyError:
				pass
			else:
				kwargs[key] = param
	else:
		params.update(kwargs)
		kwargs = params
	tess = constructor(scaler, **kwargs)

	# grow the tesselation
	tess.tesselate(xyt_data[colnames], verbose=verbose, **kwargs)

	# partition the dataset into the cells of the tesselation
	if knn is None:
		cell_index = tess.cell_index(xyt_data, min_location_count=strict_min_location_count)
	else:
		if strict_min_location_count is None:
			strict_min_location_count = min_location_count
		cell_index = tess.cell_index(xyt_data, knn=knn, \
			min_location_count=strict_min_location_count, \
			metric='euclidean')

	stats = CellStats(cell_index, points=xyt_data, tesselation=tess)

	stats.param['method'] = method
	if transloc_length:
		stats.param['transloc_length'] = transloc_length
	else:
		stats.param['ref_distance'] = ref_distance
	if min_distance:
		stats.param['min_distance'] = min_distance
	if avg_distance:
		stats.param['avg_distance'] = avg_distance
	if max_distance:
		stats.param['max_distance'] = max_distance
	if not plugin:
		if min_location_count:
			stats.param['min_location_count'] = min_location_count
		if avg_location_count:
			stats.param['avg_location_count'] = avg_location_count
		if max_location_count:
			stats.param['max_location_count'] = min_location_count
	if knn:
		stats.param['knn'] = knn
	#if spatial_overlap: # deprecated
	#	stats.param['spatial_overlap'] = spatial_overlap
	if plugin:
		stats.param.update(kwargs)
	elif method == 'kdtree' and 'max_level' in kwargs:
		stats.param['max_level'] = kwargs['max_level']

	if label is None:
		label = output_label
	analyses.add(Analyses(stats), label=label, comment=comment)

	# save `analyses`
	if output_file or xyt_path:
		if output_file is None:
			output_file = os.path.splitext(xyt_path[0])[0] + hdf_extensions[0]
		if compress:
			analyses_ = lightcopy(analyses)
		else:
			analyses_ = analyses

		save_rwa(output_file, analyses, verbose, \
			force=len(input_files)==1 and input_files[0]==output_file)

	return stats




def cell_plot(cells, xy_layer=None, output_file=None, fig_format=None, \
	show=False, verbose=False, figsize=(24.0, 18.0), dpi=None, \
	location_count_hist=False, cell_dist_hist=False, location_dist_hist=False, \
	aspect=None, delaunay=None, locations={}, voronoi=True, colors=None, title=None, \
	label=None, input_label=None):
	"""
	Partition plots.

	Plots a spatial representation of the tesselation and partition if data are 2D, and optionally
	histograms.

	Arguments:
		cells (str or CellStats):
			Path to a *.imt.rwa* file or :class:`~tramway.tesselation.CellStats` 
			instance.

		xy_layer ({None, 'delaunay', 'voronoi'}):
			Overlay Delaunay or Voronoi graph over the data points. For 2D data only.
			*Deprecated!* Please use `delaunay` and `voronoi` arguments instead.

		output_file (str):
			Path to a file in which the figure will be saved. If `cells` is a path and 
			`fig_format` is defined, `output_file` is automatically set.

		fig_format (str):
			Any image format supported by :func:`matplotlib.pyplot.savefig`.

		show (bool):
			Makes `cell_plot` show the figure(s) which is the default behavior if and only 
			if the figures are not saved.

		verbose (bool): Verbose output.

		figsize (pair of floats):
			Passed to :func:`matplotlib.pyplot.figure`. Applies only to the spatial 
			representation figure.

		dpi (int):
			Passed to :func:`matplotlib.pyplot.savefig`. Applies only to the spatial 
			representation figure.

		location_count_hist (bool):
			Plot a histogram of point counts (per cell). If the figure is saved, the 
			corresponding file will have sub-extension *.hpc*.

		cell_dist_hist (bool):
			Plot a histogram of distances between neighbor centroids. If the figure is 
			saved, the corresponding file will have sub-extension *.hcd*.

		location_dist_hist (bool):
			Plot a histogram of distances between points from neighbor cells. If the figure 
			is saved, the corresponding file will have sub-extension *.hpd*.

		aspect (str):
			Aspect ratio. Can be ``'equal'``.

		locations (dict):
			Keyword arguments to :func:`~tramway.plot.mesh.plot_points`.

		delaunay (bool or dict):
			Overlay Delaunay graph. If :class:`dict`, keyword arguments to 
			:func:`~tramway.plot.mesh.plot_delaunay`.

		voronoi (bool or dict):
			Overlay Voronoi graph. If :class:`dict`, keyword arguments to 
			:func:`~tramway.plot.mesh.plot_voronoi`.

		label/input_label (int or str or list):
			If `cells` is a filepath, label of the analysis instance in the file.

	Notes:
		See also :mod:`tramway.plot.mesh`.

	"""
	if isinstance(cells, CellStats):
		input_file = ''
	else:
		input_file = cells
		if label is None:
			labels = input_label
		else:
			labels, label = label, None
		try:
			analyses = find_analysis(input_file, labels=labels)
		except KeyError as e:
			if e.args and 'analyses' not in e.args[0]:
				raise
			# legacy code
			if isinstance(input_file, list):
				input_file = input_file[0]
			imt_path = input_file
			if imt_path is None:
				raise ValueError('undefined input file')
			# copy-paste
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
			if auto_select and verbose:
				print('selecting {} as a tesselation file'.format(imt_path))

			# load the data
			input_file = imt_path
			try:
				hdf = HDF5Store(input_file, 'r')
				cells = hdf.peek('cells')
			except:
				print(traceback.format_exc())
				warn('HDF5 libraries may not be installed', ImportWarning)
			finally:
				hdf.close()
		else:
			if isinstance(analyses, dict):
				if len(analyses) == 1:
					analyses = list(analyses.values())[0]
				else:
					raise ValueError('multiple files match')
			if not labels:
				labels = list(analyses.labels)
			if labels[1:]:
				raise ValueError('multiple instances; label is required')
			label = labels[-1]
			cells = analyses[label].data

	# guess back some input parameters
	method_name = {RegularMesh: ('grid', 'grid', 'regular grid'), \
		KDTreeMesh: ('kdtree', 'k-d tree', 'k-d tree based tesselation'), \
		KMeansMesh: ('kmeans', 'k-means', 'k-means based tesselation'), \
		GasMesh: ('gwr', 'GWR', 'GWR based tesselation')}
	try:
		method_name, pp_method_name, method_title = method_name[type(cells.tesselation)]
	except KeyError:
		method_name = pp_method_name = method_title = ''
	min_distance = cells.param.get('min_distance', 0)
	avg_distance = cells.param.get('avg_distance', None)
	min_location_count = cells.param.get('min_location_count', 0)

	# plot the data points together with the tesselation
	figs = []
	dim = cells.tesselation.cell_centers.shape[1]
	if dim == 2:
		fig = plt.figure(figsize=figsize)
		figs.append(fig)
		if 'knn' in cells.param: # if knn <= min_count, min_count is actually ignored
			plot_points(cells, **locations)
		else:
			plot_points(cells, min_count=min_location_count, **locations)
		if aspect is not None:
			fig.gca().set_aspect(aspect)
		if xy_layer == 'voronoi' or voronoi:
			if not isinstance(voronoi, dict):
				voronoi = {}
			plot_voronoi(cells, **voronoi)
			voronoi = True
		if xy_layer == 'delaunay' or delaunay: # Delaunay above Voronoi
			if not isinstance(delaunay, dict):
				delaunay = {}
			plot_delaunay(cells, **delaunay)
			delaunay = True
		if title:
			if isinstance(title, str):
				plt.title(title)
			elif delaunay == voronoi:
				plt.title(pp_method_name)
			elif delaunay:
				plt.title(pp_method_name + ' based Delaunay')
			elif voronoi:
				plt.title(pp_method_name + ' based Voronoi')


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
		subname, subext = os.path.splitext(filename)
		if subext and subext[1:] in sub_extensions.values():
			filename = subname
		if dim == 2:
			vor_file = '{}.{}.{}'.format(filename, sub_extensions['vor'], figext)
			if verbose:
				print('writing file: {}'.format(vor_file))
			fig.savefig(vor_file, dpi=dpi)


	# the complementary histograms below haven't been tested for a while [TODO]

	if location_count_hist:
		# plot a histogram of the number of points per cell
		fig = plt.figure()
		figs.append(fig)
		plt.hist(cells.location_count, bins=np.arange(0, min_location_count*20, min_location_count))
		plt.plot((min_location_count, min_location_count), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('point count (per cell)')
		if print_figs:
			hpc_file = '{}.{}.{}'.format(filename, 'hpc', figext)
			if verbose:
				print('writing file: {}'.format(hpc_file))
			fig.savefig(hpc_file)

	if cell_dist_hist:
		# plot a histogram of the distance between adjacent cell centers
		A = sparse.triu(cells.tesselation.cell_adjacency, format='coo')
		i, j, k = A.row, A.col, A.data
		label = cells.tesselation.adjacency_label
		if label is not None:
			i = i[0 < label[k]]
			j = j[0 < label[k]]
		pts = cells.tesselation.cell_centers
		dist = la.norm(pts[i,:] - pts[j,:], axis=1)
		fig = plt.figure()
		figs.append(fig)
		plt.hist(np.log(dist), bins=50)
		if avg_distance:
			dmin = np.log(min_distance)
			dmax = np.log(avg_distance)
			plt.plot((dmin, dmin), plt.ylim(), 'r-')
			plt.plot((dmax, dmax), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('inter-centroid distance (log)')
		if print_figs:
			hcd_file = '{}.{}.{}'.format(filename, sub_extensions['hcd'], figext)
			if verbose:
				print('writing file: {}'.format(hcd_file))
			fig.savefig(hcd_file)

	if location_dist_hist:
		adj = point_adjacency_matrix(cells, symetric=False)
		dist = adj.data
		fig = plt.figure()
		figs.append(fig)
		plt.hist(np.log(dist), bins=100)
		if avg_distance:
			dmin = np.log(min_distance)
			dmax = np.log(avg_distance)
			plt.plot((dmin, dmin), plt.ylim(), 'r-')
			plt.plot((dmax, dmax), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('inter-point distance (log)')
		if print_figs:
			hpd_file = '{}.{}.{}'.format(filename, sub_extensions['hpd'], figext)
			if verbose:
				print('writing file: {}'.format(hpd_file))
			fig.savefig(hpd_file)

	if show or not print_figs:
		plt.show()
	else:
		for fig in figs:
			plt.close(fig)



def find_mesh(path, method=None, full_list=False):
	"""
	*from version 0.3:* deprecated.
	"""
	if not isinstance(path, (tuple, list)):
		path = (path,)
	paths = []
	for p in path:
		if os.path.isdir(p):
			paths.append([ os.path.join(p, f) for f in os.listdir(p) if f.endswith('.rwa') ])
		else:
			if p.endswith('.rwa'):
				ps = [p]
			else:
				d, p = os.path.split(p)
				p, _ = os.path.splitext(p)
				if d:
					ps = [ os.path.join(d, f) for f in os.listdir(d) \
						if f.startswith(p) and f.endswith('.rwa') ]
				else:
					ps = [ f for f in os.listdir('.') \
						if f.startswith(p) and f.endswith('.rwa') ]
			paths.append(ps)
	paths = list(itertools.chain(*paths))
	found = False
	for path in paths:
		try:
			hdf = HDF5Store(path, 'r')
			try:
				cells = hdf.peek('cells')
				if isinstance(cells, CellStats) and \
					(method is None or cells.param['method'] == method):
					found = True
			except EnvironmentError:
				print(traceback.format_exc())
				warn('HDF5 libraries may not be installed', ImportWarning)
			finally:
				try:
					hdf.close()
				except:
					pass
		except:
			print(traceback.format_exc())
			pass
		if found: break
	if found:
		if full_list:
			path = paths
		return (path, cells)
	else:
		return (paths, None)


def find_imt(path, method=None, full_list=False):
	"""
	Alias for :func:`find_mesh` for backward compatibility.

	*from version 0.3:* deprecated.
	"""
	return find_mesh(path, method, full_list)

def find_partition(path, method=None, full_list=False):
	"""
	Alias for :func:`find_mesh`.

	*from version 0.3:* deprecated.
	"""
	return find_mesh(path, method, full_list)


