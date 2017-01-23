
import os
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from ..tesselation import *
from ..spatial.scaler import *
from ..spatial.motion import *
from ..plot.mesh import *
from ..io import *
from warnings import warn
import six


hdf_extensions = ['.h5', '.hdf', '.hdf5']
imt_extensions = [ '.imt' + ext for ext in hdf_extensions ]
fig_formats = ['png', 'pdf', 'ps', 'eps', 'svg']
sub_extensions = dict([ (a, a) for a in ['imt', 'vor', 'hpc', 'hcd', 'hpd'] ])


class UseCaseWarning(UserWarning):
	pass
class IgnoredInputWarning(UserWarning):
	pass


def tesselate(xyt_data, method='gwr', output_file=None, verbose=False, \
	scaling=False, time_scale=None, \
	knn=None, spatial_overlap=False, \
	ref_distance=None, rel_min_distance=0.8, rel_avg_distance=2.0, rel_max_distance=None, \
	min_cell_count=20, avg_cell_count=None, max_cell_count=None, \
	**kwargs):
	"""Tesselation for clouds or series of points

	This helper routine is a high-level interface to the various tesselation techniques 
	implemented in InferenceMAP.

	Returns a :class:`inferencemap.tesselation.base.CellStats` instance. See :ref:`imt file format <imt_format>`.

	Apart from the parameters defined below, extra input arguments are admited and passed to the
	initializer of the selected tesselation method. See the individual documentation of these 
	methods for more information.

	Parameters
	----------
	xyt_data: string or matrix, required
		Path to a .trxyt file or raw data in the shape of :class:`pandas.DataFrame` 
		(or any data format documented in :mod:`inferencemap.spatial.descriptor`)

	method: string, optional, default 'gwr'
		Any of 'grid', 'kdtree', 'kmeans' or 'gwr'.
		'grid' is a regular grid (see :class:`inferencemap.tesselation.base.RegularMesh`)
		'kdtree' is a dichotomy-like k-d tree (see :class:`inferencemap.tesselation.kdtree.KDTreeMesh`)
		'kmeans' is based on k-means clustering (see :class:`inferencemap.tesselation.kmeans.KMeansMesh`)
		'gwr' is based on Growing-When-Required self-organizing "gas" (see :class:`inferencemap.tesselation.gas.GasMesh`)

	output_file: string, optional
		Path to a .h5 file. The resulting tesselation and data partition will be stored in this
		file. If xyt_data is a path to a file and output_file is not defined, then output_file
		will be adapted from xyt_data with extension .imt.h5.

	verbose: boolean, optional, default False

	scaling: boolean or string, optional
		Any of 'unitrange' or 'whiten'. See :mod:`inferencemap.spatial.scaler`.

	time_scale: boolean or float, optional
		If this argument is defined and intepretable as True, the time axis is scaled by this
		factor and used as a space variable for the tesselation (2D+T or 3D+T, for example).
		This is equivalent to manually scaling the `t` column and passing `scaling=True`.

	knn: int, optional
		After growing the tesselation, a maximum number knn of nearest neighbors of each cell
		center can be used instead of the entire cell population. See also `spatial_overlap`.

	spatial_overlap: boolean, optional, default False
		In combination with non-negative knn, allows cell overlapping when knn is greater than
		the number of points in a cell.

	ref_distance: float, optional
		Supposed to be the average jump distance. Can be modified so that the cells are smaller
		or larger.

	rel_min_distance: float, optional, default 0.8
		Multiplies with `ref_distance` to define the minimum inter-cell distance.

	rel_avg_distance: float, optional, default 2
		Multiplies with `ref_distance` to define an upper on the average inter-cell distance.

	rel_max_distance: float, optional
		Multiplies with `ref_distance` to define the maximum inter-cell distance.

	min_cell_count: int, optional, default 20
		Minimum number of points per cell. Depending on the method, can be strictly enforced or
		interpreted as a hint.

	avg_cell_count: int, optional
		Hint of the average number of points per cell. Per default set to four times
		`min_cell_count`.

	max_cell_count: int, optional
		Maximum number of points per cell. This is used only by `kdtree`.

	Notes
	-----
	See demo/glycine_receptor.py for simple applications of the different methods.

	"""
	if isinstance(xyt_data, six.string_types) or isinstance(xyt_data, list):
		xyt_data, xyt_path = load_xyt(xyt_data, return_paths=True, verbose=verbose)
	else:
		xyt_path = []
		warn('TODO: test direct data input', UseCaseWarning)
	
	if ref_distance:
		jump_length = None
	else:
		jump_xy = np.asarray(jumps(xyt_data))
		jump_length = np.nanmean(np.sqrt(np.sum(jump_xy * jump_xy, axis=1)))
		if verbose:
			print('average jump distance: {}'.format(jump_length))
		ref_distance = jump_length
	min_distance = rel_min_distance * ref_distance
	avg_distance = rel_avg_distance * ref_distance
	if rel_max_distance:
		# applies only to KDTreeMesh
		max_distance = rel_max_distance * ref_distance
		if method != 'kdtree':
			warn('`rel_max_distance` is relevant only with `kdtree`', IgnoredInputWarning)
	else:
		max_distance = None

	methods = dict(grid=RegularMesh, kdtree=KDTreeMesh, kmeans=KMeansMesh, gwr=GasMesh)
	constructor = methods[method]
	if not scaling:
		scaling = 'none'
	elif scaling is True:
		scaling = 'whiten'
	scalers = dict(none=Scaler, whiten=whiten, unit=unitrange)
	scaler = scalers[scaling]()

	n_pts = float(xyt_data.shape[0])
	if min_cell_count:
		min_probability = float(min_cell_count) / n_pts
	else:
		min_probability = None
		warn('undefined `min_cell_count`; not tested', UseCaseWarning)
	if not avg_cell_count:
		avg_cell_count = 4 * min_cell_count
	if avg_cell_count:
		avg_probability = float(avg_cell_count) / n_pts
	else:
		avg_probability = None
		warn('undefined `avg_cell_count`; not tested', UseCaseWarning)
	if max_cell_count:
		# applies only to KDTreeMesh
		max_probability = float(max_cell_count) / n_pts
		if method != 'kdtree':
			warn('`max_cell_count` is relevant only with `kdtree`', IgnoredInputWarning)
	else:
		max_probability = None

	colnames = ['x', 'y']
	if 'z' in xyt_data:
		colnames.append('z')
	if time_scale:
		colnames.append('t')
		scaler.factor = [('t', time_scale)]

	# initialize a Tesselation object
	tess = constructor(scaler, \
		min_distance=min_distance, \
		avg_distance=avg_distance, \
		max_distance=max_distance, \
		min_probability=min_probability, \
		avg_probability=avg_probability, \
		max_probability=max_probability, \
		**kwargs)

	# grow the tesselation
	tess.tesselate(xyt_data[colnames], verbose=verbose)

	# partition the dataset into the cells of the tesselation
	if spatial_overlap:
		stats = tess.cellStats(xyt_data, knn=knn)
	else:
		stats = tess.cellStats(xyt_data, knn=knn, prefered='force index')

	stats.param['method'] = method
	if jump_length:
		stats.param['jump_length'] = jump_length
	else:
		stats.param['ref_distance'] = ref_distance
	if min_distance:
		stats.param['min_distance'] = min_distance
	if avg_distance:
		stats.param['avg_distance'] = avg_distance
	if max_distance:
		stats.param['max_distance'] = max_distance
	if min_cell_count:
		stats.param['min_cell_count'] = min_cell_count
	if avg_cell_count:
		stats.param['avg_cell_count'] = avg_cell_count
	if max_cell_count:
		stats.param['max_cell_count'] = min_cell_count
	if knn:
		stats.param['knn'] = knn
	if spatial_overlap:
		stats.param['spatial_overlap'] = spatial_overlap
	if method == 'kdtree':
		if 'max_level' in kwargs:
			stats.param['max_level'] = kwargs['max_level']

	# save `stats`
	if output_file or xyt_path:
		if output_file is None:
			xyt_file, _ = os.path.splitext(xyt_path[0])
			imt_path = xyt_file + imt_extensions[0]
		else:
			imt_path, imt_ext = os.path.splitext(output_file)
			if imt_ext in hdf_extensions:
				imt_path = output_file
			else:
				imt_path += imt_extensions[0]

		try:
			store = HDF5Store(imt_path, 'w', verbose and 1 < verbose)
			if verbose:
				print('writing file: {}'.format(imt_path))
			store.poke('cells', stats)
			store.close()
		except _:
			warn('HDF5 libraries may not be installed', ImportWarning)

	return stats




def cell_plot(cells, xy_layer='voronoi', output_file=None, fig_format=None, \
	show=False, verbose=False, figsize=(24.0, 18.0), dpi=None, \
	point_count_hist=False, cell_dist_hist=False, point_dist_hist=False):
	"""Partition plots

	Plots a spatial representation of the tesselation and partition if data are 2D, and optionally
	histograms.

	Parameters
	----------
	cells: string or CellStats, required
		Path to a .imt.h5 file or CellStats instance.

	xy_layer: string, optional, default 'voronoi'
		Either 'delaunay', 'voronoi' or None. Overlay delaunay graph or voronoi over the data
		points. For 2D data only.

	output_file: string, optional
		Path to a file in which the figure will be saved. If `cells` is a path and `fig_format`
		is defined, `output_file` is automatically set.

	fig_format: string, optional
		Any image format supported by :func:`matplotlib.pyplot.savefig`.

	show: boolean, optional, default False
		Makes `cell_plot` show the figure(s) which is the default behavior if and only if the
		figures are not saved.

	verbose: boolean, optional, default False

	figsize: pair of floats, optional, default (24.0, 18.0)
		Passed to :func:`matplotlib.pyplot.figure`. Applies only to the spatial representation
		figure.

	dpi: int, optional
		Passed to :func:`matplotlib.pyplot.savefig`. Applies only to the spatial representation
		figure.

	point_count_hist: boolean, optional, default False
		Plot a histogram of point counts (per cell). If the figure is saved, the corresponding
		file will have sub-extension .hpc.

	cell_dist_hist: boolean, optional, default False
		Plot a histogram of distances between neighbor centroids. If the figure is saved, the
		corresponding file will have sub-extension .hcd.

	point_dist_hist: boolean, optional, default False
		Plot a histogram of distances between points from neighbor cells. If the figure is saved,
		the corresponding file will have sub-extension .hpd.

	Notes
	-----
	See also :mod:`inferencemap.plot.mesh`.

	"""
	if isinstance(cells, CellStats):
		input_file = ''
	else:
		input_file = cells
		if isinstance(input_file, list):
			input_file = input_file[0]
		imt_path = input_file
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
			hdf.close()
		except _:
			warn('HDF5 libraries may not be installed', ImportWarning)

	# guess back some input parameters
	method_name = {RegularMesh: ('grid', 'grid', 'regular grid'), \
		KDTreeMesh: ('kdtree', 'k-d tree', 'k-d tree based tesselation'), \
		KMeansMesh: ('kmeans', 'k-means', 'k-means based tesselation'), \
		GasMesh: ('gwr', 'GWR', 'GWR based tesselation')}
	method_name, pp_method_name, method_title = method_name[type(cells.tesselation)]
	min_distance = cells.param.get('min_distance', 0)
	avg_distance = cells.param.get('avg_distance', None)
	min_cell_count = cells.param.get('min_cell_count', 0)

	# plot the data points together with the tesselation
	dim = cells.tesselation.cell_centers.shape[1]
	if dim == 2:
		fig0 = plt.figure(figsize=figsize)
		if 'knn' in cells.param: # if knn <= min_count, min_count is actually ignored
			plot_points(cells)
		else:
			plot_points(cells, min_count=min_cell_count)
		if xy_layer == 'delaunay':
			plot_delaunay(cells)
			plt.title(pp_method_name + ' based delaunay')
		elif xy_layer == 'voronoi':
			if method_name in ['kmeans']: # scipy.spatial.Voronoi based ridge_vertices (buggy)
				plot_voronoi(cells, color='rrrr', negative=True)
			else:
				plot_voronoi(cells)
			plt.title(pp_method_name + ' based voronoi')
		else:
			plt.title(pp_method_name)


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
			fig0.savefig(vor_file, dpi=dpi)


	if point_count_hist:
		# plot a histogram of the number of points per cell
		fig1 = plt.figure()
		plt.hist(cells.cell_count, bins=np.arange(0, min_cell_count*20, min_cell_count))
		plt.plot((min_cell_count, min_cell_count), plt.ylim(), 'r-')
		plt.title(method_title)
		plt.xlabel('point count (per cell)')
		if print_figs:
			hpc_file = '{}.{}.{}'.format(filename, 'hpc', figext)
			if verbose:
				print('writing file: {}'.format(hpc_file))
			fig1.savefig(hpc_file)

	if cell_dist_hist:
		# plot a histogram of the distance between adjacent cell centers
		A = cells.tesselation.cell_adjacency.tocoo()
		i, j, k = A.row, A.col, A.data
		label = cells.tesselation.adjacency_label
		if label is not None:
			i = i[0 < label[k]]
			j = j[0 < label[k]]
		pts = cells.tesselation.cell_centers
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
		if print_figs:
			hcd_file = '{}.{}.{}'.format(filename, sub_extensions['hcd'], figext)
			if verbose:
				print('writing file: {}'.format(hcd_file))
			fig2.savefig(hcd_file)

	if point_dist_hist:
		adj = point_adjacency_matrix(cells, symetric=False)
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
		if print_figs:
			hpd_file = '{}.{}.{}'.format(filename, sub_extensions['hpd'], figext)
			if verbose:
				print('writing file: {}'.format(hpd_file))
			fig3.savefig(hpd_file)

	if show or not print_figs:
		plt.show()


