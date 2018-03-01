# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import numpy as np
import pandas as pd


def random_walk(diffusivity=None, force=None, \
		trajectory_mean_count=100, trajectory_count_sd=3, turnover=.1, \
		box=(0., 0., 1., 1.), duration=10., time_step=.05, \
		full=False, count_outside_trajectories=False):
	"""
	Generate random walks.

	Arguments:

		diffusivity (callable): takes a coordinate vector (array-like) and time (float)
			and returns the local diffusivity (float) in um2.s-1

		force (callable): takes a coordinate vector (array-like) and time (float) and
			returns the local force (:class:`~numpy.ndarray`) in um-1.s(?)

		trajectory_mean_count (int or float): average number of active trajectories at any time

		trajectory_count_sd (int or float): standard deviation of the number of active trajectories

		turnover (float): fraction of trajectories that will end at each time step

		box (array-like): origin and size of the bounding box

		duration (float): duration of the simulation in seconds

		time_step (float): duration between consecutive time steps

		full (bool): include locations that are outside the bounding box

		count_outside_trajectories (bool): include trajectories that have left the bounding box
			in `trajectory_mean_count` and `trajectory_count_sd` counts; setting this
			argument to ``False`` artificially increases the turnover rate (default)
	
	Returns:

		pandas.DataFrame: simulated trajectories with 'n' the trajectory index,
			't' the time and with other columns for location coordinates
	"""
	_box = np.asarray(box)
	dim = int(_box.size / 2)
	support_lower_bound = _box[:dim]
	support_size = _box[dim:]
	support_upper_bound = support_lower_bound + support_size
	# default maps
	def null_scalar_map(xy, t):
		return 0.
	def null_vector_map(xy, t):
		return np.zeros((dim,))
	if diffusivity is None:
		diffusivity = null_scalar_map
	if force is None:
		force = null_vector_map
	# 
	N = int(round(float(duration) / time_step))
	K = np.round(np.random.randn(N) * trajectory_count_sd + trajectory_mean_count)
	T = np.arange(time_step, duration + time_step, time_step)
	if K[1] < K[0]: # switch so that K[0] <= K[1]
		tmp = K[0]
		K[0] = K[1]
		K[1] = tmp
	if K[-2] < K[-1]: # switch so that K[end-1] >= K[end]
		tmp = K[-1]
		K[-1] = K[-2]
		K[-2] = tmp
	xs = []
	X = np.array([])
	knew = n0 = 0
	for t, k in zip(T, K):
		k = int(k)
		if t == duration:
			kupdate = k
		elif X.size:
			kupdate = max(knew, int(round(min(k, X.shape[0]) * (1. - turnover))))
		else:
			kupdate = 0
		knew = k - kupdate
		if X.size:
			X, n = X[:kupdate], n[:kupdate]
			D = np.array([ diffusivity(x, t) for x in X ])
			if not np.all(0 < D):
				raise ValueError('non-positive diffusivity value')
			F = np.array([ force(x, t) for x in X ])
			dX = time_step * (D[:,np.newaxis] * F + \
				np.sqrt(2. * D.reshape(D.size, 1)) * np.random.randn(*X.shape))
			X += dX
			if not count_outside_trajectories:
				inside = np.all(np.logical_and(
						support_lower_bound <= X, X <= support_upper_bound
					), axis=1)
				X = X[inside]
				n = n[inside]
				kdiscarded = np.count_nonzero(~inside)
				knew += kdiscarded
		if knew == 0:
			assert 0 < X.size
			Xnew = np.zeros((knew, dim), dtype=X.dtype)
			nnew = np.zeros((knew, ), dtype=n.dtype)
		else:
			Xnew = np.random.rand(knew, dim) * support_size + support_lower_bound
			nnew = np.arange(n0, n0 + knew)
			n0 += knew
		if X.size:
			X = np.concatenate((Xnew, X))
			n = np.concatenate((nnew, n))
		else:
			X = Xnew
			n = nnew
		if np.unique(n).size < n.size:
			print((n, kupdate, knew, t))
			raise RuntimeError
		xs.append(np.concatenate((n.reshape(n.size, 1), X, np.full((n.size, 1), t)), axis=1))
	columns = 'xyz'
	if dim <= 3:
		columns = [ d for d in columns[:dim] ]
	else:
		columns = [ 'x'+str(i) for i in range(dim) ]
	columns = ['n'] + columns + ['t']
	data = np.concatenate(xs, axis=0)
	data = data[np.lexsort((data[:,-1], data[:,0]))]
	points = pd.DataFrame(data=data, columns=columns)
	if not full:
		points = crop(points, _box)
	return points


def crop(points, box):
	"""
	Remove the locations outside a bounding box.

	When a location is discarded, the corresponding trajectory is splitted into two distinct
	trajectories.

	Arguments:

		locations (pandas.DataFrame): locations with trajectory indices in column 'n',
			times in column 't' and coordinates in the other columns

		box (array-like): origin and size of the bounding box

	Returns:

		pandas.DataFrame: filtered locations
	"""
	box = np.asarray(box)
	dim = int(box.size / 2)
	support_lower_bound = box[:dim]
	support_size = box[dim:]
	support_upper_bound = support_lower_bound + support_size
	coord_cols = [ c for c in points.columns if c not in ['n', 't'] ]
	within = np.all(np.logical_and(support_lower_bound <= points[coord_cols].values,
		points[coord_cols].values <= support_upper_bound), axis=1)
	points = points.copy()
	points['n'] += np.cumsum(np.logical_not(within), dtype=points.index.dtype)
	single_point = 0 < points['n'].diff().values[1:]
	single_point[:-1] = np.logical_and(single_point[:-1], single_point[1:])
	ok = np.r_[True, np.logical_not(single_point)]
	points = points.iloc[ok]
	within = within[ok]
	points['n'] -= (points['n'].diff() - 1).clip_lower(0).cumsum()
	points = points.iloc[within]
	points.index = np.arange(points.shape[0])
	return points


def truth(cells, t=None, diffusivity=None, force=None):
	"""
	Generate maps for the true diffusivity/force distribution.

	Arguments:

		cells (Distributed): distributed data ready for inference

		t (float): time as understood by `diffusivity` and `force`

		diffusivity (callable): admits the coordinates of a single location (array-like)
			and time (float, if `t` is defined) and returns the local diffusivity (float)

		force (callable): admits the coordinates of a single location (array-like)
			and time (float, if `t` is defined) and returns the local force (array-like)

	Returns:

		pandas.DataFrame: diffusivity/force maps
	"""
	dim = cells.cells[0].center.size
	I, DF = [], []
	for i in cells.cells:
		cell = cells.cells[i]
		I.append(i)
		if diffusivity is None:
			D = []
		elif t is None:
			D = [diffusivity(cell.center)]
		else:
			D = [diffusivity(cell.center, t)]
		if force is None:
			F = []
		elif t is None:
			F = force(cell.center)
		else:
			F = force(cell.center, t)
		DF.append(np.concatenate((D, F)))
	DF = np.vstack(DF)
	if diffusivity is None:
		columns = []
	else:
		columns = [ 'diffusivity' ]
	if force is not None:
		columns += [ 'force x' + str(col+1) for col in range(dim) ]
	return pd.DataFrame(index=I, data=DF, columns = columns)


