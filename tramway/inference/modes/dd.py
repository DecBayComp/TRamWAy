# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..base import *
from warnings import warn
from math import pi, log
import numpy as np
import pandas as pd
from ..diffusivity import DiffusivityWarning
from scipy.optimize import minimize
from collections import OrderedDict


setup = {'arguments': OrderedDict((
		('localization_error',	('-e', dict(type=float, default=0.01, help='localization error'))),
		('diffusivity_prior',	('-d', dict(type=float, default=0.01, help='prior on the diffusivity'))),
		('jeffreys_prior',	('-j', dict(action='store_true', help="Jeffreys' prior"))),
		('min_diffusivity',	dict(type=float, default=0, help='minimum diffusivity value allowed')))),
		'cell_sampling': 'group'}


def dd_neg_posterior(diffusivity, cells, square_localization_error, diffusivity_prior, jeffreys_prior, dt_mean, \
	index_map=None, min_diffusivity=0):
	observed_min = np.min(diffusivity)
	if observed_min < min_diffusivity and not np.isclose(observed_min, min_diffusivity):
		warn(DiffusivityWarning(observed_min, min_diffusivity))
	noise_dt = square_localization_error
	j = 0
	result = 0.0
	for i in cells.cells:
		cell = cells.cells[i]
		n = cell.tcount
		if n == 0:
			continue
		# posterior calculations
		if cell.cache['dxy2'] is None:
			cell.cache['dxy2'] = np.sum(cell.dxy * cell.dxy, axis=1) # dx**2 + dy**2 + ..
		diffusivity_dt = 4.0 * (diffusivity[j] * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
		result += n * log(pi) + np.sum(np.log(diffusivity_dt)) # sum(log(4*pi*Dtot*dt))
		result += np.sum(cell.cache['dxy2'] / diffusivity_dt) # sum((dx**2+dy**2+..)/(4*Dtot*dt))
		# prior
		if diffusivity_prior:
			area = cells.grad_sum(i, index_map)
			# gradient of diffusivity
			gradD = cells.grad(i, diffusivity, index_map)
			if gradD is None:
				#raise RuntimeError('missing gradient')
				continue
			result += diffusivity_prior * np.dot(gradD * gradD, area)
		j += 1
	if jeffreys_prior:
		result += 2.0 * np.sum( log(diffusivity * dt_mean + square_localization_error) - \
					log(diffusivity) )
	return result

def inferDD(cells, localization_error=0.0, diffusivity_prior=None, jeffreys_prior=None, \
	min_diffusivity=0, **kwargs):
	sq_loc_err = localization_error * localization_error
	initial = []
	j = 0
	for i in cells.cells:
		cell = cells.cells[i]
		if 0 < cell.tcount:
			if not np.all(0 < cell.dt):
				warn('translocation dts are non-positive', RuntimeWarning)
				cell.dt = abs(cell.dt)
			cell.cache = dict(dxy2=None, area=None) # initialize cached constants
			dt_mean_i = np.mean(cell.dt)
			initial.append((i, j, \
				dt_mean_i, \
				np.mean(cell.dxy * cell.dxy) / (2.0 * dt_mean_i)))
			j += 1
	i, j, dt_mean, initial_diffusivity = (np.array(xs) for xs in zip(*initial))
	index = np.full(cells.adjacency.shape[0], -1, dtype=int)
	index[i] = j
	if min_diffusivity is not None:
		kwargs['bounds'] = [(min_diffusivity,None)] * initial_diffusivity.size
	result = minimize(dd_neg_posterior, initial_diffusivity,
		args=(cells, sq_loc_err, diffusivity_prior, jeffreys_prior, dt_mean, index, min_diffusivity),
		**kwargs)
	return pd.DataFrame(data=result.x[:,np.newaxis], index=i, columns=['diffusivity'])

