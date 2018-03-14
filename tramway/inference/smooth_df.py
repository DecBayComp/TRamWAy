# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import ChainArray
from .base import *
from warnings import warn
from math import pi, log
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import OrderedDict


setup = {'name': 'smooth.df',
	'provides': 'df',
	'arguments': OrderedDict((
		('localization_error',	('-e', dict(type=float, default=0.03, help='localization error'))),
		('diffusivity_prior',	('-d', dict(type=float, default=0.01, help='prior on the diffusivity'))),
		('jeffreys_prior',	('-j', dict(action='store_true', help="Jeffreys' prior"))),
		('min_diffusivity',	dict(type=float, default=0., help='minimum diffusivity value allowed')),
		('diffusive_drift',	dict(action='store_true', help='make the diffusive drift available')))),
		'cell_sampling': 'group'}


def smooth_df_neg_posterior(x, df, cells, squared_localization_error, diffusivity_prior, jeffreys_prior,
		dt_mean, min_diffusivity):
	# extract `D` and `F`
	df.update(x)
	D, F = df['D'], df['F']
	#
	if min_diffusivity is not None:
		observed_min = np.min(D)
		if observed_min < min_diffusivity and not np.isclose(observed_min, min_diffusivity):
			warn(DiffusivityWarning(observed_min, min_diffusivity))
	noise_dt = squared_localization_error
	# for all cell
	result = 0.
	for j, i in enumerate(cells):
		cell = cells[i]
		n = len(cell) # number of translocations
		# various posterior terms
		D_dt = D[j] * cell.dt
		Dtot = 4. * (D_dt + noise_dt) # 4*(D+Dnoise)*dt
		dxy_minus_drift_dt = cell.dxy - np.outer(D_dt, F[j])
		Ftot = np.sum(dxy_minus_drift_dt * dxy_minus_drift_dt, axis=1)
		result += n * log(pi) + np.sum(np.log(Dtot)) + np.sum(Ftot / Dtot)
		# priors
		gradD = cells.grad(i, D) # spatial gradient of the local diffusivity
		if gradD is not None:
			# `grad_sum` memoizes and can be called several times at no extra cost
			result += diffusivity_prior * cells.grad_sum(i, gradD * gradD)
	if jeffreys_prior:
		result += 2. * np.sum(np.log(D * dt_mean + noise_dt) - np.log(D))
	return result


def infer_smooth_DF(cells, localization_error=0.03, diffusivity_prior=None, jeffreys_prior=False,
		min_diffusivity=0, diffusive_drift=False, **kwargs):
	# initial values and sanity checks
	index, dt_mean, initial = [], [], []
	for i in cells:
		cell = cells[i]
		if not bool(cell):
			raise ValueError('empty cells')
		# ensure that translocations are properly oriented in time
		if not np.all(0 < cell.dt):
			warn('translocation dts are not all positive', RuntimeWarning)
			cell.dxy[cell.dt < 0] *= -1.
			cell.dt[ cell.dt < 0] *= -1.
		# initialize the local diffusivity parameter
		dt_mean_i = np.mean(cell.dt)
		initial_i = np.mean(cell.dxy * cell.dxy) / (2. * dt_mean_i)
		#
		index.append(i)
		dt_mean.append(dt_mean_i)
		initial.append(initial_i)
	any_cell = cell
	index, dt_mean, initialD = np.array(index), np.array(dt_mean), np.array(initial)
	initialF = np.zeros((len(cells), any_cell.dim), dtype=initialD.dtype)
	df = ChainArray('D', initialD, 'F', initialF)
	# parametrize the optimization algorithm
	if min_diffusivity is not None:
		kwargs['bounds'] = [(min_diffusivity, None)] * initialD.size + \
			[(None, None)] * initialF.size
	#cell.cache = None # no cache needed
	sq_loc_err = localization_error * localization_error
	args = (df, cells, sq_loc_err, diffusivity_prior, jeffreys_prior, dt_mean, min_diffusivity)
	result = minimize(smooth_df_neg_posterior, df.combined, args=args, **kwargs)
	# collect the result
	df.update(result.x)
	D, F = df['D'], df['F']
	index = np.array(list(cells.keys()))
	DF = pd.DataFrame(np.c_[D[:,np.newaxis], F], index=index, \
		columns=[ 'diffusivity' ] + \
			[ 'force ' + col for col in any_cell.space_cols ])
	# derivate the diffusive drifts
	if diffusive_drift:
		index_, drift = [], []
		for i, D_i in zip(index, D):
			gradD_i = cells.grad(i, D)
			if gradD_i is not None:
				index_.append(i)
				drift.append( D_i * gradD_i )
		drift = pd.DataFrame(np.stack(drift, axis=0), index=index_, \
			columns=[ 'drift ' + col for col in any_cell.space_cols ])
		DF = DF.join(drift)
	return DF

