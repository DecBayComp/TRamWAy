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


setup = {'arguments': OrderedDict((
		('localization_error',	('-e', dict(type=float, default=0.01, help='localization error'))),
		('jeffreys_prior',	('-j', dict(action='store_true', help="Jeffreys' prior"))),
		('min_diffusivity',	dict(type=float, default=0, help='minimum diffusivity value allowed'))))}


def df_neg_posterior(x, df, cell, square_localization_error=0.0, jeffreys_prior=False, \
		min_diffusivity=0):
	"""
	Adapted from InferenceMAP's *dfPosterior* procedure::

		for (int i = 0; i < ZONES[CURRENT_ZONE].translocations; i++) {
			const double dt = ZONES[CURRENT_ZONE].dt[i];
			const double dx = ZONES[CURRENT_ZONE].dx[i];
			const double dy = ZONES[CURRENT_ZONE].dy[i];

			D_bruit = LOCALIZATION_ERROR*LOCALIZATION_ERROR/dt;

			result += - log(4.0*PI*(FxFyD[2]+D_bruit)*dt ) - pow(fabs(dx-FxFyD[2]*FxFyD[0]*dt ),2.0)/(4.0*(FxFyD[2]+D_bruit)*dt) - pow(fabs(dy-FxFyD[2]*FxFyD[1]*dt ),2.0)/(4.0*(FxFyD[2]+D_bruit)*dt);
		}

		if (JEFFREYS_PRIOR == 1) {
			result += 2.0*log(FxFyD[2]) - 2.0*log(FxFyD[2]*ZONES[CURRENT_ZONE].dtMean + LOCALIZATION_ERROR*LOCALIZATION_ERROR);
		}
		return -result;

	"""
	df.update(x)
	D, F = df['D'], df['F']
	if D < min_diffusivity:
		warn(DiffusivityWarning(D, min_diffusivity))
	noise_dt = square_localization_error
	n = cell.dt.size # number of translocations
	D_dt = D * cell.dt
	D_term = 4.0 * (D_dt + noise_dt) # 4*(D+Dnoise)*dt
	dxdy_minus_D_F_dt = cell.dxy - np.outer(D_dt, F)
	F_term = np.sum(dxdy_minus_D_F_dt * dxdy_minus_D_F_dt, axis=1)
	neg_posterior_d = n * log(pi) + np.sum(np.log(D_term)) # sum(log(4*pi*Dtot*dt))
	neg_posterior_f = np.sum(F_term / D_term)
	if jeffreys_prior:
		neg_posterior_d += 2.0 * (log(D * np.mean(cell.dt) + noise_dt) - log(D))
	return neg_posterior_d + neg_posterior_f


def inferDF(cell, localization_error=0.0, jeffreys_prior=False, min_diffusivity=0, **kwargs):
	if isinstance(cell, Distributed):
		index, inferred = zip(*[ (i,
				inferDF(c, localization_error, jeffreys_prior, min_diffusivity,
					**kwargs))
			for i, c in cell.items() if bool(c) ])
		inferred = pd.DataFrame(data=np.stack(inferred, axis=0), \
			index=np.array(index), \
			columns=[ 'diffusivity' ] + \
				[ 'force '+col for col in next(iter(cell.values())).space_cols ])
		return inferred
	else:
		if not np.all(0 < cell.dt):
			warn('translocation dts are non-positive', RuntimeWarning)
			cell.dxy[cell.dt < 0] *= -1.
			cell.dt[ cell.dt < 0] *= -1.
		initialD = np.mean(cell.dxy * cell.dxy) / (2.0 * np.mean(cell.dt))
		initialF = np.zeros(cell.dim, dtype=initialD.dtype)
		df = ChainArray('D', initialD, 'F', initialF)
		if min_diffusivity is not None:
			if 'bounds' in kwargs:
				print(kwargs['bounds'])
			kwargs['bounds'] = [(min_diffusivity, None)] + [(None, None)] * initialF.size
		#cell.cache = None # no cache needed
		sq_loc_err = localization_error * localization_error
		result = minimize(df_neg_posterior, df.combined, \
			args=(df, cell, sq_loc_err, jeffreys_prior, min_diffusivity), \
			**kwargs)
		#df.update(result.x)
		#return (df['D'], df['F'])
		return result.x # needless to split df.combined into D and F

