# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


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


def d_neg_posterior(diffusivity, cell, squared_localization_error, jeffreys_prior, min_diffusivity):
	"""
	Adapted from InferenceMAP's *dPosterior* procedure::

		for (int i = 0; i < ZONES[CURRENT_ZONE].translocations; i++) {
			const double dt = ZONES[CURRENT_ZONE].dt[i];
			const double dx = ZONES[CURRENT_ZONE].dx[i];
			const double dy = ZONES[CURRENT_ZONE].dy[i];

			D_bruit = LOCALIZATION_ERROR*LOCALIZATION_ERROR/dt;

			result += - log(4.0*PI*(D[0]+D_bruit)*dt) - (dx*dx)/(4.0*(D[0]+D_bruit)*dt) - (dy*dy)/(4.0*(D[0]+D_bruit)*dt);
		}

		if (JEFFREYS_PRIOR == 1) {
			result += - (D[0]*ZONES[CURRENT_ZONE].dtMean + LOCALIZATION_ERROR*LOCALIZATION_ERROR);
		}
		return -result;

	"""
	if diffusivity < min_diffusivity and not np.isclose(diffusivity, min_diffusivity):
		warn(DiffusivityWarning(diffusivity, min_diffusivity))
	noise_dt = squared_localization_error
	if cell.cache is None:
		cell.cache = np.sum(cell.dxy * cell.dxy, axis=1) # dx**2 + dy**2 + ..
	n = cell.cache.size # number of translocations
	D_dt = 4.0 * (diffusivity * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
	if np.any(np.isclose(D_dt, 0)):
		raise RuntimeError('near-0 diffusivity; increase `localization_error`')
	d_neg_posterior = n * log(pi) + np.sum(np.log(D_dt)) # sum(log(4*pi*Dtot*dt))
	d_neg_posterior += np.sum(cell.cache / D_dt) # sum((dx**2+dy**2+..)/(4*Dtot*dt))
	if jeffreys_prior:
		d_neg_posterior += diffusivity * np.mean(cell.dt) + noise_dt
	return d_neg_posterior


def inferD(cell, localization_error=0.03, jeffreys_prior=False, min_diffusivity=0, **kwargs):
	if isinstance(cell, Distributed):
		args = (localization_error, jeffreys_prior, min_diffusivity)
		inferred = {i: inferD(c, *args, **kwargs) \
			for i, c in cell.items() if bool(c)}
		inferred = pd.DataFrame(data={'diffusivity': pd.Series(data=inferred)})
		return inferred
	else:
		# sanity checks
		if not np.all(0 < cell.dt):
			warn('translocation dts are not all positive', RuntimeWarning)
			cell.dxy[cell.dt < 0] *= -1.
			cell.dt[ cell.dt < 0] *= -1.
		#assert not np.isclose(np.mean(cell.dt), 0)
		# initialize the diffusivity value and cell cache
		initialD = np.mean(cell.dxy * cell.dxy) / (2.0 * np.mean(cell.dt))
		cell.cache = None # clear the cache (optional, since `run` also clears it)
		# parametrize the optimization procedure
		if min_diffusivity is not None:
			kwargs['bounds'] = [(min_diffusivity,None)]
		# run the optimization
		sq_loc_err = localization_error * localization_error
		result = minimize(d_neg_posterior, initialD, \
			args=(cell, sq_loc_err, jeffreys_prior, min_diffusivity), \
			**kwargs)
		# return the resulting optimal diffusivity value
		return result.x[0]

