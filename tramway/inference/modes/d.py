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
		('jeffreys_prior',	('-j', dict(action='store_true', help="Jeffreys' prior"))),
		('min_diffusivity',	dict(type=float, default=0, help='minimum diffusivity value allowed'))))}


def d_neg_posterior(diffusivity, cell, square_localization_error=0.0, jeffreys_prior=False, \
	min_diffusivity=0):
	if diffusivity < min_diffusivity and not np.isclose(diffusivity, min_diffusivity):
		warn(DiffusivityWarning(diffusivity, min_diffusivity))
	noise_dt = square_localization_error
	if cell.cache is None:
		cell.cache = np.sum(cell.dxy * cell.dxy, axis=1) # dx**2 + dy**2 + ..
	n = cell.cache.size # number of translocations
	diffusivity_dt = 4.0 * (diffusivity * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
	if np.any(np.isclose(diffusivity_dt, 0)):
		raise RuntimeError('near-0 diffusivity; increase `localization_error`')
	d_neg_posterior_dt = n * log(pi) + np.sum(np.log(diffusivity_dt)) # sum(log(4*pi*Dtot*dt))
	d_neg_posterior_dxy = np.sum(cell.cache / diffusivity_dt) # sum((dx**2+dy**2+..)/(4*Dtot*dt))
	if jeffreys_prior:
		d_neg_posterior_dt += diffusivity * np.mean(cell.dt) + noise_dt
	return d_neg_posterior_dt + d_neg_posterior_dxy


def inferD(cell, localization_error=0.0, jeffreys_prior=False, min_diffusivity=0, **kwargs):
	sq_loc_err = localization_error * localization_error
	if isinstance(cell, Distributed):
		inferred = {i: inferD(c, localization_error, jeffreys_prior, \
				min_diffusivity, **kwargs) \
			for i, c in cell.cells.items() if 0 < c.tcount}
		inferred = pd.DataFrame(data={'diffusivity': pd.Series(data=inferred)})
		return inferred
	else:
		#assert not np.isclose(np.mean(cell.dt), 0)
		if not np.all(0 < cell.dt):
			warn('translocation dts are non-positive', RuntimeWarning)
			cell.dt = abs(cell.dt)
		initial_diffusivity = np.mean(cell.dxy * cell.dxy) / (2.0 * np.mean(cell.dt))
		cell.cache = None # initialize empty cache
		if min_diffusivity is not None:
			kwargs['bounds'] = [(min_diffusivity,None)]
		result = minimize(d_neg_posterior, initial_diffusivity, \
			args=(cell, sq_loc_err, jeffreys_prior, min_diffusivity), \
			**kwargs)
		return result.x[0]

