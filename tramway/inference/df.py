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


def df_neg_posterior(x, df, cell, square_localization_error=0.0, jeffreys_prior=False, \
	min_diffusivity=0):
	df.update(x)
	diffusivity = df['D']
	force = df['F']
	if diffusivity < min_diffusivity:
		warn(DiffusivityWarning(diffusivity, min_diffusivity))
	noise_dt = square_localization_error
	n = cell.dt.size # number of translocations
	diffusivity_dt = diffusivity * cell.dt
	diffusivity_term = 4.0 * (diffusivity_dt + noise_dt) # 4*(D+Dnoise)*dt
	residual = cell.dxy - np.outer(diffusivity_dt, force)
	force_term = np.sum(residual * residual, axis=1)
	neg_posterior_d = n * log(pi) + np.sum(np.log(diffusivity_term)) # sum(log(4*pi*Dtot*dt))
	neg_posterior_f = np.sum(force_term / diffusivity_term)
	if jeffreys_prior:
		neg_posterior_d += 2.0 * (log(diffusivity * np.mean(cell.dt) + noise_dt) - log(diffusivity))
	return neg_posterior_d + neg_posterior_f


def inferDF(cell, localization_error=0.0, jeffreys_prior=False, min_diffusivity=0, **kwargs):
	if isinstance(cell, Distributed):
		sq_loc_err = localization_error * localization_error
		index, inferred = zip(*[ (i,
				inferDF(c, sq_loc_err, jeffreys_prior, min_diffusivity, **kwargs))
			for i, c in cell.cells.items() if 0 < c.tcount ])
		inferred = pd.DataFrame(data=np.stack(inferred, axis=0), \
			index=np.array(index), \
			columns=[ 'diffusivity' ] + \
				[ 'force '+col for col in next(iter(cell.cells.values())).space_cols ])
		return inferred
	else:
		if not np.all(0 < cell.dt):
			warn('translocation dts are non-positive', RuntimeWarning)
			cell.dt = abs(cell.dt)
		initial_diffusivity = np.mean(cell.dxy * cell.dxy) / (2.0 * np.mean(cell.dt))
		initialF = np.zeros(cell.dim, dtype=initial_diffusivity.dtype)
		df = ChainArray('D', initial_diffusivity, 'F', initialF)
		if min_diffusivity is not None:
			if 'bounds' in kwargs:
				print(kwargs['bounds'])
			kwargs['bounds'] = [(min_diffusivity, None)] + [(None, None)] * initialF.size
		#cell.cache = None # no cache needed
		result = minimize(df_neg_posterior, df.combined, \
			args=(df, cell, localization_error, jeffreys_prior, min_diffusivity), \
			**kwargs)
		# note that localization_error here is actually the square localization error
		#df.update(result.x)
		#return (df['D'], df['F'])
		return result.x # needless to split df.combined into D and F

