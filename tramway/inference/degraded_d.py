# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
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


setup = {'name':    'degraded.d',
        'provides': 'd',
        'arguments': OrderedDict((
        ('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
        ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
        ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')))),
        'cell_sampling':    'individual'}


def d_neg_posterior(diffusivity, cell, sigma2, jeffreys_prior, dt_mean, \
    min_diffusivity):
    """
    Adapted from InferenceMAP's *dPosterior* procedure:

    .. code-block:: c++

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
    if diffusivity < min_diffusivity:# and not np.isclose(diffusivity, min_diffusivity):
        warn(DiffusivityWarning(diffusivity, min_diffusivity))
    noise_dt = sigma2
    if cell.cache is None:
        cell.cache = np.sum(cell.dr * cell.dr, axis=1) # dx**2 + dy**2 + ..
    n = len(cell) # number of translocations
    D_dt = 4. * (diffusivity * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
    if np.any(D_dt <= 0):# or np.any(np.isclose(D_dt, 0)):
        raise RuntimeError('negative diffusion')
    d_neg_posterior = n * log(pi) + np.sum(np.log(D_dt)) # sum(log(4*pi*Dtot*dt))
    d_neg_posterior += np.sum(cell.cache / D_dt) # sum((dx**2+dy**2+..)/(4*Dtot*dt))
    if jeffreys_prior:
        d_neg_posterior += 2. * log(diffusivity * dt_mean + sigma2)
    return d_neg_posterior


def infer_D(cells, localization_error=None, jeffreys_prior=False, min_diffusivity=None, **kwargs):
    if isinstance(cells, Distributed): # multiple cells
        localization_error = cells.get_localization_error(kwargs, 0.03, True, \
                localization_error=localization_error)
        args = (localization_error, jeffreys_prior, min_diffusivity)
        inferred = { i: infer_D(c, *args, **kwargs) for i, c in cells.items() }
        inferred = pd.DataFrame({'diffusivity': pd.Series(inferred)})
        return inferred
    else: # single cell
        cell = cells
        # sanity checks
        if not bool(cell):
            raise ValueError('empty cell')
        if cell.dr.shape[1] == 0:
            raise ValueError('translocation array has no column')
        if cell.dt.shape[1:]:
            raise ValueError('time deltas are structured in multiple dimensions')
        # ensure that translocations are properly oriented in time
        if not np.all(0 < cell.dt):
            warn('translocation dts are not all positive', RuntimeWarning)
            cell.dr[cell.dt < 0] *= -1.
            cell.dt[cell.dt < 0] *= -1.
        #assert not np.isclose(np.mean(cell.dt), 0)
        # initialize the diffusivity value and cell cache
        dt_mean = np.mean(cell.dt)
        D_initial = np.mean(cell.dr * cell.dr) / (2. * dt_mean)
        cell.cache = None # clear the cache (optional, since `run` also clears it)
        # parametrize the optimization procedure
        if min_diffusivity is not False:
            if min_diffusivity is None:
                noise_dt = localization_error
                min_diffusivity = (1e-16-noise_dt) / np.max(cell.dt)
            kwargs['bounds'] = [(min_diffusivity,None)]
        # run the optimization
        result = minimize(d_neg_posterior, D_initial, \
            args=(cell, localization_error, jeffreys_prior, dt_mean, min_diffusivity), \
            **kwargs)
        # return the resulting optimal diffusivity value
        return result.x[0]

