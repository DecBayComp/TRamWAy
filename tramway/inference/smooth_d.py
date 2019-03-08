# -*- coding: utf-8 -*-

# Copyright © 2017-2018, Institut Pasteur
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


setup = {'name': ('standard.d', 'smooth.d'),
    'provides': 'd',
    'arguments': OrderedDict((
        ('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
        ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
        ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
        ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')),
        ('max_iter',        dict(type=int, help='maximum number of iterations')),
        ('tol',             dict(type=float, help='tolerance for scipy minimizer')),
        ('epsilon',         dict(args=('--eps',), kwargs=dict(type=float, help='if defined, every gradient component can recruit all of the neighbours, minus those at a projected distance less than this value'), translate=True)))),
    'cell_sampling': 'group'}


def smooth_d_neg_posterior(diffusivity, cells, sigma2, diffusivity_prior, \
    jeffreys_prior, dt_mean, min_diffusivity, index, reverse_index, grad_kwargs):
    """
    Adapted from InferenceMAP's *dDDPosterior* procedure:

    .. code-block:: c++

        for (int a = 0; a < NUMBER_OF_ZONES; a++) {
            ZONES[a].gradDx = dvGradDx(DD,a);
            ZONES[a].gradDy = dvGradDy(DD,a);
            ZONES[a].priorActive = true;
        }

        for (int z = 0; z < NUMBER_OF_ZONES; z++) {
            const double gradDx = ZONES[z].gradDx;
            const double gradDy = ZONES[z].gradDy;
            const double D = DD[z];

            for (int j = 0; j < ZONES[z].translocations; j++) {
                const double dt = ZONES[z].dt[j];
                const double dx = ZONES[z].dx[j];
                const double dy = ZONES[z].dy[j];
                const double Dnoise = LOCALIZATION_ERROR*LOCALIZATION_ERROR/dt;

                result += - log(4.0*PI*(D + Dnoise)*dt) - ( dx*dx + dy*dy)/(4.0*(D+Dnoise)*dt);
            }

            if (ZONES[z].priorActive == true) {
                result -= D_PRIOR*(gradDx*gradDx*ZONES[z].areaX + gradDy*gradDy*ZONES[z].areaY);
                if (JEFFREYS_PRIOR == 1) {
                    result += 2.0*log(D) - 2.0*log(D*ZONES[z].dtMean + LOCALIZATION_ERROR*LOCALIZATION_ERROR);
                }
            }
        }

        return -result;

    """
    if min_diffusivity is not None:
        observed_min = np.min(diffusivity)
        if observed_min < min_diffusivity and not np.isclose(observed_min, min_diffusivity):
            warn(DiffusivityWarning(observed_min, min_diffusivity))
    noise_dt = sigma2
    result = 0.
    for j, i in enumerate(index):
        cell = cells[i]
        n = len(cell)
        # posterior calculations
        if cell.cache is None:
            cell.cache = dict(dr2=None)
        if cell.cache['dr2'] is None:
            cell.cache['dr2'] = np.sum(cell.dr * cell.dr, axis=1) # dx**2 + dy**2 + ..
        D_dt = 4. * (diffusivity[j] * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
        result += n * log(pi) + np.sum(np.log(D_dt)) # sum(log(4*pi*Dtot*dt))
        result += np.sum(cell.cache['dr2'] / D_dt) # sum((dx**2+dy**2+..)/(4*Dtot*dt))
        # prior
        if diffusivity_prior:
            # gradient of diffusivity
            gradD = cells.grad(i, diffusivity, reverse_index, **grad_kwargs)
            if gradD is not None:
                result += diffusivity_prior * cells.grad_sum(i, gradD * gradD)
    if jeffreys_prior:
        result += 2. * np.sum(np.log(diffusivity * dt_mean + sigma2))
    return result

def infer_smooth_D(cells, diffusivity_prior=None, jeffreys_prior=None, \
    min_diffusivity=None, max_iter=None, epsilon=None, **kwargs):

    # initial values
    index, reverse_index, n, dt_mean, D_initial, min_diffusivity, D_bounds, _ = \
        smooth_infer_init(cells, min_diffusivity=min_diffusivity, jeffreys_prior=jeffreys_prior)

    # gradient options
    grad_kwargs = {}
    if epsilon is not None:
        if compatibility:
            warn('epsilon should be None for backward compatibility with InferenceMAP', RuntimeWarning)
        grad_kwargs['eps'] = epsilon

    # parametrize the optimization procedure
    if min_diffusivity is not None:
        kwargs['bounds'] = D_bounds
    if max_iter:
        options = kwargs.get('options', {})
        options['maxiter'] = max_iter
        kwargs['options'] = options

    # run the optimization
    localization_error = cells.get_localization_error(kwargs, 0.03, True)
    result = minimize(smooth_d_neg_posterior, D_initial, \
        args=(cells, localization_error, diffusivity_prior, jeffreys_prior, dt_mean, min_diffusivity, index, reverse_index, grad_kwargs), \
        **kwargs)

    # format the result
    D = result.x
    DD = pd.DataFrame(D, index=index, columns=['diffusivity'])

    return DD

