# -*- coding: utf-8 -*-

# Copyright © 2017 2018, Institut Pasteur
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


setup = {'name': ('smooth.dd', 'smooth.ddrift'),
        'provides': ('dd', 'ddrift'),
        'arguments': OrderedDict((
                ('localization_error',  ('-e', dict(type=float, default=0.03, help='localization error'))),
                ('diffusivity_prior',   ('-d', dict(type=float, default=1., help='prior on the diffusivity'))),
                ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
                ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')),
                ('max_iter',            dict(type=int, help='maximum number of iterations')),
                ('epsilon',             dict(args=('--eps',), kwargs=dict(type=float, help='if defined, every gradient component can recruit all of the neighbours, minus those at a projected distance less than this value'), translate=True)))),
        'cell_sampling': 'group'}


def smooth_dd_neg_posterior(x, dd, cells, squared_localization_error, diffusivity_prior,
                jeffreys_prior, dt_mean, min_diffusivity, index, reverse_index, grad_kwargs):
        # extract `D` and `drift`
        dd.update(x)
        D, drift = dd['D'], dd['drift']
        #
        if min_diffusivity is not None:
                observed_min = np.min(D)
                if observed_min < min_diffusivity and not np.isclose(observed_min, min_diffusivity):
                        warn(DiffusivityWarning(observed_min, min_diffusivity))
        noise_dt = squared_localization_error
        # for all cell
        result = 0.
        for j, i in enumerate(index):
                cell = cells[i]
                n = len(cell) # number of translocations
                # various posterior terms
                denominator = 4. * (D[j] * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
                dr_minus_drift_dt = cell.dr - np.outer(cell.dt, drift[j])
                # non-directional squared displacement
                ndsd = np.sum(dr_minus_drift_dt * dr_minus_drift_dt, axis=1)
                result += n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)
                # priors
                gradD = cells.grad(i, D, reverse_index, **grad_kwargs) # spatial gradient of the local diffusivity
                if gradD is not None:
                        # `grad_sum` memoizes and can be called several times at no extra cost
                        result += diffusivity_prior * cells.grad_sum(i, gradD * gradD)
        if jeffreys_prior:
                result += 2. * np.sum(np.log(D * dt_mean + squared_localization_error))
        return result


def infer_smooth_DD(cells, localization_error=0.03, diffusivity_prior=1., jeffreys_prior=False,
        min_diffusivity=None, max_iter=None, epsilon=None, **kwargs):

        # initial values
        index, reverse_index, n, dt_mean, D_initial, min_diffusivity, D_bounds, _ = \
                smooth_infer_init(cells, min_diffusivity=min_diffusivity, jeffreys_prior=jeffreys_prior)
        initial_drift = np.zeros((len(index), cells.dim), dtype=D_initial.dtype)
        drift_bounds = [(None, None)] * initial_drift.size # no bounds
        dd = ChainArray('D', D_initial, 'drift', initial_drift)

        # gradient options
        grad_kwargs = {}
        if epsilon is not None:
                if compatibility:
                        warn('epsilon should be None for backward compatibility with InferenceMAP', RuntimeWarning)
                grad_kwargs['eps'] = epsilon

        # parametrize the optimization algorithm
        if min_diffusivity is not None:
                kwargs['bounds'] = D_bounds + drift_bounds
        if max_iter:
                options = kwargs.get('options', {})
                options['maxiter'] = max_iter
                kwargs['options'] = options

        # run the optimization
        #cell.cache = None # no cache needed
        sle = localization_error * localization_error # sle = squared localization error
        args = (dd, cells, sle, diffusivity_prior, jeffreys_prior, dt_mean, min_diffusivity, index, reverse_index, grad_kwargs)
        result = minimize(smooth_dd_neg_posterior, dd.combined, args=args, **kwargs)

        # collect the result
        dd.update(result.x)
        D, drift = dd['D'], dd['drift']
        DD = pd.DataFrame(np.hstack((D[:,np.newaxis], drift)), index=index, \
                columns=[ 'diffusivity' ] + \
                        [ 'drift ' + col for col in cells.space_cols ])

        return DD

