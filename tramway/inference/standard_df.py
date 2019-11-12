# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import ChainArray
from .base import *
from .gradient import *
from warnings import warn
from math import pi, log
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import OrderedDict


setup = {'name': ('standard.df', 'smooth.df'),
    'provides': 'df',
    'arguments': OrderedDict((
        ('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
        ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
        #('potential_prior',     ('-v', dict(type=float, help='prior on the potential (use force_prior instead)'))),
        ('force_prior',         ('-f', dict(type=float, help='prior on the amplitude of the force'))),
        ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
        ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')),
        ('max_iter',        dict(type=int, help='maximum number of iterations')),
        ('rgrad',       dict(help="alternative gradient for the regularization; can be 'delta'/'delta0' or 'delta1'")),
        ('tol',             dict(type=float, help='tolerance for scipy minimizer')))),
    'cell_sampling': 'group'}
setup_with_grad_arguments(setup)


def smooth_df_neg_posterior(x, df, cells, sigma2, diffusivity_prior,
        force_prior, jeffreys_prior, dt_mean, min_diffusivity,
        index, reverse_index, grad_kwargs):
    # extract `D` and `F`
    df.update(x)
    D, F = df['D'], df['F']
    #
    if min_diffusivity is not None:
        observed_min = np.min(D)
        if observed_min < min_diffusivity and not np.isclose(observed_min, min_diffusivity):
            warn(DiffusivityWarning(observed_min, min_diffusivity))
    noise_dt = sigma2
    # for all cell
    result = 0.
    for j, i in enumerate(index):
        cell = cells[i]
        n = len(cell) # number of translocations
        # various posterior terms
        D_dt = D[j] * cell.dt
        denominator = 4. * (D_dt + noise_dt) # 4*(D+Dnoise)*dt
        dr_minus_drift_dt = cell.dr - np.outer(D_dt, F[j])
        # non-directional squared displacement
        ndsd = np.sum(dr_minus_drift_dt * dr_minus_drift_dt, axis=1)
        result += n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)
        # priors
        if diffusivity_prior:
            gradD = cells.grad(i, D, reverse_index, **grad_kwargs) # spatial gradient of the local diffusivity
            if gradD is not None:
                # `grad_sum` memoizes and can be called several times at no extra cost
                result += diffusivity_prior * cells.grad_sum(i, gradD * gradD)
        if force_prior:
            result += force_prior * cells.grad_sum(i, F * F)
    if jeffreys_prior:
        result += 2. * np.sum(np.log(D * dt_mean + sigma2) - np.log(D))
    return result


def df_neg_posterior1(x, df, cells, sigma2, diffusivity_prior,
        force_prior, jeffreys_prior, dt_mean, min_diffusivity,
        index, reverse_index, grad_kwargs):
    # extract `D` and `F`
    df.update(x)
    D, F = df['D'], df['F']
    #
    if min_diffusivity is not None:
        observed_min = np.min(D)
        if observed_min < min_diffusivity and not np.isclose(observed_min, min_diffusivity):
            warn(DiffusivityWarning(observed_min, min_diffusivity))
    noise_dt = sigma2
    # for all cell
    result = 0.
    for j, i in enumerate(index):
        cell = cells[i]
        n = len(cell) # number of translocations
        # various posterior terms
        D_dt = D[j] * cell.dt
        denominator = 4. * (D_dt + noise_dt) # 4*(D+Dnoise)*dt
        dr_minus_drift_dt = cell.dr - np.outer(D_dt, F[j])
        # non-directional squared displacement
        ndsd = np.sum(dr_minus_drift_dt * dr_minus_drift_dt, axis=1)
        result += n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)
        # priors
        if diffusivity_prior:
            deltaD = cells.local_variation(i, D, reverse_index, **grad_kwargs) # spatial gradient of the local diffusivity
            if deltaD is not None:
                # `grad_sum` memoizes and can be called several times at no extra cost
                result += diffusivity_prior * cells.grad_sum(i, deltaD * deltaD)
        if force_prior:
            result += force_prior * cells.grad_sum(i, F * F)
    if jeffreys_prior:
        result += 2. * np.sum(np.log(D * dt_mean + sigma2) - np.log(D))
    return result


def infer_smooth_DF(cells, diffusivity_prior=None, force_prior=None, potential_prior=None,
        jeffreys_prior=False, min_diffusivity=None, max_iter=None, epsilon=None, rgrad=None,
        verbose=False, **kwargs):
    """
    Argument `potential_prior` is an alias for `force_prior` which penalizes the large force amplitudes.
    """

    # initial values
    localization_error = cells.get_localization_error(kwargs, 0.03, True)
    index, reverse_index, n, dt_mean, D_initial, min_diffusivity, D_bounds, _ = \
        smooth_infer_init(cells, min_diffusivity=min_diffusivity, jeffreys_prior=jeffreys_prior,
        sigma2=localization_error)
    F_initial = np.zeros((len(index), cells.dim), dtype=D_initial.dtype)
    df = ChainArray('D', D_initial, 'F', F_initial)

    # gradient options
    grad_kwargs = get_grad_kwargs(epsilon=epsilon, **kwargs)

    # parametrize the optimization algorithm
    default_lBFGSb_options = dict(maxiter=1e3, maxfun=1e10, ftol=1e-6)
    # in L-BFGS-B the number of iterations is usually very low (~10-100) while the number of
    # function evaluations is much higher (~1e4-1e5);
    # with maxfun defined, an iteration can stop anytime and the optimization may terminate
    # with an error message
    if min_diffusivity is None:
        options = {}
    else:
        F_bounds = [(None, None)] * F_initial.size # no bounds
        kwargs['bounds'] = D_bounds + F_bounds
        options = dict(default_lBFGSb_options)
    options.update(kwargs.pop('options', {}))
    if max_iter:
        options['maxiter'] = max_iter
    if verbose:
        options['disp'] = verbose
    if options:
        kwargs['options'] = options

    # posterior function
    if rgrad in ('delta','delta0','delta1'):
        fun = df_neg_posterior1
    else:
        if rgrad not in (None, 'grad', 'grad1', 'gradn'):
            warn('unsupported rgrad: {}'.format(rgrad), RuntimeWarning)
        fun = smooth_df_neg_posterior

    if force_prior is None:
        if potential_prior is not None:
            raise ValueError('potential prior: {}'.format(potential_prior))
            warn('please use `force_prior` instead of `potential_prior`', PendingDeprecationWarning)
        force_prior = potential_prior

    # run the optimization
    #cell.cache = None # no cache needed
    args = (df, cells, localization_error, diffusivity_prior, force_prior, jeffreys_prior, dt_mean, min_diffusivity, index, reverse_index, grad_kwargs)
    result = minimize(fun, df.combined, args=args, **kwargs)
    if not (result.success or verbose):
        warn('{}'.format(result.message), OptimizationWarning)

    # collect the result
    df.update(result.x)
    D, F = df['D'], df['F']
    DF = pd.DataFrame(np.concatenate((D[:,np.newaxis], F), axis=1), index=index, \
        columns=[ 'diffusivity' ] + \
            [ 'force ' + col for col in cells.space_cols ])

    return DF

