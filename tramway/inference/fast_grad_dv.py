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
import time
from .gradient import *
from .stochastic_dv import LocalDV, make_regions
from .optimization import sparse_grad


setup = {'name': ('fast.grad.dv', 'fast.gradient.dv'),
        'provides': 'dv',
        'arguments': OrderedDict((
        ('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
        ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
        ('potential_prior',     ('-v', dict(type=float, help='prior on the potential'))),
        ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
        ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')),
        ('max_iter',        dict(type=int, help='maximum number of iterations (~100)')),
        ('compatibility',       ('-c', '--inferencemap', '--compatible',
                    dict(action='store_true', help='InferenceMAP compatible'))),
        ('export_centers',      dict(action='store_true')),
        ('verbose',         ()))),
    'cell_sampling': 'connected'}
setup_with_grad_arguments(setup)


def local_dv_neg_posterior(j, x, dv, cells, sigma2, jeffreys_prior,
    dt_mean, index, reverse_index, grad_kwargs,
    y0, verbose, posterior_info):
    """
    """
    Dj = x[j]
    if np.any(np.isnan(Dj)):
        raise ValueError('D is nan')
    V = x[int(x.size/2):]
    #

    noise_dt = sigma2

    # for all cell
    i = index[j]
    cell = cells[i]
    n = len(cell) # number of translocations

    # spatial gradient of the local potential energy
    gradV = cells.grad(i, V, reverse_index, **grad_kwargs)
    if gradV is None or np.any(np.isnan(gradV)):
        dv.undefined_grad(i, 'V')
        gradV = np.zeros(cell.dim)

    # various posterior terms
    #print(cell.dt)
    D_dt = Dj * cell.dt
    denominator = 4. * (D_dt + noise_dt)
    if np.any(denominator <= 0):
        raise ValueError('undefined posterior; local diffusion value: %s', Dj)
    dr_minus_drift = cell.dr + np.outer(D_dt, gradV)
    # non-directional squared displacement
    ndsd = np.sum(dr_minus_drift * dr_minus_drift, axis=1)
    raw_posterior = n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)

    if np.isnan(raw_posterior):
        raise ValueError('undefined posterior; local diffusion value: %s', Dj)

    # priors
    standard_priors = 0.
    V_prior = dv.potential_prior(j)
    if V_prior:
        deltaV = cells.local_variation(i, V, reverse_index, **grad_kwargs)
        if deltaV is not None:
            standard_priors += V_prior * cells.grad_sum(i, deltaV * deltaV, reverse_index)
    D_prior = dv.diffusivity_prior(j)
    if D_prior:
        D = x[:int(x.size/2)]
        # spatial gradient of the local diffusivity
        deltaD = cells.local_variation(i, D, reverse_index, **grad_kwargs)
        if deltaD is not None:
            # `grad_sum` memoizes and can be called several times at no extra cost
            standard_priors += D_prior * cells.grad_sum(i, deltaD * deltaD, reverse_index)
    if jeffreys_prior:
        if Dj <= 0:
            raise ValueError('non positive diffusivity')
        standard_priors += jeffreys_prior * 2. * np.log(Dj * dt_mean[j] + sigma2) - np.log(Dj)

    priors = standard_priors
    result = raw_posterior + priors

    if posterior_info is not None:
        if iter_num is None:
            info = [i, raw_posterior, result]
        else:
            info = [iter_num, i, raw_posterior, result]
        posterior_info.append(info)

    return result

def _local_dv_neg_posterior(*args, **kwargs):
    try:
        return local_dv_neg_posterior(*args, **kwargs)
    except ValueError:
        return 0.

def dv_neg_posterior_grad(m, dv):
    def col2rows(j):
        i = j % m
        return dv.region(i)
    def grad(x, *args):
        _grad, _ = sparse_grad(_local_dv_neg_posterior, x, col2rows, None, args)
        if _grad is None:
            return np.full_like(x, np.nan)
        else:
            return _grad
    return grad

def dv_neg_posterior(m):
    def fun(x, *args):
        try:
            return sum( local_dv_neg_posterior(j, x, *args) for j in range(m) )
        except ValueError:
            return np.inf
    return fun

def infer_fast_grad_DV(cells, diffusivity_prior=None, potential_prior=None, \
    jeffreys_prior=False, min_diffusivity=None, max_iter=None, epsilon=None, \
    export_centers=False, verbose=True, compatibility=False, \
    D0=None, V0=None, **kwargs):

    localization_error = cells.get_localization_error(kwargs, 0.03, True)

    # initial values
    index, reverse_index, n, dt_mean, D_initial, min_diffusivity, D_bounds, border = \
        smooth_infer_init(cells, min_diffusivity=min_diffusivity, jeffreys_prior=jeffreys_prior,
        sigma2=localization_error)
    # V initial values
    if V0 is None:
        try:
            if compatibility:
                raise Exception # skip to the except block
            volume = [ cells[i].volume for i in index ]
        except:
            V_initial = -np.log(n / np.max(n))
        else:
            density = n / np.array([ np.inf if v is None else v for v in volume ])
            density[density == 0] = np.min(density[0 < density])
            V_initial = np.log(np.max(density)) - np.log(density)
    else:
        if np.isscalar(V0):
            V_initial = np.full(D_initial.size, V0)
        elif V0.size == D_initial.size:
            V_initial = V0
        else:
            raise ValueError('wrong size for V0')
    if D0 is not None:
        if np.isscalar(D0):
            D_initial[...] = D0
        elif D0.size == D_initial.size:
            D_initial = D0
        else:
            raise ValueError('wrong size for D0')

    dv = LocalDV(D_initial, V_initial, diffusivity_prior, potential_prior,
            minimum_diffusivity=min_diffusivity)
    posteriors = None

    # gradient options
    grad_kwargs = get_grad_kwargs(kwargs, epsilon=epsilon, compatibility=compatibility)

    # parametrize the optimization algorithm
    default_lBFGSb_options = dict(maxiter=1e3, maxfun=1e10, maxcor=dv.combined.size, ftol=1e-8)
    # in L-BFGS-B the number of iterations is usually very low (~10-100) while the number of
    # function evaluations is much higher (~1e4-1e5);
    # with maxfun defined, an iteration can stop anytime and the optimization may terminate
    # with an error message
    if min_diffusivity is None:
        bounds = None
        options = {}
    else:
        V_bounds = [(None, None)] * V_initial.size
        bounds = D_bounds + V_bounds
        options = dict(default_lBFGSb_options)
    options.update(kwargs.pop('options', {}))
    options.update(**kwargs) # for backward compatibility
    if max_iter:
        options['maxiter'] = max_iter
    if verbose:
        options['disp'] = verbose
    if options:
        _kwargs = dict(options = options)
    else:
        _kwargs = {}

    m = len(index)
    fun = dv_neg_posterior(m)
    # posterior function input arguments
    args = (dv, cells, localization_error, jeffreys_prior, dt_mean,
            index, reverse_index, grad_kwargs)

    dv.regions = make_regions(cells, index, reverse_index)
    grad = dv_neg_posterior_grad(m, dv)

    # get the initial posterior value so that it is subtracted from the further evaluations
    y0 = fun(dv.combined, *(args + (0., False, None)))
    if verbose:
        print('At X0\tactual posterior= {}\n'.format(y0))
    #y0 = 0.
    args = args + (y0, 1 < verbose, posteriors)

    # run the optimization routine
    result = minimize(fun, dv.combined, args=args,
            method='L-BFGS-B' if bounds else 'BFGS',
            bounds=bounds, jac=grad, **_kwargs)
    if not (result.success or verbose):
        warn('{}'.format(result.message), OptimizationWarning)

    y = np.array(result.x)

    # collect the result
    dv.update(y)
    D, V = dv.D, dv.V
    if np.any(V < 0):
        V -= np.min(V)
    DVF = pd.DataFrame(np.stack((D, V), axis=1), index=index, \
        columns=[ 'diffusivity', 'potential'])

    # derivate the forces
    index_, F = [], []
    for i in index:
        gradV = cells.grad(i, V, reverse_index, **grad_kwargs)
        if gradV is not None:
            index_.append(i)
            F.append(-gradV)
    if F:
        F = pd.DataFrame(np.stack(F, axis=0), index=index_, \
            columns=[ 'force ' + col for col in cells.space_cols ])
    else:
        warn('not any cell is suitable for evaluating the local force', RuntimeWarning)
        F = pd.DataFrame(np.zeros((0, len(cells.space_cols)), dtype=V.dtype), \
            columns=[ 'force ' + col for col in cells.space_cols ])
    DVF = DVF.join(F)

    # add extra information if required
    if export_centers:
        xy = np.vstack([ cells[i].center for i in index ])
        DVF = DVF.join(pd.DataFrame(xy, index=index, \
            columns=cells.space_cols))
        #DVF.to_csv('results.csv', sep='\t')

    # format the posteriors
    #posteriors = pd.DataFrame(np.array(posteriors), columns=['fit', 'total'])

    return DVF#, posteriors

