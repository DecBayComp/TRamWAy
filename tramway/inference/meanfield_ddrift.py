# -*- coding: utf-8 -*-

# Copyright © 2019-2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
from .meanfield import nandot, unify_priors, DiffusivityDrift
from math import *
import numpy as np
import pandas as pd
from collections import OrderedDict


setup = {'name': ('meanfield.dd', 'meanfield diffusivity,drift'),
        'provides': ('dd', 'ddrift', 'diffusivity,drift'),
        'arguments': OrderedDict((
            #('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
            ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
            ('drift_prior',         dict(type=float, help='prior on the amplitude of the drift')),
            ('diffusivity_time_prior',  ('--time-d', dict(type=float, help='prior on the temporal variations of the diffusivity'))),
            ('drift_time_prior',        dict(type=float, help='prior on the temporal variations of drift amplitude')),
            ('verbose',         ()))),
        'default_rgrad':    'delta0'}


def infer_meanfield_DD(cells, diffusivity_spatial_prior=None, drift_spatial_prior=None,
        diffusivity_time_prior=None, drift_time_prior=None,
        diffusivity_prior=None, drift_prior=None,
        diffusion_prior=None, diffusion_spatial_prior=None, diffusion_time_prior=None,
        dt=None, tol=1e-6, max_iter=1e4, verbose=True, **kwargs):
    """
    """
    #grad_kwargs = get_grad_kwargs(kwargs)
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)

    # aliases
    if diffusivity_prior is None:
        diffusivity_prior = diffusion_prior
    if diffusivity_spatial_prior is None:
        diffusivity_spatial_prior = diffusion_spatial_prior
    if diffusivity_time_prior is None:
        diffusivity_time_prior = diffusion_time_prior
    if drift_spatial_prior is None:
        drift_spatial_prior = drift_prior

    mf = DiffusivityDrift(cells, dt,
            unify_priors(diffusivity_prior, diffusivity_spatial_prior, diffusivity_time_prior),
            unify_priors(drift_prior, drift_spatial_prior, drift_time_prior),
            **kwargs)

    n = mf.n
    dt = mf.dt
    # reminder: dr and dr2 are AVERAGE (square) displacements
    dr = mf.dr
    dr2 = mf.dr2

    # constants
    aD_constant_factor = 1. / (4. * dt)
    b_mu_constant_factor = n / (2. * dt)

    # initial values
    a_mu = dr
    chi2_over_n = dr2 - np.sum(a_mu * a_mu, axis=1)
    aD = aD_constant_factor * chi2_over_n
    bD = n / (aD * aD)
    b_mu = b_mu_constant_factor / aD

    # regularize

    if mf.regularize_diffusivity:
        AD = BD = None

        if verbose:
            print(' ** regularizing diffusivity: ** ')

        neg_L = np.inf
        k = 0
        try:
            while True:
                neg_L_prev = neg_L

                AD, BD = mf.regularize_D(aD, bD, AD, BD)

                inv_abD = .5 / (AD * AD * BD)
                neg_L = np.sum(
                        n * (np.log(AD) - inv_abD) + \
                        (.5 + inv_abD) / (2. * AD * dt) * n * (chi2_over_n + 2. / b_mu) + \
                        .5 * (np.log(BD) + np.log(b_mu))
                        ) + \
                        mf.neg_log_prior('D', AD, aD)

                if verbose:
                    print('[{}] approx -logP\': {}  max AD: {}'.format(k, neg_L, np.max(AD)))

                # stopping criteria
                if np.isinf(neg_L) or np.isnan(neg_L):
                    resolution = 'DIVERGENCE: L UNDEFINED'
                    break

                elif neg_L_prev - neg_L < -tol:
                    resolution = 'DIVERGENCE: -L INCREASES'
                    break

                if neg_L_prev - neg_L < tol:
                    resolution = 'CONVERGENCE: DELTA -L < TOL'
                    break

                elif max_iter and k == max_iter:
                    resolution = 'MAXIMUM ITERATION REACHED'
                    break

                elif mf.static_landscape('D'):
                    break
                k += 1

        except KeyboardInterrupt:
            resolution = 'INTERRUPTED'
            pass
        if verbose:
            print(resolution.lower())

        aD, bD = AD, BD

    if mf.regularize_drift:
        A_mu = B_mu = None
        inv_abD = .5 / (aD * aD * bD)

        if verbose:
            print(' ** regularizing drift: ** ')

        neg_L = np.inf
        k = 0
        try:
            while True:
                neg_L_prev = neg_L

                A_mu, B_mu = mf.regularize_mu(a_mu, b_mu, A_mu, B_mu)
                chi2_over_n = dr2 - 2 * np.sum(dr * A_mu, axis=1) + np.sum(A_mu * A_mu, axis=1)

                neg_L = np.nansum(
                        (.5 + inv_abD) / (aD * dt) * n * (chi2_over_n + 2. / B_mu) + \
                        .5 * (np.log(bD) + np.log(B_mu))
                        ) + \
                        mf.neg_log_prior('mu', A_mu, a_mu)

                if verbose:
                    print('[{}] approx -logP\': {}  max ||A_mu||: {}'.format(k, neg_L, np.sqrt(np.max(np.sum(A_mu*A_mu,1)))))

                # stopping criteria
                if np.isinf(neg_L) or np.isnan(neg_L):
                    resolution = 'DIVERGENCE: L UNDEFINED'
                    break

                elif neg_L_prev - neg_L < -tol:
                    resolution = 'DIVERGENCE: -L INCREASES'
                    break

                if neg_L_prev - neg_L < tol:
                    resolution = 'CONVERGENCE: DELTA -L < TOL'
                    break

                elif max_iter and k == max_iter:
                    resolution = 'MAXIMUM ITERATION REACHED'
                    break

                elif mf.static_landscape('D'):
                    break
                k += 1

        except KeyboardInterrupt:
            resolution = 'INTERRUPTED'
            pass
        if verbose:
            print(resolution.lower())

        a_mu, b_mu = A_mu, B_mu

    D = aD
    drift = a_mu / dt[:,np.newaxis]

    DD = pd.DataFrame(np.hstack((D[:,np.newaxis], drift)), index=mf.index, \
        columns=[ 'diffusivity' ] + \
            [ 'drift ' + col for col in cells.space_cols ])

    return DD


__all__ = ['setup', 'DiffusivityDrift', 'infer_meanfield_DD']

