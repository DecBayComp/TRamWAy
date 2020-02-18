# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
from .meanfield import nandot, unify_priors, FrictionDrift
from math import *
import numpy as np
import pandas as pd
from collections import OrderedDict


setup = {'name': ('meanfield.fd', 'meanfield friction,drift'),
        'provides': ('fd', 'friction,drift'),
        'arguments': OrderedDict((
            #('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
            ('friction_prior',  dict(type=float, help='prior on the friction')),
            ('drift_prior',         dict(type=float, help='prior on the amplitude of the drift')),
            ('friction_time_prior', dict(type=float, help='prior on the temporal variations of the friction')),
            ('drift_time_prior',    dict(type=float, help='prior on the temporal variations of drift amplitude')),
            ('verbose',         ()))),
        'default_rgrad':    'delta0'}



def infer_meanfield_friction_drift(cells, friction_spatial_prior=None, drift_spatial_prior=None,
        friction_time_prior=None, drift_time_prior=None,
        friction_prior=None, drift_prior=None,
        dt=None, tol=1e-6, max_iter=1e4, verbose=True, **kwargs):
    """
    """
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)

    mf = FrictionDrift(cells, dt,
            unify_priors(friction_prior, friction_spatial_prior, friction_time_prior),
            unify_priors(drift_prior, drift_spatial_prior, drift_time_prior),
            **kwargs)

    n = mf.n
    dt = mf.dt
    # reminder: dr and dr2 are AVERAGE (square) displacements
    dr = mf.dr
    dr2 = mf.dr2

    # initial values
    a_mu = dr
    chi2_over_n = dr2 - np.sum(a_mu * a_mu, axis=1)
    a_psi = 2. / chi2_over_n
    b_psi = n / (a_psi * a_psi)
    b_mu = n * a_psi

    # regularize
    A_psi = B_psi = A_mu = B_mu = None

    if mf.regularize_friction:

        # no need to recompute chi2_over_n as long as a_mu == dr
        #chi2_over_n = dr2 - 2 * np.sum(dr * a_mu, axis=1) + np.sum(a_mu * a_mu, axis=1)

        if verbose:
            print(' ** regularizing friction: ** ')

        neg_L = np.inf
        k = 0
        try:
            while True:
                neg_L_prev = neg_L

                A_psi, B_psi = mf.regularize_psi(a_psi, b_psi, A_psi, B_psi)

                neg_L = nandot(n, \
                            1./ (2 * A_psi * A_psi * B_psi) - \
                            np.log(A_psi) + \
                            A_psi * (chi2_over_n / 2 + 1. / b_mu)
                            ) + \
                        .5 * np.sum(np.log(B_psi) + np.log(b_mu)) + \
                        mf.neg_log_prior('psi', A_psi, a_psi)

                if verbose:
                    print('[{}] approx -logP\': {}  max A_psi: {}'.format(k, neg_L, np.max(A_psi)))

                # stopping criteria
                if np.isinf(neg_L) or np.isnan(neg_L):
                    resolution = 'DIVERGENCE: L UNDEFINED'
                    break

                elif neg_L_prev - neg_L < -tol:
                    resolution = 'DIVERGENCE: -L INCREASES'
                    break

                elif neg_L_prev - neg_L < tol:
                    resolution = 'CONVERGENCE: DELTA -L < TOL'
                    break

                elif max_iter and k == max_iter:
                    resolution = 'MAXIMUM ITERATION REACHED'
                    break

                elif mf.static_landscape('psi'):
                    break
                k += 1

        except KeyboardInterrupt:
            resolution = 'INTERRUPTED'
            pass
        if verbose:
            print(resolution.lower())

        a_psi, b_psi = A_psi, B_psi

    if mf.regularize_drift:

        if verbose:
            print(' ** regularizing drift: ** ')

        neg_L = np.inf
        k = 0
        try:
            while True:
                neg_L_prev = neg_L

                A_mu, B_mu = mf.regularize_mu(a_mu, b_mu, A_mu, B_mu)
                chi2_over_n = dr2 - 2 * np.sum(dr * A_mu, axis=1) + np.sum(A_mu * A_mu, axis=1)

                neg_L = nandot(n,
                            1./ (2 * a_psi * a_psi * b_psi) - \
                            np.log(a_psi) + \
                            a_psi * (chi2_over_n / 2 + 1. / B_mu)
                            ) + \
                        .5 * np.sum(np.log(b_psi) + np.log(B_mu)) + \
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

                elif neg_L_prev - neg_L < tol:
                    resolution = 'CONVERGENCE: DELTA -L < TOL'
                    break

                elif max_iter and k == max_iter:
                    resolution = 'MAXIMUM ITERATION REACHED'
                    break

                elif mf.static_landscape('mu'):
                    break

                k += 1

        except KeyboardInterrupt:
            resolution = 'INTERRUPTED'
            pass
        if verbose:
            print(resolution.lower())

        a_mu, b_mu = A_mu, B_mu

    gamma = a_psi * (2 * dt)
    drift = a_mu / dt[:,np.newaxis]

    FD = pd.DataFrame(np.hstack((gamma[:,np.newaxis], drift)), index=mf.index, \
        columns=[ 'friction' ] + \
            [ 'drift ' + col for col in cells.space_cols ])

    return FD


__all__ = ['setup', 'infer_meanfield_friction_drift', 'FrictionDrift']

