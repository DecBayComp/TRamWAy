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
from .gradient import setup_with_grad_arguments
from .meanfield import nandot, unify_priors, FrictionPotential
from .meanfield_dv import merge_a, merge_b, _nan
from math import *
import numpy as np
import pandas as pd
from collections import OrderedDict
from warnings import warn


setup = {'name': 'meanfield friction,potential',
        'provides': 'friction,potential',
        'arguments': OrderedDict((
            #('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
            ('friction_prior',   dict(type=float, help='prior on the friction')),
            ('potential_prior',     ('-v', dict(type=float, help='prior on the effective potential energy'))),
            ('friction_time_prior',   dict(type=float, help='prior on the temporal variations of the friction')),
            ('potential_time_prior',     ('--time-v', dict(type=float, help='prior on the temporal variations of the effective potential'))),
            ('verbose',         ()))),
        'default_grad':     'onesided_gradient',
        'default_rgrad':    'delta0'}
setup_with_grad_arguments(setup)



def infer_meanfield_friction_potential(cells, friction_spatial_prior=None, potential_spatial_prior=None,
        friction_time_prior=None, potential_time_prior=None,
        friction_prior=None, potential_prior=None,
        dt=None, tol=1e-6, max_iter=1e4, eps=1e-3, verbose=False, a_psi_formula='exact', **kwargs):
    """
    """
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)

    mf = FrictionPotential(cells, dt,
            unify_priors(friction_prior, friction_spatial_prior, friction_time_prior),
            unify_priors(potential_prior, potential_spatial_prior, potential_time_prior),
            **kwargs)

    n = mf.n
    dt = mf.dt
    mean_dr = mf.dr
    mean_dr2 = mf.dr2
    sum_dr2 = n * mf.dr2
    index = mf.index

    B, Bstar, B2 = mf.B, mf.Bstar, mf.B2

    try:
        if False:#compatibility:
            raise Exception # skip to the except block
        volume = [ cells[i].volume for i in index ]
    except:
        V_initial = -np.log(n / np.max(n))
    else:
        density = n / np.array([ np.inf if v is None else v for v in volume ])
        density[density == 0] = np.min(density[0 < density])
        V_initial = np.log(np.max(density)) - np.log(density)

    # constants
    bV_constant_factor = {s: B2[s] * n for s in B2}

    if a_psi_formula.startswith('exact'):
        exact_a_psi_formula = True
        approximate_a_psi_formula = a_psi_formula.endswith('+')
    elif a_psi_formula.startswith('approx'):
        approximate_a_psi_formula = True
        exact_a_psi_formula = a_psi_formula.endswith('+')
    else:
        raise ValueError("a_psi_formula='{}' not supported".format(a_psi_formula))

    # initial values
    chi2_over_n = mean_dr2 - np.sum(mean_dr * mean_dr, axis=1)
    psi = 2. / chi2_over_n#mean_dr2
    #V = np.zeros_like(D)
    V = V_initial

    fixed_neighbours = False
    slow_neighbours = True

    resolution = None
    aVs, bVs, a_psis, b_psis = [], [], [], []
    grad_aVs, grad_aV2s = [], []
    for s in mf.gradient_sides:

        # 1. adjust psi and V jointly at each bin considering the other bins constant (no smoothing)
        a_psi, aV = psi, V

        if verbose:
            if mf.gradient_sides[1:]:
                print(' **          gradient side: \'{}\'          ** '.format(s))
            print(' ** preliminary fit (no regularization): ** ')

        i = 0
        neg_L = np.inf
        try:
            while True:
                neg_L_prev = neg_L

                # new aV
                if not fixed_neighbours or i==0:
                    gradV_plus_VB = _nan(mf.diff_grad(aV, s))
                aV = np.sum(  Bstar[s] * (gradV_plus_VB + mean_dr * a_psi[:,np.newaxis])  ,axis=1)
                if slow_neighbours:
                    grad_aV = gradV_plus_VB - aV[:,np.newaxis] * B[s]
                else:
                    grad_aV = mf.grad(aV, s)
                grad_aV2 = np.nansum(grad_aV * grad_aV, axis=1)

                # new a_psi
                if approximate_a_psi_formula:
                    chi2_over_n = mean_dr2 + \
                            2 * np.nansum(mean_dr * grad_aV, axis=1) / a_psi + \
                            grad_aV2 / (a_psi * a_psi)
                    approx_a_psi = 2 / chi2_over_n
                if exact_a_psi_formula:
                    exact_a_psi = (1 + np.sqrt(mean_dr2 * grad_aV2)) / mean_dr2
                a_psi = approx_a_psi if a_psi_formula.startswith('approx') else exact_a_psi

                # new bV and b_psi
                bV = bV_constant_factor[s] / a_psi
                b_psi = n / (a_psi * a_psi) * (1 + grad_aV2 / a_psi)

                if verbose:
                    print('[{}] max values for: ||gradV|| {} V {} psi {}'.format(i, np.sqrt(np.nanmax(grad_aV2)), np.nanmax(np.abs(aV)), np.nanmax(a_psi)))

                # -logP (no regularizing prior); constants are omitted
                neg_L = nandot(n, \
                        -np.log(a_psi) + \
                        mean_dr2 / a_psi / 2 + \
                        np.sum(mean_dr * grad_aV, axis=1) + \
                        grad_aV2 / a_psi / 2)

                if verbose:
                    print('[{}] approx -logP: {}'.format(i, neg_L))
                # stopping criterion
                if np.abs(neg_L_prev - neg_L) == 0:
                    # see meanfield_dv.py for some explanation
                    break

                i += 1

        except KeyboardInterrupt:
            # an interruption at any step makes the global resolution be tagged as interrupted
            resolution = 'INTERRUPTED'
            if verbose:
                print('interrupted')
                print('D={}'.format(aD))
                print('V={}'.format(aV))
                print('L={}'.format(-neg_L))
            pass


        # 2. regularize
        A_psi = B_psi = AV = BV = None

        if mf.regularize_friction:
            if slow_neighbours:
                gradV = mf.grad(aV, s)
                gradV2 = np.nansum(gradV * gradV, axis=1)
            else:
                gradV2 = grad_aV2

            if verbose:
                print(' ** regularizing friction: ** ')

            i = 0
            neg_L = np.inf
            try:
                while True:
                    neg_L_prev = neg_L

                    A_psi, B_psi = mf.regularize_psi(a_psi, b_psi, A_psi, B_psi)
                    # include only the terms that depend on psi
                    neg_L = .5 * nandot(n, \
                                -2* np.log(A_psi) + \
                                mean_dr2 * A_psi + \
                                gradV2 / A_psi \
                                ) + \
                            mf.neg_log_prior('psi', A_psi, a_psi)
                    if verbose:
                        print('[{}] approx -logP\': {}'.format(i, neg_L))

                    # stopping criteria
                    if np.isnan(neg_L) or np.isinf(neg_L):
                        break
                    elif neg_L_prev - neg_L < -tol:
                        if verbose:
                            print('divergence')
                        break
                    elif neg_L_prev - neg_L < tol:
                        if verbose:
                            print('convergence')
                        break
                    elif max_iter and i == max_iter:
                        if verbose:
                            print('maximum iteration reached')
                        break
                    elif mf.static_landscape('psi'):
                        break
                    i += 1

            except KeyboardInterrupt:
                resolution = 'INTERRUPTED'
                if verbose:
                    print('interrupted')
                    print('psi={}'.format(A_psi))
                    print('L={}'.format(-neg_L))
                pass

        if mf.regularize_potential:
            # a_psi or A_psi?
            _psi = a_psi # do NOT overwrite `psi`

            dr_times_psi = mean_dr * _psi[:,np.newaxis]

            if verbose:
                print(' ** regularizing effective potential: ** ')

            i = 0
            neg_L = np.inf
            try:
                while True:
                    neg_L_prev = neg_L

                    AV, BV = mf.regularize_V(aV, bV, AV, BV)

                    #grad_aV = gradV_plus_VB - AV[:,np.newaxis] * B[s]
                    grad_aV = mf.grad(AV, s)
                    grad_aV2 = np.nansum(grad_aV * grad_aV, axis=1)

                    # include only the terms that depend on aV and bV (or AV and BV)
                    if False:
                        if i==0: # BV actually does not vary
                            B2_over_bV = _nan( B2[s] / BV )
                            B2_over_bV_star = np.array([ np.sum(B2_over_bV[_region]) for _region in mf.regions ])
                            log_bV = np.log(BV)
                            log_bV_star = np.sum([ np.nansum(log_bV[_region]) for _region in mf.regions ]) / 2
                        neg_L = nandot(n, \
                                    np.sum(mean_dr * grad_aV, axis=1) + \
                                    (1./A_psi + 1./(A_psi**3 * B_psi)) * \
                                        (grad_aV2 + B2_over_bV_star) / 2) + \
                                log_bV_star + \
                                mf.neg_log_prior('V', AV, aV)

                        if verbose:
                            print('[{}] approx -L(q): {}'.format(i, neg_L))
                    else:
                        neg_L = nandot(n, \
                                    np.sum(mean_dr * grad_aV, axis=1) + grad_aV2 / _psi / 2 ) + \
                                mf.neg_log_prior('V', AV, aV)
                        if verbose:
                            print('[{}] approx -logP\': {}'.format(i, neg_L))

                    # stopping criterion
                    if np.isnan(neg_L) or np.isinf(neg_L):
                        break
                    elif neg_L_prev - neg_L < -tol:
                        if verbose:
                            print('divergence')
                        break
                    elif neg_L_prev - neg_L < tol:
                        if verbose:
                            print('convergence')
                        break
                    elif max_iter and i == max_iter:
                        if verbose:
                            print('maximum iteration reached')
                        break
                    elif mf.static_landscape('V'):
                        break

                    i += 1

            except KeyboardInterrupt:
                resolution = 'INTERRUPTED'
                if verbose:
                    print('interrupted')
                    print('V={}'.format(AV))
                    print('L={}'.format(-neg_L))
                pass

        if A_psi is not None:
            a_psi, b_psi = A_psi, B_psi
        if AV is not None:
            aV, bV = AV, BV

        aVs.append(aV)
        bVs.append(bV)
        a_psis.append(a_psi)
        b_psis.append(b_psi)

        grad_aVs.append(grad_aV)
        grad_aV2s.append(grad_aV2)

    # merge
    aV = merge_a(aVs)
    bV = merge_b(bVs)
    a_psi = merge_a(a_psis, '>')
    b_psi = merge_b(b_psis)

    neg_L, _ctr = 0., 0
    for _psi,_V,_gradV,_gradV2 in zip(a_psis, aVs, grad_aVs, grad_aV2s):
        neg_L += nandot(n, \
                    log(2*pi) - np.log(_psi) + \
                    mean_dr2 * _psi / 2 + \
                    np.sum(mean_dr * _gradV, axis=1) + \
                    _gradV2 / _psi / 2)
        if mf.regularize_friction:
            neg_L += mf.neg_log_prior('psi', _psi)
        if mf.regularize_potential:
            neg_L += mf.neg_log_prior('V', _V)
        _ctr += 1
    neg_L /= _ctr


    gamma = a_psi * (2 * dt)
    V = aV * 2

    if np.any(V<0):
        V -= np.nanmin(V)

    FV = pd.DataFrame(np.stack((gamma, V), axis=1), index=index, \
        columns=['friction', 'potential'])

    info = dict(resolution=resolution, log_likelyhood=-neg_L)

    return FV, info


__all__ = ['setup', 'FrictionPotential', 'infer_meanfield_friction_potential']

