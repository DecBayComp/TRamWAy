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
from .gradient import setup_with_grad_arguments
from .meanfield import nandot, unify_priors, DiffusivityPotential
from math import *
import numpy as np
import pandas as pd
from collections import OrderedDict
from warnings import warn


setup = {'name': ('meanfield.dv', 'meanfield diffusivity,potential'),
        'provides': ('dv', 'diffusivity,potential'),
        'arguments': OrderedDict((
            #('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
            ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
            ('potential_prior',     ('-v', dict(type=float, help='prior on the potential energy'))),
            ('diffusivity_time_prior',   ('--time-d', dict(type=float, help='prior on the temporal variations of the diffusivity'))),
            ('potential_time_prior',     ('--time-v', dict(type=float, help='prior on the temporal variations of the potential'))),
            ('verbose',         ()))),
        'default_grad':     'onesided_gradient',
        'default_rgrad':    'delta0'}
setup_with_grad_arguments(setup)


def _nan(x):
    """modifies in-place and returns `x` itself."""
    x[np.isnan(x)] = 0.
    return x

def merge_a(_as, sign=None, eps=None):
    if not _as[1:]:
        return _as[0]
    _as = np.stack(_as, axis=-1)
    _undefined = np.isnan(_as)
    _as[_undefined] = 0.
    #if sign:
    #    _undefined |= (_as < eps) | (1./eps < _as)
    #else:
    #    _undefined |= (_as < -1./eps) | (1./eps < _as)
    #_as[_undefined] = 0.
    _n = np.sum(~_undefined, axis=1)
    _undefined = _n == 0
    _n[_undefined] = 1. # any value other than 0
    _a = np.sum(_as, axis=1) / _n
    _a[_undefined] = np.nan
    return _a
def merge_b(_bs, eps=None):
    if not _bs[1:]:
        return _bs[0]
    _bs = np.stack(_bs, axis=-1)
    _undefined = np.isnan(_bs)# | np.isinf(_bs)
    #_bs[_undefined] = 0.
    #_undefined |= (_bs < eps) | (1./eps < _bs)
    _bs[_undefined] = np.inf
    _sum_inv = np.sum(1. / _bs, axis=1)
    _n = np.sum(~_undefined, axis=1)
    _defined = 0 < _n
    _sum_inv, _n = _sum_inv[_defined], _n[_defined]
    _b = np.full(_bs.shape[0], np.nan, dtype=_bs.dtype)
    _b[_defined] = (_n * _n) / _sum_inv
    return _b


def cond(a):
    _bound = 1e8
    return np.maximum(-_bound, np.minimum(a, _bound))


def infer_meanfield_DV(cells, diffusivity_spatial_prior=None, potential_spatial_prior=None,
        diffusivity_time_prior=None, potential_time_prior=None,
        diffusivity_prior=None, potential_prior=None,
        diffusion_prior=None, diffusion_spatial_prior=None, diffusion_time_prior=None,
        dt=None, tol=1e-6, max_iter=1e4, eps=1e-3, verbose=False, aD_formula='exact', **kwargs):
    """
    """
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)

    # aliases
    if diffusivity_prior is None:
        diffusivity_prior = diffusion_prior
    if diffusivity_spatial_prior is None:
        diffusivity_spatial_prior = diffusion_spatial_prior
    if diffusivity_time_prior is None:
        diffusivity_time_prior = diffusion_time_prior

    mf = DiffusivityPotential(cells, dt,
            unify_priors(diffusivity_prior, diffusivity_spatial_prior, diffusivity_time_prior),
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
    approx_aD_constant_factor = 1. / (4. * dt)
    bD_constant_factor = n / (2. * dt)
    exact_aD_constant_factor_1 = mean_dr2 / (2. * dt)
    exact_aD_constant_factor_2 = exact_aD_constant_factor_1 / (2. * dt)
    bV_constant_factor = {s: B2[s] * bD_constant_factor for s in B2}

    if aD_formula.startswith('exact'):
        exact_aD_formula = True
        approximate_aD_formula = aD_formula.endswith('+')
    elif aD_formula.startswith('approx'):
        approximate_aD_formula = True
        exact_aD_formula = aD_formula.endswith('+')
    else:
        raise ValueError("aD_formula='{}' not supported".format(aD_formula))

    # initial values
    chi2_over_n = mean_dr2 - np.sum(mean_dr * mean_dr, axis=1)
    D = approx_aD_constant_factor * chi2_over_n#mean_dr2
    #V = np.zeros_like(D)
    V = V_initial

    fixed_neighbours = False
    slow_neighbours = True

    resolution = None
    aVs, bVs, aDs, bDs = [], [], [], []
    grad_aVs, grad_aV2s = [], []
    for s in mf.gradient_sides:

        # 1. adjust D and V jointly with no regularization
        aD, aV = D, V

        if verbose:
            if mf.gradient_sides[1:]:
                print(' **          gradient side: \'{}\'          ** '.format(s))
            print(' ** preliminary fit (no regularization): ** ')

        i = 0
        neg_L = np.inf
        try:
            while True:
                # depends on the implementation
                neg_L_prev = neg_L

                # new aV
                if not fixed_neighbours or i==0:
                    gradV_plus_VB = _nan(mf.diff_grad(aV, s))
                aV = np.sum(  Bstar[s] * (gradV_plus_VB + cond( mean_dr / aD[:,np.newaxis] ))  ,axis=1)
                if slow_neighbours:
                    grad_aV = gradV_plus_VB - aV[:,np.newaxis] * B[s]
                else:
                    grad_aV = mf.grad(aV, s)
                grad_aV2 = np.nansum(grad_aV * grad_aV, axis=1)

                # new aD
                if approximate_aD_formula:
                    aD2 = aD * aD
                    chi2_over_n = mean_dr2 + 2 * aD * np.sum(mean_dr * _nan(grad_aV), axis=1) + aD2 * grad_aV2
                    approx_aD = approx_aD_constant_factor * chi2_over_n
                if exact_aD_formula:
                    sqrt_discr = np.sqrt( 1. + exact_aD_constant_factor_2 * grad_aV2 )
                    exact_aD = exact_aD_constant_factor_1 / (1. + sqrt_discr)
                aD = approx_aD if aD_formula.startswith('approx') else exact_aD

                # new bV and bD
                bV = aD * bV_constant_factor[s]
                inv_aD = cond( 1./ aD ) # what if aD~0?
                inv_aD2 = inv_aD * inv_aD
                bD = n * inv_aD2 + bD_constant_factor * grad_aV2 * inv_aD

                if verbose:
                    if aD_formula.endswith('+'):
                        print('[{}] max aD difference: {}'.format(i, np.max(np.abs(exact_aD-approx_aD))))
                    print('[{}] max values for: ||gradV|| {:.2f} aV:bV {:.2f}:{:.2f} aD:bD {:.2f}:{:.2f}'.format(i, np.sqrt(np.nanmax(grad_aV2)), np.nanmax(np.abs(aV)), np.nanmax(bV), np.nanmax(aD), np.nanmax(bD)))

                if False: # -L(q)
                    inv_aD2_bD = inv_aD2 / bD
                    inv_aD3_bD = inv_aD2_bD * inv_aD

                    B2_over_bV = B2[s] / bV
                    B2_over_bV_star = np.array([ np.sum(B2_over_bV[_region]) for _region in mf.regions ])
                    log_bV = np.log(bV)
                    log_bV_star = np.sum([ np.nansum(log_bV[_region]) for _region in mf.regions ])
                    neg_L = (nandot(n, \
                                2 * np.log(aD) - inv_aD2_bD + \
                                1./ (2 * dt) * ( \
                                    (inv_aD + inv_aD3_bD) * mean_dr2 + \
                                    2 * np.sum(grad_aV * mean_dr, axis=1) + \
                                    aD * (grad_aV2 + B2_over_bV_star) \
                                    )) + \
                            np.nansum( np.log(bD) ) + log_bV_star) / 2

                    if verbose:
                        print('[{}] approx -L(q): {}'.format(i, neg_L))
                    # stopping criterion
                    if neg_L_prev - neg_L < tol:
                        break
                else: # -logP
                    neg_L = nandot(n, \
                                np.log( aD * dt ) + \
                                1./(4* dt) * ( \
                                    cond( mean_dr2 / aD ) + \
                                    2* np.sum(mean_dr * grad_aV, axis=1) + \
                                    grad_aV2 * aD \
                                ))

                    if verbose:
                        print('[{}] approx -logP: {}'.format(i, neg_L))
                    # stopping criterion
                    if np.abs(neg_L_prev - neg_L) == 0:
                        # important note: the effective potential values are scaled to their proper range;
                        #                 this happens from a corner to the opposite corner and generates a propagating frontier
                        #                 between two plateau;
                        #                 when this diagonal frontier includes many bins, the cost function begins to increase;
                        #                 this should not stop the algorithm to update more effective potential parameters;
                        #                 only full convergence is a reliable stopping criterion here
                        break
                i += 1

        except KeyboardInterrupt:
            # an interruption at any step makes the global resolution be tagged as interrupted
            resolution = 'INTERRUPTED'
            if verbose:
                print('interrupted')
                print('D={}'.format(aD))
                print('V={}'.format(aV))
                try:
                    print('L={}'.format(-neg_L))
                except:
                    print('L={}'.format(-neg_L))
            pass


        # 2. regularize
        AD = BD = AV = BV = None

        if mf.regularize_diffusivity:
            if slow_neighbours:
                gradV = mf.grad(aV, s)
                gradV2 = np.nansum(gradV * gradV, axis=1)
            else:
                gradV2 = grad_aV2

            if verbose:
                print(' ** regularizing diffusivity: ** ')

            i = 0
            neg_L = np.inf
            try:
                while True:
                    neg_L_prev = neg_L

                    AD, BD = mf.regularize_D(aD, bD, AD, BD)
                    # include only the terms that depend on D
                    neg_L = nandot(n, \
                                np.log( AD ) + \
                                ( cond( mean_dr2 / AD ) + gradV2 * AD ) / (4*dt) \
                                ) + \
                            mf.neg_log_prior('D', AD, aD)
                    if verbose:
                        print('[{}] approx -logP\': {}'.format(i, neg_L))

                    # stopping criteria
                    if np.isnan(neg_L) or np.isinf(neg_L):
                        break
                    elif neg_L_prev - neg_L < -1e2*tol:
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
                    elif mf.static_landscape('D'):
                        break
                    i += 1

            except KeyboardInterrupt:
                resolution = 'INTERRUPTED'
                if verbose:
                    print('interrupted')
                    print('D={}'.format(AD))
                    print('L={}'.format(-neg_L))
                pass

        if mf.regularize_potential:
            # aD or AD?
            _D = aD # do NOT overwrite `D`

            dr_over_D = cond( mean_dr / _D[:,np.newaxis] )

            if verbose:
                print(' ** regularizing effective potential: ** ')

            i = 0
            neg_L = np.inf
            try:
                while True:
                    neg_L_prev = neg_L

                    AV, BV = mf.regularize_V(aV, bV, AV, BV)

                    #grad_aV = gradV_plus_VB - AV[:,np.newaxis] * B[s] # constant neighbours
                    #grad_aV = mf.diff_grad(AV, s) - aV[:,np.newaxis] * B[s] # neighbours only vary
                    grad_aV = mf.grad(AV, s) # all vary
                    grad_aV2 = np.nansum(grad_aV * grad_aV, axis=1)

                    # include only the terms that depend on aV and bV (or AV and BV)
                    if False:
                        if i==0: # BV actually does not vary
                            B2_over_bV = _nan( B2[s] / BV )
                            B2_over_bV_star = np.array([ np.sum(B2_over_bV[_region]) for _region in mf.regions ])
                            log_bV = np.log(BV)
                            log_bV_star = np.sum([ np.nansum(log_bV[_region]) for _region in mf.regions ]) / 2
                        neg_L = nandot(n / (4 * dt), \
                                    2 * np.sum(mean_dr * grad_aV, axis=1) + \
                                    cond(1./aD + 1./(aD**3 * bD)) * (grad_aV2 + B2_over_bV_star)) + \
                                log_bV_star + \
                                mf.neg_log_prior('V', AV, aV)

                        if verbose:
                            print('[{}] approx -L(q): {}'.format(i, neg_L))
                    else:
                        neg_L = nandot(n / (2* dt), \
                                    np.sum(mean_dr * grad_aV, axis=1) + grad_aV2 * _D / 2 ) + \
                                mf.neg_log_prior('V', AV, aV)
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

        if AD is not None:
            aD, bD = AD, BD
        if AV is not None:
            aV, bV = AV, BV

        aVs.append(aV)
        bVs.append(bV)
        aDs.append(aD)
        bDs.append(bD)

        grad_aVs.append(grad_aV)
        grad_aV2s.append(grad_aV2)

    # merge
    aV = merge_a(aVs)
    bV = merge_b(bVs)
    aD = merge_a(aDs, '>')
    bD = merge_b(bDs)

    neg_L, _ctr = 0., 0
    for _D,_V,_gradV,_gradV2 in zip(aDs, aVs, grad_aVs, grad_aV2s):
        neg_L += nandot(n, \
                    np.log( 4*pi* _D * dt ) + \
                    1./(4* dt) * ( \
                        mean_dr2 / _D + \
                        2* np.sum(mean_dr * _gradV, axis=1) + \
                        _gradV2 * _D \
                    ))
        if mf.regularize_diffusivity:
            neg_L += mf.neg_log_prior('D', _D)
        if mf.regularize_potential:
            neg_L += mf.neg_log_prior('V', _V)
        _ctr += 1
    neg_L /= _ctr


    D = aD
    V = aV / dt

    if np.any(V<0):
        V -= np.nanmin(V)

    DV = pd.DataFrame(np.stack((D, V), axis=1), index=index, \
        columns=['diffusivity', 'potential'])

    # derivate the forces
    #index_, F = [], []
    #for i in index:
    #    gradV = mf.local_grad(i, V, mf.gradient_sides[0])#cells.grad(i, V, reverse_index, **grad_kwargs)
    #    if gradV is not None:
    #        index_.append(i)
    #        F.append(-gradV)
    #if F:
    #    F = pd.DataFrame(np.stack(F, axis=0), index=index_, \
    #        columns=[ 'force ' + col for col in cells.space_cols ])
    #else:
    #    warn('not any cell is suitable for evaluating the local force', RuntimeWarning)
    #    F = pd.DataFrame(np.zeros((0, len(cells.space_cols)), dtype=V.dtype), \
    #        columns=[ 'force ' + col for col in cells.space_cols ])
    #DVF = DV.join(F)

    info = dict(resolution=resolution, log_likelyhood=-neg_L)

    return DV, info


__all__ = ['setup', 'DiffusivityPotential', 'infer_meanfield_DV']

