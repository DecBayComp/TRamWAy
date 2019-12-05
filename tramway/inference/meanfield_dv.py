# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
from .gradient import get_grad_kwargs, setup_with_grad_arguments
from math import *
import numpy as np
import pandas as pd
from collections import OrderedDict
from warnings import warn


setup = {'name': 'meanfield.dv',
        'provides': 'dv',
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


class ConvergenceWarning(UserWarning):
    pass


def infer_meanfield_DV(cells, diffusivity_spatial_prior=None, potential_spatial_prior=None,
        diffusivity_time_prior=None, potential_time_prior=None,
        diffusivity_prior=None, potential_prior=None,
        diffusion_prior=None, diffusion_spatial_prior=None, diffusion_time_prior=None,
        dt=None, tol=1e-6, eps=1e-3, verbose=False, **kwargs):
    """
    """
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)
    index, reverse_index, n, dt_mean, _, _, _, _ = \
        smooth_infer_init(cells, sigma2=0)#localization_error)
    if dt is None:
        dt = dt_mean
        #assert np.allclose(dt, dt[0])
    elif np.isscalar(dt):
        dt = np.full_like(dt_mean, dt)
    dtype = dt.dtype
    dim = cells.dim

    if diffusivity_prior is None:
        diffusivity_prior = diffusion_prior
    if diffusivity_spatial_prior is None:
        if diffusion_spatial_prior is None:
            diffusivity_spatial_prior = diffusivity_prior
        else:
            diffusivity_spatial_prior = diffusion_spatial_prior
    if potential_spatial_prior is None:
        potential_spatial_prior = potential_prior
    if diffusivity_time_prior is None:
        diffusivity_time_prior = diffusion_time_prior
    reg = diffusivity_spatial_prior or diffusivity_time_prior or potential_spatial_prior or potential_time_prior
    if reg:
        if diffusivity_time_prior or potential_time_prior:
            warn('temporally regularized meanfield DV may not converge; use manual interruption', ConvergenceWarning)
    else:
        warn('meanfield DV is known to fail without regularization', ConvergenceWarning)

    index = np.array(index)
    ok = 1<n
    #print(ok.size, np.sum(ok))
    if not np.all(ok):
        reverse_index[index] -= np.cumsum(~ok)
        reverse_index[index[~ok]] = -1
        index, n, dt = index[ok], n[ok], dt[ok]

    if kwargs.get('grad_selection_angle', None) is None:
        kwargs['grad_selection_angle'] = .5
    grad_kwargs = get_grad_kwargs(kwargs)
    #sides = grad_kwargs.get('side', '<>')
    sides = grad_kwargs.get('side', '>')

    #while True:
    #    ok, all_ok = np.ones(index.size, dtype=bool), True
    #    for i in index:
    #        cells.grad(i, dt, reverse_index, **grad_kwargs)
    #        k, _neighbours_lt, _, _neighbours_gt, _ = cells[i].cache['onesided_gradient']
    #        if sides == '<':
    #            if _neighbours_lt is None:
    #                ok[k] = all_ok = False
    #        elif sides == '>':
    #            if _neighbours_gt is None:
    #                ok[k] = all_ok = False
    #        elif _neighbours_lt is None and _neighbours_gt is None:
    #            ok[k] = all_ok = False
    #    #print(np.sum(ok))
    #    if all_ok:
    #        break
    #    else:
    #        reverse_index[index] -= np.cumsum(~ok)
    #        reverse_index[index[~ok]] = -1
    #        index, n, dt = index[ok], n[ok], dt[ok]
    #    cells.clear_caches()
    if index.size == 0:
        raise ValueError('no valid cell')

    D_spatial_prior, D_time_prior = diffusivity_spatial_prior, diffusivity_time_prior
    V_spatial_prior = None if potential_spatial_prior is None else potential_spatial_prior * dt
    V_time_prior = None if potential_time_prior is None else potential_time_prior * dt

    def f_(_x, sign=None, invalid=0., inplace=True):
        _valid = ~(np.isnan(_x) | np.isinf(_x))
        #if sign:
        #    _valid[_valid] = (eps < _x[_valid]) & (_x[_valid] < 1./eps)
        #else:
        #    _valid[_valid] = (-1./eps < _x[_valid]) & (_x[_valid] < 1./eps)
        __x = _x if inplace else np.array(_x) # copy
        __x[~_valid] = invalid
        return __x

    grad_kwargs['na'] = 0.#np.nan
    def local_grad(i, x, s):
        grad_kwargs['side'] = s
        return cells.grad(i, x, reverse_index, **grad_kwargs)
    def grad(x, s):
        return np.stack([ local_grad(i, x, s) for i in index ])
    _one = np.zeros(n.size, dtype=dtype) # internal
    def diff_grad(x, s):
        _x = np.array(x) # copy
        _dx = []
        for k in range(x.size):
            _x[k] = 0
            _dx.append(local_grad(index[k], _x, s))
            _x[k] = x[k]
        return _dx

    mean_dr, sum_dr2 = [], []
    Bstar, B2 = {s: [] for s in sides}, {s: [] for s in sides}
    C_neighbours, C = [], []
    Ct_neighbours, Ct = [], []
    regions = []
    Z, Zt = [], []

    for k, i in enumerate(index):
        # local variables
        mean_dr.append(np.mean(cells[i].dr, axis=0, keepdims=True))
        sum_dr2.append(np.sum(cells[i].dr * cells[i].dr))
        # spatial gradients
        _one[k] = 1
        for s in sides:
            _B = local_grad(i, _one, s)
            _valid = ~np.isnan(_B)
            if np.all(np.isnan(_B)):
                _B2 = 0.
            else:
                if np.any(np.isnan(_B)):
                    _B[np.isnan(_B)] = 0.
                _B2 = np.dot(_B, _B)
            if _B2 == 0:
                _B_over_B2 = np.zeros_like(_B)
            else:
                _B_over_B2 = _B / _B2
            B2[s].append(_B2)
            Bstar[s].append(_B_over_B2)
        _one[k] = 0
        # spatial smoothing
        _neighbours = cells.neighbours(i)
        if _neighbours.size:
            _dr = np.stack([ cells[j].center for j in _neighbours ], axis=0) - cells[i].center[np.newaxis,:]
            _C = 1. / np.sum(_dr*_dr,axis=1) / len(_dr)
            _neighbours = reverse_index[_neighbours]
            _Z = np.sum(_C)
        else:
            _C = _neighbours = None
            _Z = 0.
        C.append(_C)
        C_neighbours.append(_neighbours)
        Z.append(_Z)
        regions.append(np.r_[k,_neighbours])
        # temporal smoothing
        _neighbours = cells.time_neighbours(i)
        if _neighbours.size:
            _dt = np.array([ cells[j].center_t for j in _neighbours ]) - cells[i].center_t
            _Ct = 1. / (_dt*_dt) / len(_dt)
            _neighbours = reverse_index[_neighbours]
            _Zt = np.sum(_Ct)
        else:
            _Ct = _neighbours = None
            _Zt = 0.
        Ct.append(_Ct)
        Ct_neighbours.append(_neighbours)
        Zt.append(_Zt)

    mean_dr = np.vstack(mean_dr)
    sum_dr2 = np.array(sum_dr2)
    mean_dr2 = sum_dr2 / n
    chi2_over_n = mean_dr2 - np.sum(mean_dr * mean_dr, axis=1)
    Bstar = {s: np.vstack(Bstar[s]) for s in Bstar}
    B2 = {s: np.array(B2[s]) for s in B2}

    C, Ct = list(zip(C_neighbours, C)), list(zip(Ct_neighbours, Ct))
    Z, Zt = np.array(Z), np.array(Zt)

    def background(_a, _C, sign=None):
        return np.array([
                0. if __neighbours is None else np.dot(f_(_a[__neighbours], sign), __C)
                for __neighbours, __C in _C
            ])

    # constants
    aD_constant_factor = 1. / (4. * dt)
    bD_constant_factor = n / (2. * dt)
    alt_aD_constant_factor_1 = mean_dr2 / (2. * dt)
    alt_aD_constant_factor_2 = alt_aD_constant_factor_1 / (2. * dt)
    bV_constant_factor = {s: B2[s] * bD_constant_factor for s in B2}

    # initial values
    D = aD_constant_factor * mean_dr2#chi2_over_n
    V = np.zeros_like(D)

    neg_L_global_constant = np.dot(n, np.log(4 * pi * dt))
    L_local_factor_for_D = aD_constant_factor * sum_dr2
    L_local_factor_for_V = bD_constant_factor[:,np.newaxis] * mean_dr
    L_local_factor_for_D_times_V = aD_constant_factor * n

    # priors
    if reg:
        BD_constant_term = BV_constant_term = 0.
        if diffusivity_spatial_prior is not None:
            BD_constant_term += 2 * D_spatial_prior * Z
        if diffusivity_time_prior is not None:
            BD_constant_term += 2 * D_time_prior * Zt
        if potential_spatial_prior is not None:
            BV_constant_term += 2 * V_spatial_prior * Z
        if potential_time_prior is not None:
            BV_constant_term += 2 * V_time_prior * Zt

    def merge_a(_as, sign=None):
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
    def merge_b(_bs):
        _bs = np.stack(_bs, axis=-1)
        _undefined = np.isnan(_bs)# | np.isinf(_bs)
        #_bs[_undefined] = 0.
        #_undefined |= (_bs < eps) | (1./eps < _bs)
        _bs[_undefined] = np.inf
        _sum_inv = np.sum(1. / _bs, axis=1)
        _n = np.sum(~_undefined, axis=1)
        _defined = 0 < _n
        _sum_inv, _n = _sum_inv[_defined], _n[_defined]
        _b = np.full(_bs.shape[0], np.nan, dtype=dtype)
        _b[_defined] = (_n * _n) / _sum_inv
        return _b

    gradV, gradV2 = {}, {}
    resolution = None
    neg_L = np.inf
    try:
        while True:
            neg_L_prev = neg_L

            dr_over_D = mean_dr / D[:,np.newaxis]

            aVs, bVs, aDs, bDs, neg_Ls = [], [], [], [], []
            for s in sides:

                gradV[s] = _gradV = grad(V, s)
                gradV2[s] = np.sum(_gradV * _gradV, axis=1)
                if verbose:
                    print('max values for side:', s, '||gradV||', np.sqrt(np.nanmax(gradV2[s])), 'V', np.nanmax(np.abs(V)), 'D', np.nanmax(D))

                aV = np.sum(  Bstar[s] * (diff_grad(V, s) + dr_over_D)  ,axis=1)
                bV = D * bV_constant_factor[s]

                D2 = D * D
                #chi2_over_n = mean_dr2 + 2 * D * np.sum(mean_dr * gradV[s], axis=1) + D2 * gradV2[s]

                #aD = chi2_over_n * aD_constant_factor

                sqrt_discr = np.sqrt( 1. + alt_aD_constant_factor_2 * gradV2[s] )
                aD = alt_aD_constant_factor_1 / (1. + sqrt_discr)

                if verbose: # debug
                    # a few assertions (to be removed in the future)
                    roots = alt_aD_constant_factor_1[:,np.newaxis] / np.stack(( 1. - sqrt_discr, 1. + sqrt_discr ), axis=1)
                    roots[roots<=0] = np.nan
                    root_count = np.sum(~np.isnan(roots), axis=1)
                    #print(np.stack(np.unique(root_count,return_counts=True),axis=1))
                    multiple_roots, = (1<root_count).nonzero()
                    opt_aD = roots[multiple_roots]
                    chi2_over_nD = mean_dr2[multiple_roots][:,np.newaxis] / opt_aD + 2 * np.sum(mean_dr[multiple_roots] * gradV[s][multiple_roots], axis=1, keepdims=True) + opt_aD
                    minus_logP_over_n = np.log(opt_aD) + aD_constant_factor[multiple_roots][:,np.newaxis] * chi2_over_nD
                    discard = np.argmax(minus_logP_over_n, axis=1)
                    roots[multiple_roots, discard] = np.nan
                    first_root, second_root = roots[:,0], roots[:,1]
                    assert np.all(np.isnan(first_root))
                    assert not np.any(np.isnan(second_root))

                bD = n / D2 + bD_constant_factor * gradV2[s] / D

                # priors
                if reg:
                    BD = bD + BD_constant_term
                    BV = bV + BV_constant_term
                    AD = aD * bD
                    AV = aV * bV
                    if diffusivity_spatial_prior is not None:
                        # reminder: C comes without mu (global diffusion prior) or lambda (global potential prior)
                        AD += 2 * D_spatial_prior * background(aD, C)
                    if potential_spatial_prior is not None:
                        AV += 2 * V_spatial_prior * background(aV, C)
                    if diffusivity_time_prior is not None:
                        AD += 2 * D_time_prior * background(aD, Ct)
                    if potential_time_prior is not None:
                        AV += 2 * V_time_prior * background(aV, Ct)
                    AD /= BD
                    AV /= BV
                    aD, bD, aV, bV = AD, BD, AV, BV

                #print(np.nanmax(aD), np.nanmax(bD), np.nanmax(np.abs(aV)), np.nanmax(bV))
                #aD = f_(aD, '>', np.nan)
                #bD = f_(bD, '>', np.nan)
                #aV = f_(aV, invalid=np.nan)
                #bV = f_(bV, '>', np.nan)

                # ELBO
                grad_aV = grad(aV, s)
                grad_aV2 = np.sum(grad_aV * grad_aV, axis=1)
                B2_over_bV = B2[s] / bV
                B2_over_bV_star = np.array([ np.sum(B2_over_bV[_region]) for _region in regions ])
                log_bV = np.log(bV)
                log_bV_star = np.array([ np.sum(log_bV[_region]) for _region in regions ])

                #neg_L = neg_L_global_constant + \
                # let us ignore constants
                neg_L = \
                        np.dot(n, f_( np.log(aD) - 1./ (2. * aD * aD * bD) )) + \
                        np.dot(f_( 1./ aD + 1./ (aD**3 * bD) ), L_local_factor_for_D) + \
                        np.sum(L_local_factor_for_V * f_( grad_aV )) + \
                        np.dot(f_( L_local_factor_for_D_times_V * aD ),f_( grad_aV2 + B2_over_bV_star )) + \
                        .5 * np.sum(f_( np.log(bD) + log_bV_star ))

                aVs.append(aV)
                bVs.append(bV)
                aDs.append(aD)
                bDs.append(bD)
                neg_Ls.append(neg_L)

            # merge
            aV = merge_a(aVs)
            bV = merge_b(bVs)
            aD = merge_a(aDs, '>')
            bD = merge_b(bDs)
            neg_L = np.nanmean(neg_Ls)

            # step forward
            D, V = aD, aV

            # stopping criterion
            if verbose:
                print('-logP approx', neg_L)
            if abs(neg_L - neg_L_prev) < tol:
                resolution = 'CONVERGENCE: DELTA -L < TOL'
                break

    except KeyboardInterrupt:
        resolution = 'INTERRUPTED'
        if verbose:
            print('interrupted')
            print('D={}'.format(D))
            print('V={}'.format(V))
            print('L={}'.format(-neg_L - neg_L_global_constant))
        pass

    neg_L += neg_L_global_constant
    V /= dt

    if np.any(V<0):
        V -= np.min(V)

    DV = pd.DataFrame(np.stack((D, V), axis=1), index=index, \
        columns=['diffusivity', 'potential'])

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
    DVF = DV.join(F)

    info = dict(resolution=resolution)

    return DVF, info


__all__ = ['setup', 'infer_meanfield_DV']

