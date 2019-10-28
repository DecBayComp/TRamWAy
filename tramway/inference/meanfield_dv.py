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


def infer_meanfield_DV(cells, diffusivity_spatial_prior=None, potential_spatial_prior=None,
        diffusivity_time_prior=None, potential_time_prior=None,
        diffusivity_prior=None, potential_prior=None,
        diffusion_prior=None, diffusion_spatial_prior=None, diffusion_time_prior=None,
        dt=None, tol=1e-6, verbose=True, **kwargs):
    """
    """
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)
    index, reverse_index, n, dt_mean, _, _, _, _ = \
        smooth_infer_init(cells, sigma2=0)#localization_error)
    if dt is None:
        dt = dt_mean
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

    index = np.array(index)
    ok = 1<n
    print(ok.size, np.sum(ok))
    if not np.all(ok):
        reverse_index[index] -= np.cumsum(~ok)
        reverse_index[index[~ok]] = -1
        index, n, dt = index[ok], n[ok], dt[ok]

    grad_kwargs = get_grad_kwargs(kwargs)
    sides = grad_kwargs.pop('side', '<>')
    if sides in ('<','>'):
        grad_kwargs['side'] = sides

    while True:
        ok, all_ok = np.ones(index.size, dtype=bool), True
        for k, i in enumerate(index):
            _grad = cells.grad(i, dt, reverse_index, **grad_kwargs)
            if _grad is None:
                ok[k] = all_ok = False
        print(np.sum(ok))
        if all_ok:
            break
        else:
            reverse_index[index] -= np.cumsum(~ok)
            reverse_index[index[~ok]] = -1
            index, n, dt = index[ok], n[ok], dt[ok]
        cells.clear_caches()
    if index.size == 0:
        raise ValueError('no valid cell')

    #grad_kwargs['na'] = 0.
    def grad(i, x, s):
        grad_kwargs['side'] = s
        _grad = cells.grad(i, x, reverse_index, **grad_kwargs)
        if _grad is None:
            _grad = np.full(dim, 0.)
        return _grad
    _one = np.zeros(n.size, dtype=dtype) # internal
    def diff_grad(x, s):
        _dx = []
        _x = np.array(x) # copy
        for k in range(x.size):
            _x[k] = 0
            _dx.append(grad(index[k], _x, s))
            _x[k] = x[k]
        return _dx

    mean_dr, sum_dr2, chi2 = [], [], []
    Bstar, B2 = {s: [] for s in sides}, {s: [] for s in sides}
    C_neighbours, C = [], []
    regions = []

    for k, i in enumerate(index):
        # local variables
        _dr = np.mean(cells[i].dr, axis=0, keepdims=True)
        mean_dr.append(_dr)
        sum_dr2.append(np.sum(cells[i].dr * cells[i].dr))
        _dr = cells[i].dr - _dr
        chi2.append(np.sum(_dr * _dr))
        # spatial gradients
        _one[k] = 1
        for s in sides:
            _B = grad(i, _one, s)
            _B2 = np.dot(_B, _B)
            if _B2 == 0:
                raise ValueError('cannot compute finite differences')
            Bstar[s].append(_B / _B2)
            B2[s].append(_B2)
        _one[k] = 0
        # spatial smoothing
        _neighbours = cells.neighbours(i)
        __neighbours = reverse_index[_neighbours]
        C_neighbours.append(__neighbours)
        _dr = np.stack([ cells[j].center for j in _neighbours ], axis=0) - cells[i].center[np.newaxis,:]
        C.append(1./np.sum(_dr*_dr,axis=1)/len(_dr))
        regions.append(np.r_[k,__neighbours])

    mean_dr = np.vstack(mean_dr)
    sum_dr2 = np.array(sum_dr2)
    mean_dr2 = sum_dr2 / n
    chi2_over_n = np.array(chi2) / n
    Bstar = {s: np.vstack(Bstar[s]) for s in Bstar}
    B2 = {s: np.array(B2[s]) for s in B2}

    aD_scale = 1. / (4. * dt)
    bD_scale = n / (2. * dt)
    bV_scale = {s: B2[s] * bD_scale for s in B2}
    nsides = len(sides)
    D = aD_scale * chi2_over_n
    V = np.zeros_like(D)

    neg_L_global_constant = np.dot(n, np.log(4 * pi * dt))
    L_local_factor_for_D = aD_scale * sum_dr2
    L_local_factor_for_V = bD_scale[:,np.newaxis] * mean_dr
    L_local_factor_for_D_times_V = aD_scale * n

    gradV, gradV2 = {}, {}
    #for s in sides:
    #    gradV[s] = np.vstack([ grad(i, V, s) for i in index ])
    #    gradV2[s] = np.sum(gradV * gradV, axis=1)

    # priors
    sumC = np.array([ np.sum(_C) for _C in C ]) # used for both D and V
    # do not scale time;
    # assume constant window shift and let time_prior hyperparameters bear all the responsibility
    if diffusivity_spatial_prior is None:
        BD_constant_term = 0.
    else:
        BD_constant_term = 2. * diffusivity_spatial_prior * sumC
    if diffusivity_time_prior is not None:
        BD_constant_term += 2. * diffusivity_time_prior
    if potential_spatial_prior is None:
        BV_constant_term = 0.
    else:
        BV_constant_term = 2. * potential_spatial_prior * sumC
    if potential_time_prior is not None:
        BV_constant_term += 2. * potential_time_prior

    neg_L = np.inf
    try:
        while True:
            neg_L_prev = neg_L

            dr_over_D = mean_dr / D[:,np.newaxis]

            aVs, bVs, aDs, bDs, neg_Ls = [], [], [], [], []
            for s in sides:

                gradV[s] = _gradV = np.vstack([ grad(i, V, s) for i in index ])
                gradV2[s] = np.sum(_gradV * _gradV, axis=1)

                aV = -np.sum(  Bstar[s] * (diff_grad(V, s) + dr_over_D)  ,axis=1)
                bV = D * bV_scale[s]

                D2 = D * D
                chi2_over_n = mean_dr2 + 2 * D * np.sum(mean_dr * gradV[s], axis=1) + D2 * gradV2[s]

                aD = chi2_over_n * aD_scale
                bD = n / D2 + bD_scale * gradV2[s] / D

                # priors
                BD = bD + BD_constant_term
                BV = bV + BV_constant_term
                AD = aD * (bD / 2.)
                AV = aV * (bV / 2.)
                if diffusivity_spatial_prior is not None:
                    # reminder: C comes without mu (global diffusion prior) or lambda (global potential prior)
                    AD += diffusivity_spatial_prior * \
                            np.array([ np.dot(aD[_neighbours], _C) for _neighbours, _C in zip(C_neighbours, C) ])
                    AV += potential_spatial_prior * \
                            np.array([ np.dot(aV[_neighbours], _C) for _neighbours, _C in zip(C_neighbours, C) ])
                if diffusivity_time_prior is not None:
                    aD_neighbours, aV_neighbours = [], []
                    for i in index:
                        _neighbours = reverse_index[cells.time_neighbours(i)]
                        if _neighbours.size == 0:
                            aD_neighbours.append(0.)
                            aV_neighbours.append(0.)
                        else:
                            aD_neighbours.append( np.mean(aD[_neighbours]) )
                            aV_neighbours.append( np.mean(aV[_neighbours]) )
                    AD += diffusivity_time_prior * np.array(aD_neighbours)
                    AV += potential_time_prior * np.array(aV_neighbours)
                AD /= BD / 2.
                AV /= BV / 2.
                aD, bD, aV, bV = AD, BD, AV, BV

                # ELBO
                grad_aV = np.vstack([ grad(i, aV, s) for i in index ])
                grad_aV2 = np.sum(grad_aV * grad_aV, axis=1)
                B2_over_bV = B2[s] / bV
                B2_over_bV_star = np.array([ np.sum(B2_over_bV[_region]) for _region in regions ])
                log_bV = np.log(bV)
                log_bV_star = np.array([ np.sum(log_bV[_region]) for _region in regions ])

                #neg_L = neg_L_global_constant + \
                # let us ignore constants
                neg_L = \
                        np.dot(n, np.log(aD) - 1./ (2. * aD * aD * bD)) + \
                        np.dot(1./ aD + 1./ (aD**3 * bD), L_local_factor_for_D) + \
                        np.sum(L_local_factor_for_V * grad_aV) + \
                        np.dot(L_local_factor_for_D_times_V * aD, grad_aV2 + B2_over_bV_star) + \
                        .5 * np.sum(np.log(bD) + log_bV_star)

                aVs.append(aV)
                bVs.append(bV)
                aDs.append(aD)
                bDs.append(bD)
                #gradV[s] = grad_aV # should merge before
                #gradV2[s] = grad_aV2
                neg_Ls.append(neg_L)

            # merge
            aV = np.mean(np.stack(aVs, axis=-1), axis=1)
            bV = (nsides*nsides) / np.sum( 1./ np.stack(bVs,axis=-1) ,axis=1)
            aD = np.mean(np.stack(aDs, axis=-1), axis=1)
            bD = (nsides*nsides) / np.sum( 1./ np.stack(bDs,axis=-1) ,axis=1)
            neg_L = np.mean(neg_Ls)

            # step forward
            D, V = aD, aV
            # stopping criterion
            if abs(neg_L - neg_L_prev) < tol:
                break

    except KeyboardInterrupt:
        print('interrupted')
        print('D={}'.format(D))
        print('V={}'.format(V))
        print('L=-{}'.format(neg_L + neg_L_global_constant))
        pass

    neg_L += neg_L_global_constant
    V /= dt

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

    return DVF


__all__ = ['setup', 'infer_meanfield_DV']

