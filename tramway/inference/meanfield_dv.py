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
from warnings import warn
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
            ('verbose',         ())))}
setup_with_grad_arguments(setup)


def infer_meanfield_DV(cells, diffusivity_spatial_prior=None, potential_spatial_prior=None,
        diffusivity_time_prior=None, potential_time_prior=None,
        diffusivity_prior=None, potential_prior=None,
        diffusion_prior=None, diffusion_spatial_prior=None, diffusion_time_prior=None,
        D0=None, V0=None, dt=None, tol=1e-6, verbose=True, **kwargs):
    """
    """
    grad_kwargs = get_grad_kwargs(kwargs)
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)
    index, reverse_index, n, dt_mean, D, _, _, border = \
        smooth_infer_init(cells, sigma2=0)#localization_error)
    if dt is None:
        dt = dt_mean
    elif np.isscalar(dt):
        dt = np.full_like(dt_mean, dt)
    # initial values
    if D0 is not None:
        if np.isscalar(D0):
            D[...] = D0
        elif D0.size == D.size:
            D = np.array(D0) # copy
        else:
            raise ValueError('wrong size for D0')
    if V0 is None:
        try:
            volume = [ cells[i].volume for i in index ]
        except:
            V = -np.log(n / np.max(n))
        else:
            density = n / np.array([ np.inf if v is None else v for v in volume ])
            density[density == 0] = np.min(density[0 < density])
            V = np.log(np.max(density)) - np.log(density)
    else:
        if np.isscalar(V0):
            # V0 may be given as an integer
            V = np.full(D.size, V0, dtype=D.dtype)
        elif V0.size == D.size:
            V = np.array(V0) # copy
        else:
            raise ValueError('wrong size for V0')

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

    neighbours = []
    Bu, B, B2 = [], [], []
    C = []
    sum_dr, sum_dr2 = [], []
    for i in index:
        _neighbours = cells.neighbours(i)
        neighbours.append(reverse_index[_neighbours])
        # r_l - r_k \forall l in N_k, with N_k the set of spatial neighbours of k
        _d = np.stack([ cells[j].center for j in _neighbours ], axis=0) - cells[i].center[np.newaxis,:]
        # C_{V_l} = \lambda / (|| r_l - r_k ||^2 * |N_k|)  let's omit lambda for now
        _C = 1./ np.sum(_d * _d, axis=1)
        _scale = np.sqrt(_C)
        _u = _d * _scale[:,np.newaxis]
        _C /= len(_neighbours)
        # B_l = 1 / (|| r_l - r_k || * |N_k|)
        _Bl = _scale / len(_neighbours)
        # B^2 = \sum_{l \in N_k} B_l^2
        _B2 = np.sum(_Bl * _Bl)
        #assert not np.any(_scale == 0) # null inter-cell distance?
        #assert not np.any(np.isinf(_scale)) # infinite inter-cell distance?
        # Bu is \{B_l u_{k->l}\}_{\forall l}
        _Bu = _u * _Bl[:,np.newaxis]
        Bu.append(_Bu)
        B2.append(_B2)
        _B = np.sum(_Bu, axis=0, keepdims=True)
        B.append(_B)
        C.append(_C)
        sum_dr.append(np.sum(cells[i].dr, axis=0, keepdims=True))
        sum_dr2.append(np.sum(cells[i].dr * cells[i].dr))
    region_size = np.array([ 1+len(_n) for _n in neighbours ])
    B = np.vstack(B)
    B2 = np.array(B2)
    sum_dr = np.vstack(sum_dr)
    sum_dr2 = np.array(sum_dr2)

    L_local_factor_for_D = sum_dr2 / (4. * dt)
    L_local_factor_for_V = sum_dr / (2. * dt[:,np.newaxis])
    neg_L_global_constant = log(4. * pi) * np.dot(n, dt) + (1. + log(2. * pi)) * np.sum(region_size)
    aD_constant_factor = 1. / (4. * n * dt)
    aV_constant_factor1 = np.sum(B * sum_dr, axis=1) / (n * B2)
    aV_constant_factor2 = [ np.dot(_Bu, _B) for _Bu, _B in zip(Bu, B / B2[:,np.newaxis]) ]
    bV_constant_factor = (n * B2) / (2. * dt)

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

            # mean-field parameters
            aD, bD, aV, bV = [], [], [], []
            for _i, i in enumerate(index):
                V_neighbours = V[neighbours[_i]]
                nabla_V = np.dot(V_neighbours - V[_i], Bu[_i])
                nabla_V_star = nabla_V
                residual_dr = cells[i].dr - nabla_V_star[np.newaxis,:]
                chi2 = np.sum(residual_dr * residual_dr)
                _aD = chi2 * aD_constant_factor[_i]
                _aV = aV_constant_factor1[_i] / _aD + _aD / D[_i] * np.dot(aV_constant_factor2[_i], V_neighbours)
                _bD = n[_i] / (_aD * _aD) * (1. + (_aD * np.dot(nabla_V_star, nabla_V_star)) / (2. * dt[_i]))
                _bV = _aD * bV_constant_factor[_i]
                aD.append(_aD)
                bD.append(_bD)
                aV.append(_aV)
                bV.append(_bV)
            aD, bD, aV, bV = np.array(aD), np.array(bD), np.array(aV), np.array(bV)

            # priors
            BD = bD + BD_constant_term
            BV = bV + BV_constant_term
            AD = aD * (bD / 2.)
            AV = aV * (bV / 2.)
            if diffusivity_spatial_prior is not None:
                # reminder: C comes without mu (global diffusion prior) or lambda (global potential prior)
                AD += diffusivity_spatial_prior * \
                        np.array([ np.dot(aD[_neighbours], _C) for _neighbours, _C in zip(neighbours, C) ])
                AV += potential_spatial_prior * \
                        np.array([ np.dot(aV[_neighbours], _C) for _neighbours, _C in zip(neighbours, C) ])
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
            nablaV_neighbours = [ np.dot(aV[_neighbours], _Bu) for _neighbours, _Bu in zip(neighbours, Bu) ]
            nabla_V_star = np.stack(nablaV_neighbours, axis=0) - B * aV[:,np.newaxis]
            nabla_V_star2 = np.sum(nabla_V_star * nabla_V_star, axis=1)
            B2_over_bV = B2 / bV
            B2_over_bV_neighbours = np.array([ np.sum(B2_over_bV[_neighbours]) for _neighbours in neighbours ])
            B2_over_bV_star = B2_over_bV + B2_over_bV_neighbours
            log_bV = np.log(bV)
            log_bV_neighbours = np.array([ np.sum(log_bV[_neighbours]) for _neighbours in neighbours ])
            log_bV_star = log_bV + log_bV_neighbours

            neg_L = neg_L_global_constant + np.dot(n, np.log(aD) - 1./ (2. * aD * aD * bD)) + \
                    np.dot(1./ aD + 1./ (aD**3 * bD), L_local_factor_for_D) + \
                    np.sum(nabla_V_star * L_local_factor_for_V) + \
                    np.dot((n * aD) / (4. * dt), nabla_V_star2 + B2_over_bV_star) + \
                    .5 * np.sum(np.log(bD) + log_bV_star)

            # step forward
            D, V = aD, aV
            # stopping criterion
            if abs(neg_L - neg_L_prev) < tol:
                break

    except KeyboardInterrupt:
        print('interrupted')
        print('D={} V={}'.format(D,V))
        pass

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

