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


setup = {'name': ('meanfield.dd', 'meanfield.ddrift'),
        'provides': ('dd', 'ddrift'),
        'arguments': OrderedDict((
            #('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
            ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
            ('drift_prior',         dict(type=float, help='prior on the amplitude of the drift')),
            ('diffusivity_time_prior',  ('--time-d', dict(type=float, help='prior on the temporal variations of the diffusivity'))),
            ('drift_time_prior',        dict(type=float, help='prior on the temporal variations of drift amplitude')),
            ('verbose',         ()))),
        'default_rgrad':    'delta0'}
#setup_with_grad_arguments(setup)


def infer_meanfield_DD(cells, diffusivity_spatial_prior=None, drift_spatial_prior=None,
        diffusivity_time_prior=None, drift_time_prior=None,
        diffusivity_prior=None, drift_prior=None,
        diffusion_prior=None, diffusion_spatial_prior=None, diffusion_time_prior=None,
        dt=None, tol=1e-6, verbose=True, **kwargs):
    """
    """
    grad_kwargs = get_grad_kwargs(kwargs)
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)
    index, reverse_index, n, dt_mean, _, _, _, _ = \
        smooth_infer_init(cells, sigma2=0)#localization_error)
    if dt is None:
        dt = dt_mean
    elif np.isscalar(dt):
        dt = np.full_like(dt_mean, dt)

    if diffusivity_prior is None:
        diffusivity_prior = diffusion_prior
    if diffusivity_spatial_prior is None:
        if diffusion_spatial_prior is None:
            diffusivity_spatial_prior = diffusivity_prior
        else:
            diffusivity_spatial_prior = diffusion_spatial_prior
    if diffusivity_time_prior is None:
        diffusivity_time_prior = diffusion_time_prior

    if drift_spatial_prior is None:
        drift_spatial_prior = drift_prior
    if drift_time_prior is None:
        drift_time_prior = drift_prior
    mu_spatial_prior, mu_time_prior = drift_spatial_prior, drift_time_prior

    spatial_reg = diffusivity_spatial_prior or drift_spatial_prior
    time_reg = diffusivity_time_prior or drift_time_prior
    reg = spatial_reg or time_reg

    index = np.array(index)
    ok = 1<n
    #print(ok.size, np.sum(ok))
    if not np.all(ok):
        reverse_index[index] -= np.cumsum(~ok)
        reverse_index[index[~ok]] = -1
        index, n, dt = index[ok], n[ok], dt[ok]

    def spatial_background(__i, __x):
        _dx = []
        _r0 = cells[__i].center
        for __j in cells.neighbours(__i):
            __dr = cells[__j].center - _r0
            __dx = __x[reverse_index[__j]] / np.sum(__dr * __dr)
            if not np.isnan(__dx):
                _dx.append( __dx )
        if _dx:
            return np.mean(_dx)
        else:
            return 0.
    def temporal_background(__i, __x):
        _dx = []
        _t0 = cells[__i].center_t
        for __j in cells.time_neighbours(__i):
            __dt = cells[__j].center_t - _t0
            __dx = __x[reverse_index[__j]] / np.sum(__dt * __dt)
            if not np.isnan(__dx):
                _dx.append( __dx )
        if _dx:
            return np.mean(_dx)
        else:
            return 0.

    dr, chi2 = [], []
    if reg:
        reg_Z = []
        time_reg_Z = []
        ones = np.ones(n.size, dtype=dt.dtype)
    for _i, i in enumerate(index):
        # local variables
        dr_mean = np.mean(cells[i].dr, axis=0, keepdims=True)
        dr_centered = cells[i].dr - dr_mean
        dr2 = np.sum(dr_centered * dr_centered, axis=1)
        dr.append(dr_mean)
        chi2.append(np.sum(dr2))
        # regularization constants
        if reg:
            if spatial_reg:
                reg_Z.append( spatial_background(i, ones) )
            if time_reg:
                time_reg_Z.append( temporal_background(i, ones) )
    dr = np.vstack(dr)
    chi2 = np.array(chi2)
    if reg:
        reg_Z = np.array(reg_Z)
        time_reg_Z = np.array(time_reg_Z)

    aD = chi2 / (4. * n * dt)
    a_mu = dr
    bD = n / (aD * aD)
    b_mu = n / (2. * dt * aD)

    # priors
    if diffusivity_spatial_prior or diffusivity_time_prior:
        AD = aD * bD
        BD = bD # no need for copying
        # do not scale time;
        # assume constant window shift and let time_prior hyperparameters bear all the responsibility
        D_spatial_penalty = []
        D_time_penalty = []
        for _i, i in enumerate(index):
            if diffusivity_spatial_prior is not None:
                D_spatial_penalty.append( spatial_background(i, aD) )
            if diffusivity_time_prior is not None:
                D_time_penalty.append( temporal_background(i, aD) )
        if diffusivity_spatial_prior is not None:
            AD += 2. * diffusivity_spatial_prior * np.array(D_spatial_penalty)
            BD += 2. * diffusivity_spatial_prior * reg_Z
        if diffusivity_time_prior is not None:
            AD += 2. * diffusivity_time_prior * np.array(D_time_penalty)
            BD += 2. * diffusivity_time_prior * time_reg_Z
        AD /= BD
        aD, bD = AD, BD
    if mu_spatial_prior or mu_time_prior:
        A_mu = a_mu * b_mu[:,np.newaxis]
        B_mu = b_mu # no need for copying
        mu_spatial_penalty = []
        mu_time_penalty = []
        for _i, i in enumerate(index):
            if mu_spatial_prior is not None:
                delta_a_mu = []
                for d in range(cells.dim):
                    delta_a_mu.append( spatial_background(i, a_mu[:,d]) )
                mu_spatial_penalty.append(delta_a_mu)
            if mu_time_prior is not None:
                ddt_a_mu = []
                for d in range(cells.dim):
                    ddt_a_mu.append( temporal_background(i, a_mu[:,d]) )
                mu_time_penalty.append(ddt_a_mu)
        if mu_spatial_prior is not None:
            A_mu += 2. * mu_spatial_prior * np.array(mu_spatial_penalty)
            B_mu += 2. * mu_spatial_prior * reg_Z
        if mu_time_prior is not None:
            A_mu += 2. * mu_time_prior * np.array(mu_time_penalty)
            B_mu += 2. * mu_time_prior * time_reg_Z
        A_mu /= B_mu[:,np.newaxis]
        a_mu, b_mu = A_mu, B_mu

    D, drift = aD, a_mu / dt[:,np.newaxis]

    DD = pd.DataFrame(np.hstack((D[:,np.newaxis], drift)), index=index, \
        columns=[ 'diffusivity' ] + \
            [ 'drift ' + col for col in cells.space_cols ])

    return DD


__all__ = ['setup', 'infer_meanfield_DD']

