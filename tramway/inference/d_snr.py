# -*- coding: utf-8 -*-

# Copyright © 2018, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
from .d import d_neg_posterior
from .smooth_d import smooth_d_neg_posterior
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import OrderedDict


setup = {'name': 'd.snr',
    'provides': 'snr',
    'arguments': OrderedDict((
        ('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
        ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
        ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
        ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')),
        ('max_iter',        dict(type=int, help='maximum number of iterations')))),
    'cell_sampling': 'group'}


def infer_snr_d(cells, localization_error=None, diffusivity_prior=None, jeffreys_prior=None, \
    min_diffusivity=None, max_iter=None, **kwargs):
    '''
    Infer signal-to-noise ratio related variables useful for Bayes factor calculation.

    Diffusion is inferred using the D mode, with or without spatial smoothing.
    '''
    if max_iter:
        options = kwargs.get('options', {})
        options['maxiter'] = max_iter
        kwargs['options'] = options
    # initial values and common calculations
    localization_error = cells.get_localization_error(kwargs, 0.03, True, \
            localization_error=localization_error)
    index, reverse_index, n, dt_mean, D_initial, min_diffusivity, D_bounds, _ = \
        smooth_infer_init(cells, min_diffusivity=min_diffusivity, jeffreys_prior=jeffreys_prior)
    index = np.asarray(index)
    nnz = 1 < n
    if diffusivity_prior is None:
        if min_diffusivity is not None:
            kwargs['bounds'] = [D_bounds[0]] # they are all the same
        result = []
        for i, dt, D in zip(index, dt_mean, D_initial):
            r = minimize(d_neg_posterior, D,
                args = (cells[i], localization_error, jeffreys_prior, dt, min_diffusivity),
                **kwargs)
            result.append(r.x[0])
        D = np.array(result)
    else:
        # infer diffusivity D (defined at cells `index`)
        if min_diffusivity is not None:
            kwargs['bounds'] = D_bounds
        result = minimize(smooth_d_neg_posterior, D_initial, \
            args=(cells, localization_error, diffusivity_prior, jeffreys_prior, dt_mean, min_diffusivity, reverse_index), \
            **kwargs)
        D = result.x
    # compute diffusivity gradient g (defined at cells `g_index`)
    g_index, g = [], []
    g_defined = np.zeros(len(index), dtype=bool)
    for j, i in enumerate(index):
        gradD = cells.grad(i, D, reverse_index)
        if gradD is not None and nnz[j]:
            # the nnz condition does not prevent the gradient to be defined
            # but such cases are excluded anyway in the calculation of zeta_spurious
            g_defined[j] = True
            g_index.append(i)
            g.append(gradD[np.newaxis,:])
    g = np.concatenate(g, axis=0)
    # compute mean displacement m and variances V and V_prior (defined at cells `index`)
    sum_pts  = lambda a: np.sum(a, axis=0, keepdims=True)
    sum_dims = lambda a: np.sum(a, axis=1, keepdims=True)
    m, dts, dr, dr2 = [], [], [], []
    for i in index:
        cell = cells[i]
        m.append(np.mean(cell.dr, axis=0, keepdims=True))
        dr.append(sum_pts(cell.dr))
        dr2.append(sum_pts(cell.dr * cell.dr))
        dts.append(cell.dt)
    m   = np.concatenate(m, axis=0)
    dts = np.concatenate(dts)
    n   = n[:,np.newaxis]
    dr  = np.concatenate(dr, axis=0)
    dr2 = np.concatenate(dr2, axis=0)
    V   = sum_dims(dr2 - dr * dr / n) / n #(n - 1)
    n_prior   = np.sum(n)    - n
    dr_prior  = sum_pts(dr)  - dr
    dr2_prior = sum_pts(dr2) - dr2
    V_prior   = sum_dims(dr2_prior - dr_prior * dr_prior / n_prior) / n_prior #(n_prior - 1)
    # compute zeta_total (defined at cells `index`) and zeta_spurious (defined at cells `g_index`)
    sd = np.sqrt(V)
    zeta_total = np.zeros_like(m)
    zeta_total[nnz] = m[nnz] / sd[nnz]
    dt = np.median(dts)
    zeta_spurious = g * dt / sd[g_defined]
    # format the output
    result = pd.DataFrame(
        np.concatenate((n, V_prior), axis=1),#, dr, dr2
        index=index,
        columns=['n'] + \
            ['V_prior'],# + \
            #['dr '+col for col in cells.space_cols] + \
            #['dr2 '+col for col in cells.space_cols],
        )
    result = result.join(pd.DataFrame(
        np.concatenate((V, D[:,np.newaxis], zeta_total), axis=1)[nnz],
        index=index[nnz],
        columns=['V'] + \
            ['diffusivity'] + \
            ['zeta_total '+col for col in cells.space_cols],
        ))
    result = result.join(pd.DataFrame(
        zeta_spurious,
        index=g_index,
        columns=['zeta_spurious '+col for col in cells.space_cols],
        ))
    return result

