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
from .gradient import get_grad_kwargs, setup_with_grad_arguments
from math import *
import numpy as np
import pandas as pd
from collections import OrderedDict
from .meanfield_friction_drift import MeanfieldDrift


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
#setup_with_grad_arguments(setup)


class DiffusivityDrift(MeanfieldDrift):
    def __init__(self, cells, dt=None,
            diffusivity_spatial_prior=None, diffusivity_time_prior=None,
            drift_spatial_prior=None, drift_time_prior=None,
            _inherit=False):
        MeanfieldDrift.__init__(self, cells, dt,
                diffusivity_spatial_prior=diffusivity_spatial_prior,
                diffusivity_time_prior=diffusivity_time_prior,
                drift_spatial_prior=drift_spatial_prior,
                drift_time_prior=drift_time_prior,
                _inherit=_inherit)


def infer_meanfield_DD(cells, diffusivity_spatial_prior=None, drift_spatial_prior=None,
        diffusivity_time_prior=None, drift_time_prior=None,
        diffusivity_prior=None, drift_prior=None,
        diffusion_prior=None, diffusion_spatial_prior=None, diffusion_time_prior=None,
        dt=None, tol=1e-6, verbose=True, **kwargs):
    """
    """
    #grad_kwargs = get_grad_kwargs(kwargs)
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)

    if drift_spatial_prior is None:
        drift_spatial_prior = drift_prior
    alternatives = [ diffusion_spatial_prior, diffusivity_prior, diffusion_prior ]
    while diffusivity_spatial_prior is None and alternatives:
        diffusivity_spatial_prior = alternatives.pop(0)
    alternatives = [ diffusion_time_prior, diffusivity_prior, diffusion_prior ]
    while diffusivity_time_prior is None and alternatives:
        diffusivity_time_prior = alternatives.pop(0)

    mf = DiffusivityDrift(cells, dt,
            diffusivity_spatial_prior, diffusivity_time_prior,
            drift_spatial_prior, drift_time_prior)

    n = mf.n
    dt = mf.dt
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
    if mf.regularize_diffusion or mf.regularize_drift:

        AD, A_mu = aD, a_mu
        BD = None if mf.regularize_diffusion else bD
        B_mu = None if mf.regularize_drift else b_mu
        neg_L = np.inf
        k = 0
        try:
            while True:
                neg_L_prev = neg_L

                if mf.regularize_diffusion:
                    AD, BD = mf.regularize_D(aD, bD, AD, BD)
                if mf.regularize_drift:
                    A_mu, B_mu = mf.regularize_mu(a_mu, b_mu, A_mu, B_mu)
                    chi2_over_n = dr2 - 2 * np.sum(dr * A_mu, axis=1) + np.sum(A_mu * A_mu, axis=1)

                inv_abD = .5 / (AD * AD * BD)
                neg_L = np.sum(
                        n * (np.log(AD) - inv_abD) + \
                        (.5 + inv_abD) / (2. * AD * dt) * n * (chi2_over_n + 2. / B_mu) + \
                        .5 * (np.log(BD) + np.log(B_mu))
                        )

                # stopping criterion
                if verbose:
                    print('[{}] approx -logP: {}  max AD: {}  max ||A_mu||: {}'.format(k, neg_L, np.max(AD), np.sqrt(np.max(np.sum(A_mu*A_mu,1)))))

                if abs(neg_L - neg_L_prev) < tol:
                    resolution = 'CONVERGENCE: DELTA -L < TOL'
                    break

                if False:
                    aD = aD_constant_factor * chi2_over_n
                    bD = n / (aD * aD)
                    a_mu = A_mu
                    b_mu = b_mu_constant_factor / aD

                k += 1

        except KeyboardInterrupt:
            resolution = 'INTERRUPTED'
            pass

        aD, a_mu = AD, A_mu

    else:
        resolution = 'CONVERGENCE: TRIVIAL'

    D = aD
    drift = a_mu / dt[:,np.newaxis]

    DD = pd.DataFrame(np.hstack((D[:,np.newaxis], drift)), index=mf.index, \
        columns=[ 'diffusivity' ] + \
            [ 'drift ' + col for col in cells.space_cols ])

    return DD


__all__ = ['setup', 'DiffusivityDrift', 'infer_meanfield_DD']

