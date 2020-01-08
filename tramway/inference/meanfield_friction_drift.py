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
#from .gradient import get_grad_kwargs, setup_with_grad_arguments
from math import *
import numpy as np
import pandas as pd
from collections import OrderedDict


setup = {'name': ('meanfield.fd', 'meanfield friction,drift'),
        'provides': ('fd', 'friction,drift'),
        'arguments': OrderedDict((
            #('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
            ('friction_prior',   dict(type=float, help='prior on the friction')),
            ('drift_prior',         dict(type=float, help='prior on the amplitude of the drift')),
            ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
            ('friction_time_prior',  dict(type=float, help='prior on the temporal variations of the fluctuation friction')),
            ('drift_time_prior',        dict(type=float, help='prior on the temporal variations of drift amplitude')),
            ('diffusivity_time_prior',   ('--time-d', dict(type=float, help='prior on the temporal variations of the diffusivity'))),
            ('verbose',         ()))),
        'default_rgrad':    'delta0'}
#setup_with_grad_arguments(setup)


def _regularize_D_or_psi(aD, bD, a_psi, b_psi):
    _exc = ValueError('either specify aD and bD, or a_psi and b_psi')
    if aD is None:
        if bD is not None or a_psi is None or b_psi is None:
            raise _exc
        return False
    else:
        if bD is None or a_psi is not None or b_psi is not None:
            raise _exc
        return True

class Meanfield(object):
    def __init__(self, cells, dt=None,
            diffusivity_spatial_prior=None, diffusivity_time_prior=None,
            friction_spatial_prior=None, friction_time_prior=None,
            _inherit=False):
        index, reverse_index, n, dt_mean, _, _, _, _ = \
            smooth_infer_init(cells, sigma2=0)#localization_error)
        if dt is None:
            dt = dt_mean
        elif np.isscalar(dt):
            dt = np.full_like(dt_mean, dt)

        index = np.array(index)
        ok = 1<n
        #print(ok.size, np.sum(ok))
        if not np.all(ok):
            reverse_index[index] -= np.cumsum(~ok)
            reverse_index[index[~ok]] = -1
            index, n, dt = index[ok], n[ok], dt[ok]

        dr, dr2 = [], []
        for i in index:
            dr_mean = np.mean(cells[i].dr, axis=0, keepdims=True)
            dr.append(dr_mean)
            dr2.append(np.sum(cells[i].dr * cells[i].dr))
        dr = np.vstack(dr)
        dr2 = np.array(dr2) / n

        self.cells = cells
        self.n = n
        self.dt = dt
        self.dr = dr
        self.dr2 = dr2
        self.index = index
        self.reverse_index = reverse_index

        self.spatial_prior, self.time_prior = {}, {}
        self.add_feature('D')
        self.add_feature('psi')

        self.psi_spatial_prior = friction_spatial_prior / (2 * dt) \
                if friction_spatial_prior else None
        self.psi_time_prior = friction_time_prior / (2 * dt) \
                if friction_time_prior else None
        self.D_spatial_prior = diffusivity_spatial_prior \
                if diffusivity_spatial_prior else None
        self.D_time_prior = diffusivity_time_prior \
                if diffusivity_time_prior else None

        self._first_warning = True

        if not _inherit:
            self.__post_init__()

    @property
    def psi_spatial_prior(self):
        return self.spatial_prior['psi']

    @psi_spatial_prior.setter
    def psi_spatial_prior(self, psi_spatial_prior):
        self.spatial_prior['psi'] = psi_spatial_prior

    @property
    def D_spatial_prior(self):
        return self.spatial_prior['D']

    @D_spatial_prior.setter
    def D_spatial_prior(self, D_spatial_prior):
        self.spatial_prior['D'] = D_spatial_prior

    def __post_init__(self):
        any_spatial_prior = not all([ p is None for p in self.spatial_prior.values() ])
        any_time_prior = not all([ p is None for p in self.time_prior.values() ])
        if any_spatial_prior or any_time_prior:

            spatial_constants = []
            time_constants = []
            ones = np.ones(self.n.size, dtype=self.dt.dtype)
            for i in self.index:
                if any_spatial_prior:
                    spatial_constants.append( self.spatial_background(i, ones) )
                if any_time_prior:
                    time_constants.append( self.temporal_background(i, ones) )
            spatial_constants = np.array(spatial_constants) if any_spatial_prior else None
            time_constants = np.array(time_constants) if any_time_prior else None

            self.set_B_constants(spatial_constants, time_constants)

    def spatial_background(self, i, x):
        dxs = []
        r0 = self.cells[i].center
        for j in self.cells.neighbours(i):
            dr = self.cells[j].center - r0
            dx = x[self.reverse_index[j]] / np.sum(dr * dr)
            if not np.isnan(dx):
                dxs.append(dx)
        return np.mean(dxs) if dxs else 0.

    def temporal_background(self, i, x):
        dxs = []
        t0 = self.cells[i].center_t
        for j in self.cells.time_neighbours(i):
            dt = self.cells[j].center_t - t0
            dx = x[self.reverse_index[j]] / (dt * dt)
            if not np.isnan(dx):
                dxs.append(dx)
        return np.mean(dxs) if dxs else 0.

    def set_B_constants(self, spatial_constants, time_constants):
        self.A_additive_term, self.B_additive_term = {}, {}
        priors = self.spatial_prior
        for v in priors:
            B_additive_term = 0.
            if self.spatial_prior.get(v, None) is not None:
                B_additive_term = B_additive_term + \
                        2 * self.spatial_prior[v] * spatial_constants
            if self.time_prior.get(v, None) is not None:
                B_additive_term = B_additive_term + \
                        2 * self.time_prior[v] * time_constants
            self.B_additive_term[v] = B_additive_term

    @property
    def regularize_friction(self):
        return not (self.psi_spatial_prior is None and self.psi_time_prior is None)
    @property
    def regularize_diffusion(self):
        return not (self.D_spatial_prior is None and self.D_time_prior is None)

    def set_psi_regularizer(self, a_psi=None, aD=None):
        if a_psi is None:
            a_psi = 1. / aD
        self.set_regularizer('psi', a_psi)

    def set_D_regularizer(self, aD=None, a_psi=None):
        if aD is None:
            aD = 1. / a_psi
        self.set_regularizer('D', aD)

    def regularize_psi(self, a_psi=None, b_psi=None, A_psi=None, B_psi=None, oneshot=True,
            aD=None, bD=None, AD=None, BD=None):
        _in_D = _regularize_D_or_psi(aD, bD, a_psi, b_psi)
        if _in_D:
            a_psi, b_psi = 1./aD, bD
            A_psi, B_psi = None if AD is None else 1./AD, BD
        A_psi, B_psi = self.regularize('psi', a_psi, b_psi, A_psi, B_psi, oneshot)
        if _in_D:
            AD, BD = 1./A_psi, B_psi
            return AD, BD
        else:
            return A_psi, B_psi

    def regularize_D(self, aD=None, bD=None, AD=None, BD=None, oneshot=True,
            a_psi=None, b_psi=None, A_psi=None, B_psi=None):
        _in_D = _regularize_D_or_psi(aD, bD, a_psi, b_psi)
        if not _in_D:
            aD, bD = 1./a_psi, b_psi
            AD, BD = None if A_psi is None else 1./A_psi, B_psi
        AD, BD = self.regularize('D', aD, bD, AD, BD, oneshot)
        if _in_D:
            return AD, BD
        else:
            A_psi, B_psi = 1./AD, BD
            return A_psi, B_psi

    def set_regularizer(self, psi, a_psi):
        spatial_reg = self.spatial_prior[psi] is not None
        time_reg = self.time_prior[psi] is not None

        spatial_penalty = []
        time_penalty = []
        for i in self.index:
            if spatial_reg:
                spatial_penalty.append( self.spatial_background(i, a_psi) )
            if time_reg:
                time_penalty.append( self.temporal_background(i, a_psi) )

        A_psi_additive_term = 0.
        if spatial_reg:
            A_psi_additive_term = A_psi_additive_term + \
                    2 * self.spatial_prior[psi] * np.array(spatial_penalty)
        if time_reg:
            A_psi_additive_term = A_psi_additive_term + \
                    2 * self.time_prior[psi] * np.array(time_penalty)
        self.A_additive_term[psi] = A_psi_additive_term

    def set_norm_regularizer(self, mu, a_mu):
        spatial_reg = self.spatial_prior[mu] is not None
        time_reg = self.time_prior[mu] is not None

        mu_spatial_penalty = []
        mu_time_penalty = []
        a_mu_norm = np.sqrt(np.sum(a_mu * a_mu, axis=1))
        u_mu = a_mu / a_mu_norm[:,np.newaxis]
        for i in self.index:
            if spatial_reg:
                mu_spatial_penalty.append( self.spatial_background(i, a_mu_norm) )
            if time_reg:
                mu_time_penalty.append( self.temporal_background(i, a_mu_norm) )

        A_mu_additive_term = 0.
        if spatial_reg:
            A_mu_additive_term = A_mu_additive_term + \
                    2 * self.mu_spatial_prior * np.array(mu_spatial_penalty)
        if time_reg:
            A_mu_additive_term = A_mu_additive_term + \
                    2 * self.mu_time_prior * np.array(mu_time_penalty)
        self.A_additive_term[mu] = A_mu_additive_term[:,np.newaxis] * u_mu

    def regularize(self, psi, a_psi, b_psi, A_psi=None, B_psi=None, oneshot=True):
        if oneshot:
            if A_psi is None:
                A_psi = a_psi
            self.set_regularizer(psi, A_psi)
        A_psi = a_psi * b_psi + self.A_additive_term[psi]
        if B_psi is None:
            B_psi = b_psi + self.B_additive_term[psi]
        A_psi /= B_psi
        return A_psi, B_psi

    def regularize_norm(self, mu, a_mu, b_mu, A_mu=None, B_mu=None, oneshot=True, verbose=True):
        if A_mu is None:
            A_mu = a_mu

        elif verbose:
            a_max = np.sqrt(np.max(np.sum(a_mu*a_mu,axis=1)))
            A_max = np.sqrt(np.max(np.sum(A_mu*A_mu,axis=1)))
            if 10 < (A_max + 1) / (a_max + 1):
                print('potential divergence: max ||a||: {}  max ||A||: {}'.format(a_max, A_max))
                if self._first_warning:
                    self._first_warning = False
                    if B_mu is None:
                        B_mu = b_mu + self.B_additive_term[mu]
                    print('min b: {}  max b: {}  min B: {}  max B: {}'.format(np.min(b_mu), np.max(b_mu), np.min(B_mu), np.max(B_mu)))

        if oneshot:
            self.set_norm_regularizer(mu, A_mu)
        A_mu = a_mu * b_mu[:,np.newaxis] + self.A_additive_term[mu]
        if B_mu is None:
            B_mu = b_mu + self.B_additive_term[mu]
        A_mu /= B_mu[:,np.newaxis]
        return A_mu, B_mu

    def add_feature(self, v):
        self.spatial_prior[v] = self.time_prior[v] = None


class MeanfieldDrift(Meanfield):
    def __init__(self, cells, dt=None,
            diffusivity_spatial_prior=None, diffusivity_time_prior=None,
            friction_spatial_prior=None, friction_time_prior=None,
            drift_spatial_prior=None, drift_time_prior=None,
            _inherit=False):
        Meanfield.__init__(self, cells, dt,
                diffusivity_spatial_prior, diffusivity_time_prior,
                friction_spatial_prior, friction_time_prior,
                _inherit=True)
        self.add_feature('mu')

        dt = self.dt
        self.mu_spatial_prior = drift_spatial_prior * dt \
                if drift_spatial_prior else None
        self.mu_time_prior = drift_time_prior * dt \
                if drift_time_prior else None

        if not _inherit:
            self.__post_init__()

    @property
    def mu_spatial_prior(self):
        return self.spatial_prior['mu']

    @mu_spatial_prior.setter
    def mu_spatial_prior(self, mu_spatial_prior):
        self.spatial_prior['mu'] = mu_spatial_prior

    @property
    def mu_time_prior(self):
        return self.time_prior['mu']

    @mu_time_prior.setter
    def mu_time_prior(self, mu_time_prior):
        self.time_prior['mu'] = mu_time_prior

    @property
    def regularize_drift(self):
        return not (self.mu_spatial_prior is None and self.mu_time_prior is None)

    def set_mu_regularizer(self, a_mu):
        self.set_norm_regularizer('mu', a_mu)

    def regularize_mu(self, a_mu, b_mu, A_mu=None, B_mu=None, oneshot=True):
        return self.regularize_norm('mu', a_mu, b_mu, A_mu, B_mu, oneshot)


class FrictionDrift(MeanfieldDrift):
    def __init__(self, cells, dt=None,
            friction_spatial_prior=None, friction_time_prior=None,
            drift_spatial_prior=None, drift_time_prior=None,
            _inherit=False):
        MeanfieldDrift.__init__(self, cells, dt,
                friction_spatial_prior=friction_spatial_prior,
                friction_time_prior=friction_time_prior,
                drift_spatial_prior=drift_spatial_prior,
                drift_time_prior=drift_time_prior,
                _inherit=_inherit)


def infer_meanfield_friction_drift(cells, friction_spatial_prior=None, drift_spatial_prior=None,
        friction_time_prior=None, drift_time_prior=None,
        friction_prior=None, drift_prior=None, diffusivity_prior=None,
        dt=None, tol=1e-6, verbose=True, **kwargs):
    """
    """
    #grad_kwargs = get_grad_kwargs(kwargs)
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)

    if friction_spatial_prior is None:
        friction_spatial_prior = friction_prior
    if drift_spatial_prior is None:
        drift_spatial_prior = drift_prior

    mf = FrictionDrift(cells, dt,
            friction_spatial_prior, friction_time_prior,
            drift_spatial_prior, drift_time_prior)

    n = mf.n
    dt = mf.dt
    dr = mf.dr
    dr2 = mf.dr2

    # initial values
    a_mu = dr
    chi2_over_n = dr2 - np.sum(a_mu * a_mu, axis=1)
    a_psi = 2. / chi2_over_n
    b_psi = n / (a_psi * a_psi)
    b_mu = n * a_psi

    # regularize
    if mf.regularize_friction or mf.regularize_drift:

        A_psi, A_mu = a_psi, a_mu
        B_psi = None if mf.regularize_friction else b_psi
        B_mu = None if mf.regularize_drift else b_mu
        neg_L = np.inf
        k = 0
        try:
            while True:
                neg_L_prev = neg_L

                if mf.regularize_friction:
                    A_psi, B_psi = mf.regularize_psi(a_psi, b_psi, A_psi, B_psi)
                if mf.regularize_drift:
                    A_mu, B_mu = mf.regularize_mu(a_mu, b_mu, A_mu, B_mu)
                    chi2_over_n = dr2 - 2 * np.sum(dr * A_mu, axis=1) + np.sum(A_mu * A_mu, axis=1)

                neg_L = np.sum(
                        n * ( \
                            1./ (2 * A_psi * A_psi * B_psi) - \
                            np.log(A_psi) + \
                            A_psi * (chi2_over_n / 2 + 1. / B_mu)
                            ) + \
                        .5 * (np.log(B_psi) + np.log(B_mu))
                        )

                # stopping criterion
                if verbose:
                    print('[{}] approx -logP: {}  max A_psi: {}  max ||A_mu||: {}'.format(k, neg_L, np.max(A_psi), np.sqrt(np.max(np.sum(A_mu*A_mu,1)))))

                if np.isinf(neg_L) or np.isnan(neg_L):
                    resolution = 'DIVERGENCE: L UNDEFINED'
                    break

                elif abs(neg_L - neg_L_prev) < tol:
                    resolution = 'CONVERGENCE: DELTA -L < TOL'
                    break

                if False:
                    a_psi = 2. / chi2_over_n
                    b_psi = n / (a_psi * a_psi)
                    a_mu = A_mu
                    b_mu = n * a_psi

                k += 1

        except KeyboardInterrupt:
            resolution = 'INTERRUPTED'
            if verbose:
                print('interrupted')
            pass

        a_psi, a_mu = A_psi, A_mu

    else:
        resolution = 'CONVERGENCE: TRIVIAL'

    gamma = a_psi * (2 * dt)
    drift = a_mu / dt[:,np.newaxis]

    FD = pd.DataFrame(np.hstack((gamma[:,np.newaxis], drift)), index=mf.index, \
        columns=[ 'friction' ] + \
            [ 'drift ' + col for col in cells.space_cols ])

    return FD


__all__ = ['setup', 'infer_meanfield_friction_drift', 'Meanfield', 'MeanfieldDrift', 'FrictionDrift']

