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


class Meanfield(object):
    def __init__(self, cells, dt=None):
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


class FrictionDrift(Meanfield):
    def __init__(self, cells, dt=None,
            friction_spatial_prior=None, friction_time_prior=None,
            drift_spatial_prior=None, drift_time_prior=None,
            diffusivity_spatial_prior=None, diffusivity_time_prior=None):
        Meanfield.__init__(self, cells, dt)
        n = self.n
        dt = self.dt

        self.psi_spatial_prior = psi_spatial_prior = friction_spatial_prior / (2 * dt) \
                if friction_spatial_prior else None
        self.psi_time_prior = psi_time_prior = friction_time_prior / (2 * dt) \
                if friction_time_prior else None
        self.mu_spatial_prior = mu_spatial_prior = drift_spatial_prior * dt \
                if drift_spatial_prior else None
        self.mu_time_prior = mu_time_prior = drift_time_prior * dt \
                if drift_time_prior else None
        self.D_spatial_prior = D_spatial_prior = diffusivity_spatial_prior \
                if diffusivity_spatial_prior else None
        self.D_time_prior = D_time_prior = diffusivity_time_prior \
                if diffusivity_time_prior else None

        spatial_reg = friction_spatial_prior or drift_spatial_prior or diffusivity_spatial_prior
        time_reg = friction_time_prior or drift_time_prior or diffusivity_time_prior
        if spatial_reg or time_reg:
            # regularization constants
            reg_Z = []
            time_reg_Z = []
            ones = np.ones(n.size, dtype=dt.dtype)
            for i in self.index:
                if spatial_reg:
                    reg_Z.append( self.spatial_background(i, ones) )
                if time_reg:
                    time_reg_Z.append( self.temporal_background(i, ones) )

            B_psi_additive_term = 0.
            B_mu_additive_term = 0.
            BD_additive_term = 0.
            if reg_Z:
                reg_Z = np.array(reg_Z)
                if friction_spatial_prior:
                    B_psi_additive_term = B_psi_additive_term + \
                            2 * psi_spatial_prior * reg_Z
                if drift_spatial_prior:
                    B_mu_additive_term = B_mu_additive_term + \
                            2 * mu_spatial_prior * reg_Z
                if D_spatial_prior:
                    BD_additive_term = BD_additive_term + \
                            2 * D_spatial_prior * reg_Z
            if time_reg_Z:
                time_reg_Z = np.array(time_reg_Z)
                if friction_time_prior:
                    B_psi_additive_term = B_psi_additive_term + \
                            2 * psi_time_prior * time_reg_Z
                if drift_time_prior:
                    B_mu_additive_term = B_mu_additive_term + \
                            2 * mu_time_prior * time_reg_Z
                if D_time_prior:
                    BD_additive_term = BD_additive_term + \
                            2 * D_time_prior * time_reg_Z

            self.B_psi_additive_term = B_psi_additive_term
            self.B_mu_additive_term = B_mu_additive_term
            self.BD_additive_term = BD_additive_term

    @property
    def regularize_friction(self):
        return not (self.psi_spatial_prior is None and self.psi_time_prior is None)
    @property
    def regularize_drift(self):
        return not (self.mu_spatial_prior is None and self.mu_time_prior is None)
    @property
    def regularize_diffusion(self):
        return not (self.D_spatial_prior is None and self.D_time_prior is None)

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

    def set_psi_regularizer(self, a_psi):
        spatial_reg = self.psi_spatial_prior is not None
        time_reg = self.psi_time_prior is not None

        psi_spatial_penalty = []
        psi_time_penalty = []
        for i in self.index:
            if spatial_reg:
                psi_spatial_penalty.append( self.spatial_background(i, a_psi) )
            if time_reg:
                psi_time_penalty.append( self.temporal_background(i, a_psi) )

        A_psi_additive_term = 0.
        if spatial_reg:
            A_psi_additive_term = A_psi_additive_term + \
                    2 * self.psi_spatial_prior * np.array(psi_spatial_penalty)
        if time_reg:
            A_psi_additive_term = A_psi_additive_term + \
                    2 * self.psi_time_prior * np.array(psi_time_penalty)
        self.A_psi_additive_term = A_psi_additive_term

    def set_mu_regularizer(self, a_mu):
        spatial_reg = self.mu_spatial_prior is not None
        time_reg = self.mu_time_prior is not None

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
        self.A_mu_additive_term = A_mu_additive_term[:,np.newaxis] * u_mu

    def set_D_regularizer(self, a_psi):
        spatial_reg = self.D_spatial_prior is not None
        time_reg = self.D_time_prior is not None

        aD = 1. / a_psi

        D_spatial_penalty = []
        D_time_penalty = []
        for i in self.index:
            if spatial_reg:
                D_spatial_penalty.append( self.spatial_background(i, aD) )
            if time_reg:
                D_time_penalty.append( self.temporal_background(i, aD) )

        AD_additive_term = 0.
        if spatial_reg:
            AD_additive_term = AD_additive_term + \
                    2 * self.D_spatial_prior * np.array(D_spatial_penalty)
        if time_reg:
            AD_additive_term = AD_additive_term + \
                    2 * self.D_time_prior * np.array(D_time_penalty)
        self.AD_additive_term = AD_additive_term

    def regularize_psi(self, a_psi, b_psi, oneshot=True):
        if oneshot:
            self.set_psi_regularizer(a_psi)
        A_psi = a_psi * b_psi + self.A_psi_additive_term
        B_psi = b_psi + self.B_psi_additive_term
        A_psi /= B_psi
        return A_psi, B_psi

    def regularize_mu(self, a_mu, b_mu, oneshot=True):
        if oneshot:
            self.set_mu_regularizer(a_mu)
        a_mu = self.dr
        A_mu = a_mu * b_mu[:,np.newaxis] + self.A_mu_additive_term
        B_mu = b_mu + self.B_mu_additive_term
        A_mu /= B_mu[:,np.newaxis]
        return A_mu, B_mu

    def regularize_D(self, a_psi, b_psi, oneshot=True):
        if oneshot:
            self.set_D_regularizer(a_psi)
        AD = b_psi / a_psi + self.AD_additive_term
        BD = B_psi = b_psi + self.BD_additive_term
        A_psi = BD / AD
        return A_psi, B_psi


def infer_meanfield_friction_drift(cells, friction_spatial_prior=None, drift_spatial_prior=None,
        friction_time_prior=None, drift_time_prior=None,
        diffusivity_spatial_prior=None, diffusivity_time_prior=None,
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
    if diffusivity_spatial_prior is None:
        diffusivity_spatial_prior = diffusivity_prior

    mf = FrictionDrift(cells, dt,
            friction_spatial_prior, friction_time_prior,
            drift_spatial_prior, drift_time_prior,
            diffusivity_spatial_prior, diffusivity_time_prior)
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

    neg_L = np.inf
    k = 0
    try:
        while True:
            neg_L_prev = neg_L

            if mf.regularize_friction:
                a_psi, b_psi = mf.regularize_psi(a_psi, b_psi)
            if mf.regularize_diffusion:
                a_psi, b_psi = mf.regularize_D(a_psi, b_psi)
            if mf.regularize_drift:
                a_mu, b_mu = mf.regularize_mu(a_mu, b_mu)

                chi2_over_n = dr2 - 2 * np.sum(dr * a_mu, axis=1) + np.sum(a_mu * a_mu, axis=1)

                neg_L = np.sum(
                        n * ( \
                            1./ (2 * a_psi * a_psi * b_psi) - \
                            np.log(a_psi) + \
                            a_psi * (chi2_over_n / 2 + 1. / b_mu)
                            ) + \
                        .5 * (np.log(b_psi) + np.log(b_mu))
                        )

                # stopping criterion
                if verbose:
                    print('[{}] approx -logP: {}'.format(k, neg_L))

                if abs(neg_L - neg_L_prev) < tol:
                    resolution = 'CONVERGENCE: DELTA -L < TOL'
                    break

                #else: update and iterate
                a_psi = 2. / chi2_over_n
                b_psi = n / (a_psi * a_psi)
                b_mu = n * a_psi

                k += 1

            else:
                resolution = 'CONVERGENCE: TRIVIAL'
                break

    except KeyboardInterrupt:
        resolution = 'INTERRUPTED'
        if verbose:
            print('interrupted')
        pass

    gamma = a_psi * (2 * dt)
    drift = a_mu / dt[:,np.newaxis]

    FD = pd.DataFrame(np.hstack((gamma[:,np.newaxis], drift)), index=mf.index, \
        columns=[ 'friction' ] + \
            [ 'drift ' + col for col in cells.space_cols ])

    return FD


__all__ = ['setup', 'infer_meanfield_friction_drift', 'Meanfield', 'FrictionDrift']

