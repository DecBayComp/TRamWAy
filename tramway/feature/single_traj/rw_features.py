# -*- coding:utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#    Contributor: Maxime Duval

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""
This module is where features extractible from a single random walk are
defined.
"""

import numpy as np
import pandas as pd
import scipy
import scipy.spatial
from scipy.optimize import root_scalar as root_finder

from .rw_visualization import plot_convex_hull


def _regularize_idminmax(id_min, id_max, N):
    if id_min is None:
        id_min = 0
    else:
        if id_min < 0:
            raise ValueError(f'index min {id_min} < 0 !')
        elif id_min > N-1:
            raise ValueError(f'index min {id_min} > {N-1} !')
    if id_max is None:
        id_max = N
    else:
        if id_max < 0:
            raise ValueError(f'index min {id_max} < 0 !')
        elif id_max > N:
            raise ValueError(f'index min {id_max} > {N} !')
    if id_max - id_min < 1:
        raise ValueError(f'index min : {id_min}, index max : {id_max} ==> '
                         'empty slice !')
    return id_min, id_max


def _escape_rate(future_dists, mean_step, x):
    return np.count_nonzero(np.argmax(future_dists > x * mean_step,
                                      axis=1)) / future_dists.shape[0]


def _zero_escape_quantile(x, r, future_dists, mean_step):
    return _escape_rate(future_dists, mean_step, x) - r


def _Dq(Dabs, q, n_R=10, Rmax=0.5):
    Rmax = min(Rmax, np.max(Dabs))
    Rmin = Rmax / 4
    # Log x samples despite linear fit : because we attach more confidence to
    # low R estimates.
    Rs = np.geomspace(Rmin, Rmax, num=n_R)
    Rs = Rs[(Rs != 0) & (Rs != 1)]
    DqR = [(np.log(R) / (q-1) *
            np.log(np.mean(np.mean(Dabs < R, axis=1)**(q-1))))
           for R in Rs]
    fitprms = scipy.stats.linregress(Rs, DqR)
    return fitprms[1]


def _ergodicity_estimator(X):
    diff = (X[1:] - X[:-1])
    V = diff / np.sqrt(np.mean(diff**2))
    Cst = np.abs(np.mean(np.exp(1j * V)))**2
    n_array = np.unique(np.linspace(1, len(V)-1, min(len(V), 20)).astype(int))
    E = np.array([(np.mean(np.exp(1j * (V[n:] - V[:-n]))) -
                   Cst) if n > 0 else 1 - Cst
                  for n in n_array])
    return np.mean(E)


def _ergodicity_estimator_v2(X, w=2):
    N = len(X)
    Cst = 1/N - 1/(N*(N+1)) * np.abs(np.mean(np.exp(1j * w * (X - X[0]))))**2
    n_array = np.unique(np.linspace(1, N-1, min(N, 20)).astype(int))
    E = np.array([np.mean(np.exp(1j * w * (X[n:] - X[:-n]))) + Cst
                  for n in n_array])
    return np.mean(E)


class RandomWalk():

    def __init__(self, RW_df):
        self.data = RW_df
        self.dims = set(RW_df.columns).intersection({'x', 'y', 'z'})
        self.length = len(self.data)
        self.position = self.data.loc[:, list(self.dims)].values
        self.t = self.data.t.values
        self.dt_vec = self.t[1:] - self.t[:-1]
        self.is_dt_cst = (np.var(self.dt_vec) < 1e-10)
        self.dt = self.dt_vec[0]
        self.Dvec = self.position[:, np.newaxis] - self.position[np.newaxis, :]
        self.Dabs = np.linalg.norm(self.Dvec, axis=2)

    def __len__(self):
        return self.length

    def get_sub_time(self, id_min, id_max):
        id_min, id_max = _regularize_idminmax(id_min, id_max, self.length)
        return self.t[id_min:id_max]

    def get_sub_position(self, id_min, id_max):
        id_min, id_max = _regularize_idminmax(id_min, id_max, self.length)
        return self.position[id_min:id_max]

    def get_sub_Dabs(self, id_min, id_max):
        id_min, id_max = _regularize_idminmax(id_min, id_max, self.length)
        return self.Dabs[id_min:id_max, id_min:id_max]

    def get_sub_Dvec(self, id_min, id_max):
        id_min, id_max = _regularize_idminmax(id_min, id_max, self.length)
        return self.Dvec[id_min:id_max, id_min:id_max]

    def get_sub_steps(self, id_min, id_max):
        id_min, id_max = _regularize_idminmax(id_min, id_max, self.length)
        return np.diagonal(self.Dabs, offset=1)[id_min: id_max]

    # Features

    def feat_step(self, id_min=None, id_max=None):
        """Computes statistics on the step size.

        Parameters
        ----------
        RW : RandomWalk object, with size N
        imin : int, should be between 0 and N-2
        imax : int, should be between 2 and N

        Returns
        -------
        dict of moments of the random variable of the size of single steps.
        """
        steps = self.get_sub_steps(id_min, id_max)
        moments = scipy.stats.describe(steps)
        return {'step_mean': moments.mean, 'step_var': moments.variance,
                'step_skewness': moments.skewness,
                'step_kurtosis': moments.kurtosis,
                'step_min': moments.minmax[0], 'step_max': moments.minmax[1]}

    def temporal_msd(self, id_min=None, id_max=None, n_samples=30):
        """Computes the temporal mean squared deviation of the random walk
        between indices imin and imax.

        Parameters
        ----------
        RW : RandomWalk object, with size N
        imin : int, should be between 0 and N-2
        imax : int, should be between 2 and N

        Returns
        -------
        dict of parameters extracted from the random walk.
        """
        subDabs = self.get_sub_Dabs(id_min, id_max)
        tau_int = np.unique(np.geomspace(1, len(subDabs), num=n_samples,
                                         endpoint=False).astype(int))
        msd = np.array([np.mean(np.diagonal(subDabs, offset=i)**2)
                        for i in tau_int])
        return tau_int * self.dt, msd

    def feat_msd(self, id_min=None, id_max=None, n_samples=30):
        """Computes the alpha and diffusion (D) coefficients :
        msd(dt) = 4 * D * dt ** (alpha)
        """
        tau, msd = self.temporal_msd(id_min, id_max, n_samples)
        if len(tau) > 1:
            slope, intercept, _, _, _ = scipy.stats.linregress(np.log10(tau),
                                                               np.log10(msd))
            return {'msd_alpha': slope, 'msd_diffusion': 10**intercept / 4}
        else:
            return {'msd_alpha': np.nan, 'msd_diffusion': np.nan}

    def feat_drift(self, id_min=None, id_max=None):
        """Computes the mean drift norm of the random walk, defined as ||alpha||
        where X(t) = alpha * t.

        Parameters
        ----------
        RW : RandomWalk object, with size N
        imin : int, should be between 0 and N-2
        imax : int, should be between 2 and N

        Returns
        -------
        dict of the drift norm value.
        """
        subposition = self.get_sub_position(id_min, id_max)
        subt = self.get_sub_time(id_min, id_max)
        drift = np.mean(((subposition[1:] - subposition[0]) /
                         subt[1:, np.newaxis]), axis=0)
        return {'drift_norm': np.linalg.norm(drift)}

    def stats_pdf(self, id_min=None, id_max=None, n_samples=30):
        subDabs = self.get_sub_Dabs(id_min, id_max)
        tau_int = np.unique(np.geomspace(1, len(subDabs)-1, num=n_samples,
                                         endpoint=False).astype(int))
        pdf_stats = [scipy.stats.describe(np.diagonal(subDabs, offset=i))
                     for i in tau_int]
        pdf_stats = np.array([[x.mean, x.variance, x.skewness, x.kurtosis]
                              for x in pdf_stats])
        return tau_int * self.dt, pdf_stats

    def feat_pdf(self, id_min=None, id_max=None, n_samples=30):
        tau, pdf_stats = self.stats_pdf(id_min, id_max, n_samples)
        keys = ['pdf_alpha_mean', 'pdf_beta_mean', 'pdf_rval_mean', 
                'pdf_alpha_var', 'pdf_beta_var', 'pdf_rval_var',
                'pdf_mean_skewness', 'pdf_var_skewness',
                'pdf_mean_kurtosis', 'pdf_var_kurtosis']
        if len(tau) > 1:
            fit_mean = scipy.stats.linregress(np.log10(tau),
                                              np.log10(pdf_stats[:, 0]))
            fit_var = scipy.stats.linregress(np.log10(tau),
                                             np.log10(pdf_stats[:, 1]))
            mu_sk, var_sk = np.mean(pdf_stats[:, 2]), np.var(pdf_stats[:, 2])
            mu_ku, var_ku = np.mean(pdf_stats[:, 3]), np.var(pdf_stats[:, 3])
            vals = [fit_mean[0], 10**fit_mean[1], fit_mean[2],
                    fit_var[0], 10**fit_var[1], fit_var[2],
                    mu_sk, var_sk, mu_ku, var_ku]
            return dict(list(zip(keys, vals)))
        else:
            return dict(list(zip(keys, np.nan * np.ones(len(keys)))))

    def convex_hull(self, id_min=None, id_max=None, display=False):
        subposition = self.get_sub_position(id_min, id_max)
        hull = scipy.spatial.ConvexHull(subposition)
        area, perimeter = hull.volume, hull.area
        hull_points = subposition[hull.vertices]
        n = len(hull_points)
        DX = np.broadcast_to(hull_points[:, 0], (n, n))
        DY = np.broadcast_to(hull_points[:, 1], (n, n))
        D = (DX - DX.T)**2 + (DY - DY.T)**2
        imax = np.argmax(D)
        max_dist = np.sqrt(np.max(D))
        if display:
            plot_convex_hull(subposition, hull, imax)
        return area, perimeter, max_dist

    def gyration_tensor(self, id_min=None, id_max=None):
        X = self.get_sub_position(id_min, id_max)
        Xm = np.mean(X, axis=0)
        a, b = np.mean((X - Xm[np.newaxis, :])**2, axis=0)
        c = np.mean((X[:, 0] - Xm[0]) * (X[:, 1] - Xm[1]))
        return np.array([[a, c], [c, b]])

    def asymmetry(self, id_min=None, id_max=None):
        T = self.gyration_tensor(id_min, id_max)
        l1, l2 = np.linalg.eigvals(T)
        return - np.log(1 - (l1 - l2)**2 / (2 * (l1 + l2)**2))

    def efficiency(self, id_min=None, id_max=None):
        subDabs = self.get_sub_Dabs(id_min, id_max)
        num = subDabs[0, -1]**2
        denom = np.sum(np.diagonal(subDabs, offset=1)**2)
        return num / denom

    def kurtosis(self, id_min=None, id_max=None):
        X = self.get_sub_position(id_min, id_max)
        T = self.gyration_tensor(id_min, id_max)
        w, v = np.linalg.eig(T)
        r = v[:, np.argmax(w)]
        xp = np.sum(X * r, axis=1)
        xpm, xpstd = np.mean(xp), np.std(xp)
        return np.mean((xp - xpm)**4) / xpstd**4

    def straightness(self, id_min=None, id_max=None):
        subDabs = self.get_sub_Dabs(id_min, id_max)
        rs = np.diagonal(subDabs, offset=1)
        return subDabs[0, -1] / np.sum(rs)

    def feat_shape(self, id_min=None, id_max=None):
        id_min, id_max = _regularize_idminmax(id_min, id_max, self.length)
        keys = ['area', 'perimeter', 'max_dist', 'asymmetry',
                'efficiency', 'kurtosis', 'straightness']
        if id_max - id_min < 3:
            return dict(list(zip(keys, np.nan * np.ones(len(keys)))))
        else:
            area, perimeter, max_dist = self.convex_hull(id_min, id_max)
            asymmetry = self.asymmetry(id_min, id_max)
            efficiency = self.efficiency(id_min, id_max)
            kurtosis = self.kurtosis(id_min, id_max)
            straightness = self.straightness(id_min, id_max)
            vals = [area, perimeter, max_dist, asymmetry, efficiency,
                    kurtosis, straightness]
            return dict(list(zip(keys, vals)))

    def feat_escape_time(self, id_min=None, id_max=None):
        subDabs = self.get_sub_Dabs(id_min, id_max)
        subDfuture = subDabs.copy()
        mean_step = np.mean(np.diagonal(subDfuture, offset=1))
        subDfuture[np.tril_indices(len(subDabs))] = 0
        keys = ['escape_dist_q1', 'escape_dist_median', 'escape_dist_q3']
        try:
            qs = [0.25, 0.5, 0.75]
            d_qs = []
            for q in qs:
                args = (q, subDfuture, mean_step)
                d_qs.append(root_finder(_zero_escape_quantile, args=args,
                                        method='bisect',
                                        bracket=(0, 1000)).root)
            vals = np.array(d_qs) * mean_step
            return dict(list(zip(keys, vals)))
        except:
            return dict(list(zip(keys, np.nan * np.ones(len(keys)))))

    def feat_angle(self, id_min=None, id_max=None):
        subvec = np.diagonal(self.get_sub_Dvec(id_min, id_max),
                             offset=-1)
        unit_vec = subvec / np.linalg.norm(subvec, axis=0)[np.newaxis, :]
        prod = np.sum(unit_vec[:, :-1] * unit_vec[:, 1:], axis=0)
        angles = np.arccos(np.clip(prod, -1.0, 1.0))
        moments = scipy.stats.describe(angles)
        return {'angle_mean': moments.mean, 'angle_var': moments.variance,
                'angle_skewness': moments.skewness,
                'angle_kurtosis': moments.kurtosis,
                'angle_min': moments.minmax[0], 'angle_max': moments.minmax[1]}

    def feat_step_autocorr(self, id_min=None, id_max=None, n_samples=30):
        steps = self.get_sub_steps(id_min, id_max)
        n = len(steps)
        mean_step, var_step = np.mean(steps), np.var(steps)
        tau_int = np.unique(np.linspace(1, n, num=n_samples, endpoint=False)
                            .astype(int))
        correlations = np.array([np.sum(((steps[i:] - mean_step) *
                                         (steps[:-i] - mean_step)), axis=0)
                                 for i in tau_int])
        moments = scipy.stats.describe(correlations / (n * var_step))
        return {'autocorr_mean': moments.mean,
                'autocorr_var': moments.variance,
                'autocorr_skewness': moments.skewness,
                'autocorr_kurtosis': moments.kurtosis,
                'autocorr_min': moments.minmax[0],
                'autocorr_max': moments.minmax[1]}

    def feat_fractal_spectrum(self, id_min=None, id_max=None,
                              n_samples=30, nR=20, qs=None):
        subDabs = self.get_sub_Dabs(id_min, id_max)
        n = len(subDabs)
        chosen_points = np.random.permutation(n)[:min(n, n_samples)]
        subDabs_sampled = subDabs[chosen_points]
        if qs is None:
            D_inf = _Dq(subDabs_sampled, -1.5, n_R=nR)
            D_sup = _Dq(subDabs_sampled, 1.5, n_R=nR)
            Dqgrad0 = 1/3 * (D_sup - D_inf)
            return {'frac_grad0': Dqgrad0}
        else:
            Dq = [_Dq(subDabs_sampled, q, n_R=nR) for q in qs]
            return Dq

    def feat_ergodicity(self, id_min=None, id_max=None):
        subposition = self.get_sub_position(id_min, id_max)
        vecs = [subposition[:, i] for i in range(len(self.dims))]
        Fs1 = np.abs(np.array(list(map(_ergodicity_estimator, vecs))))
        Fs2 = np.abs(np.array(list(map(_ergodicity_estimator_v2, vecs))))
        type1 = {f'ergo_{dim}': Fs1[i] for i, dim in enumerate(self.dims)}
        type2 = {f'ergo2_{dim}': Fs2[i] for i, dim in enumerate(self.dims)}
        return {**type1, **type2}

    def feat_gaussianity(self, id_min=None, id_max=None, n_samples=30):
        subDabs = self.get_sub_Dabs(id_min, id_max)
        ns = np.unique(np.linspace(1, len(subDabs)-1, n_samples).astype(int))
        gauss_n = np.zeros(len(ns))
        for i, n in enumerate(ns):
            d_n = np.diagonal(subDabs, offset=n)
            gauss_n[i] = np.mean(d_n**4) / ((np.mean(d_n**2)**2)) - 1
        grad_gauss_n = np.gradient(gauss_n) if len(gauss_n) > 1 else np.nan
        mean_grad = np.mean(grad_gauss_n)
        mean_gauss = np.mean(gauss_n)
        return {'gaussian_grad': mean_grad, 'gaussian_mean': mean_gauss}

    def get_all_features(self, id_min=None, id_max=None, n_samples=30):
        return {**self.feat_angle(id_min, id_max),
                **self.feat_drift(id_min, id_max),
                **self.feat_ergodicity(id_min, id_max),
                **self.feat_escape_time(id_min, id_max),
                **self.feat_fractal_spectrum(id_min, id_max,
                                             n_samples=n_samples, nR=20),
                **self.feat_gaussianity(id_min, id_max, n_samples=n_samples),
                **self.feat_msd(id_min, id_max, n_samples=n_samples),
                **self.feat_pdf(id_min, id_max, n_samples=n_samples),
                **self.feat_shape(id_min, id_max),
                **self.feat_step(id_min, id_max),
                **self.feat_step_autocorr(id_min, id_max, n_samples=n_samples)}


def feat_step(RW, id_min=None, id_max=None):
    return RW.feat_step(id_min=id_min, id_max=id_max)


def feat_msd(RW, id_min=None, id_max=None, n_samples=30):
    return RW.feat_msd(id_min=id_min, id_max=id_max, n_samples=n_samples)


def feat_drift(RW, id_min=None, id_max=None):
    return RW.feat_drift(id_min=id_min, id_max=id_max)


def feat_pdf(RW, id_min=None, id_max=None, n_samples=30):
    return RW.feat_pdf(id_min=id_min, id_max=id_max, n_samples=n_samples)


def feat_shape(RW, id_min=None, id_max=None):
    return RW.feat_shape(id_min=id_min, id_max=id_max)


def feat_escape_time(RW, id_min=None, id_max=None):
    return RW.feat_escape_time(id_min=id_min, id_max=id_max)


def feat_angle(RW, id_min=None, id_max=None):
    return RW.feat_angle(id_min=id_min, id_max=id_max)


def feat_step_autocorr(RW, id_min=None, id_max=None, n_samples=30):
    return RW.feat_step_autocorr(id_min=id_min, id_max=id_max,
                                 n_samples=n_samples)


def feat_fractal_spectrum(RW, id_min=None, id_max=None, n_samples=30,
                          nR=20, qs=None):
    return RW.feat_fractal_spectrum(id_min=id_min, id_max=id_max,
                                    n_samples=n_samples, nR=nR, qs=qs)


def feat_ergodicity(RW, id_min=None, id_max=None):
    return RW.feat_ergodicity(id_min=id_min, id_max=id_max)


def feat_gaussianity(RW, id_min=None, id_max=None, n_samples=30):
    return RW.feat_gaussianity(id_min=id_min, id_max=id_max,
                               n_samples=n_samples)


def get_all_features(RW, id_min=None, id_max=None, n_samples=30):
    return RW.get_all_features(id_min=id_min, id_max=id_max,
                               n_samples=n_samples)
