"""
Functions allowing to calculate the D posterior and MAP(D) as (will be) described in the Ito-Stratonovich article.
Provides 3 functions:
- get_D_posterior
- get_MAP_D
- get_D_confidence_interval

If unsure, use `get_D_confidence_interval`, which provides MAP(D) and a confidence interval for D for the given confidence level.
If need the D posterior, use `get_D_posterior`.
"""


import logging
import warnings

import numpy as np
from numpy import exp as exp
from numpy import log as log
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincc, gammaln

from .convenience_functions import n_pi_func
from .convenience_functions import p as pow

# prior assumption for diffusivity inference that does not include D'.
# Should be described in the article Appendix


def get_D_posterior(n, zeta_t, V, V_pi, dt, sigma2, dim, _MAP_D=False):
    """Return the diffusivity posterior as a function.
    If _MAP_D flag is supplied, returns instead a single MAP(D) value

    Input:
    V   -   (biased) variance of jumps in the current bin
    V_pi    -   (biased) variance of jumps in all other bins excluding the current one
    dt  -   time step
    sigma2  -   localization error (in the units of variance)
    dim -   dimensionality of the problem
    _MAP_D  -   if True, returns MAP_D instead of the posterior

    Output:
    posterior    -   function object accepting D as the only argument
    """

    n_pi = n_pi_func(dim)
    p = pow(n, dim)
    v = 1.0 + n_pi / n * V_pi / V
    eta = np.sqrt(n_pi / (n + n_pi))
    G3 = v + eta**2 * _norm2(zeta_t - _zeta_mu(dim))

    if _MAP_D:
        return n * V * G3 / 4 / dt / (p + 1)

    # Conversions
    zeta_t = np.array(zeta_t)
    rel_loc_error = n * V / (4 * sigma2) if sigma2 > 0 else np.inf

    def posterior(D):
        """Remember gammainc is normalized to gamma"""
        min_D = sigma2 / dt
        if D <= min_D:
            ln_res = -np.inf
        else:
            ln_res = (p * log(n * V * G3 / (4 * dt * D))
                      - n * V * G3 / (4 * D * dt)
                      - log(D)
                      - log(gammainc(p, rel_loc_error * G3))
                      - gammaln(p))
        return np.exp(ln_res)
    return posterior


def get_MAP_D(n, zeta_t, V, V_pi, dt, sigma2, dim):
    return get_D_posterior(n, zeta_t, V, V_pi, dt, sigma2, dim, _MAP_D=True)


def get_D_confidence_interval(alpha, n, zeta_t, V, V_pi, dt, sigma2, dim):
    """Returns MAP_D and a confidence interval corresponding to confidence level alpha, i.e. the values of D giving the values [(1-alpha)/2, 1-(1-alpha)/2] for the D posterior integrals

    Input:
    alpha   -   confidence level,
    n       -   number of jumps in bin,
    zeta_t  -   signal-to-noise ratio for the total force: <dr>/sqrt(V),
    V       -   (biased) variance of jumps in the current bin,
    V_pi    -   (biased) variance of jumps in all other bins excluding the current one,
    dt      -   time step
    sigma2  -   localization error (in the units of variance),
    dim     -   dimensionality of the problem

    Output:
    MAP_D   -   MAP value of the diffusivity,
    CI      -   a confidence interval for the diffusivity, 2x1 np.array

    """
    n_pi = n_pi_func(dim)
    p = pow(n, dim)
    v = 1.0 + n_pi / n * V_pi / V
    eta = np.sqrt(n_pi / (n + n_pi))
    zeta_t = np.array(zeta_t)
    G3 = v + eta**2 * _norm2(zeta_t - _zeta_mu(dim))
    min_D = sigma2 / dt    # units of variance /s
    MAP_D = get_MAP_D(n, zeta_t, V, V_pi, dt, sigma2, dim)
    y_L = n * V * G3 / 4 / sigma2

    def posterior_integral(D):
        y = n * V * G3 / 4 / dt / D
        if D <= min_D:
            intg = 0
        else:
            intg = 1 - gammainc(p, y) / gammainc(p, y_L)
        return intg

    # %% Find root
    # Maximal D up to which to look for a root, units of variance /s
    # Calculated as asymptotic root times 2
    # TODO may result in error if low alpha is given as input
    q = 1 - (1 - alpha) / 2
    max_D_search = (-1 / p) * (log(p) + log(1 - q) + log(gammainc(p, y_L)) - gammaln(p))
    max_D_search = n * V * G3 / 4 / dt / exp(max_D_search)
    max_D_search *= 2
    max_D_search = np.max([max_D_search, 1e3])

    CI = np.ones(2) * np.nan
    for i, q in enumerate([(1 - alpha) / 2, 1 - (1 - alpha) / 2]):

        def solve_me(D):
            return posterior_integral(D) - q
        CI[i] = brentq(solve_me, min_D, max_D_search)

    return MAP_D, CI


def _zeta_mu(dim):
    """The center of the prior for the total force.
    For diffusivity inference should not depend on the diffusivity gradient.
    """
    return np.zeros(dim)


def _norm2(x):
    return np.sum(x**2)
