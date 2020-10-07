# -*- coding: utf-8 -*-

# Copyright © 2018, Alexander Serov
"""
The file contains posterior calculation functions.
"""

import numpy as np
from numpy import exp, log, pi
from scipy import integrate
from scipy.special import gamma, gammainc, gammaln

from .calculate_marginalized_integral import (calculate_any_lambda_integral,
                                              calculate_integral_ratio,
                                              calculate_marginalized_integral)
from .convenience_functions import n_pi_func
from .convenience_functions import p as pow


def calculate_one_2D_posterior(zeta_t, zeta_sp, zeta_a, n, V, V_pi, loc_error, dim=2, lamb='marg'):
    """
    Calculate the posterior for zeta_a for a given method for one bin in 2D

    Input:
    lambda --- float in [0, 1] or 'marg' --- for different conventions
    """

    # Check if None is present
    if any(var is None for var in [zeta_t, zeta_sp, n, V, V_pi, loc_error]):
        logging.info('None values encountered in cell. Skipping cell.')
        return [np.nan] * 3

    if lamb == 'marg':
        lamb = 'int'

    if loc_error > 0:
        rel_loc_error = n * V / (4 * loc_error)
    else:
        rel_loc_error = np.inf

    # Parameter combinations
    n_pi = n_pi_func(dim)
    p = pow(n, dim)
    p_upstairs = p + dim / 2
    u = V_pi / V
    v = 1.0 + n_pi / n * u
    eta = np.sqrt(n_pi / (n + n_pi))
    factor_za = eta**2 / (1 - eta**2)

    prefactor = (np.pi * (1 - eta**2))**(-dim / 2)

    upstairs = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                               p=p_upstairs, v=v, E=1.0, rel_loc_error=rel_loc_error, zeta_a=zeta_a, factor_za=factor_za, lamb=lamb)
    downstairs = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                                 p=p, v=v, E=eta**2, rel_loc_error=rel_loc_error, zeta_a=[0, 0], factor_za=0, lamb=lamb)
    gamma_ratio = gamma(p_upstairs) / gamma(p)
    res = prefactor * upstairs / downstairs * gamma_ratio

    return res


def calculate_one_1D_posterior_in_2D(zeta_a, zeta_t, zeta_sp, n, V, V_pi, loc_error, lamb='marg', axis='x'):
    """
    Calculate the posterior for a projection of zeta_a for a given inference convention for one bin in 2D

    Input:
    lambda: float in [0, 1] or 'marg'; for different conventions
    axis: ['x', 'y']; for which projection to calculate the posterior
    zeta_a: float; `a` projection on the selected axis
    """
    dim = 2
    # Check if None is present
    if any(var is None for var in [zeta_t, zeta_sp, n, V, V_pi, loc_error]):
        logging.info('None values encountered in cell. Skipping cell.')
        return [np.nan] * 3

    zeta_t, zeta_sp, zeta_a = map(np.asarray, [zeta_t, zeta_sp, zeta_a])
    i_post, i_marg = (0, 1) if axis == 'x' else (1, 0)
    rel_loc_error = n * V / (4 * loc_error) if loc_error > 0 else np.inf

    # Parameter combinations
    n_pi = n_pi_func(dim)
    p = pow(n, dim)
    p_upstairs = p + 1 / 2
    v = 1.0 + n_pi * V_pi / n / V
    log_prefactor = (log(n + n_pi) - log(n) - log(pi)) / 2

    def upstairs(l):
        arg = (v
               + n_pi / n * zeta_a**2
               + (zeta_t[i_post] - zeta_a - l * zeta_sp[i_post])**2
               + n_pi / (n + n_pi) * (zeta_t[i_marg] - l * zeta_sp[i_marg])**2)
        return arg

    def downstairs(l):
        arg = (v
               + n_pi / (n + n_pi) * np.sum((zeta_t - l * zeta_sp)**2))
        return arg

    # Identify break points
    with np.errstate(invalid='ignore', divide='ignore'):
        bps = set([(zeta_t[i_post] - zeta_a) / zeta_sp[i_post],
                   zeta_t[i_marg] / zeta_sp[i_marg]])
        bps |= set(zeta_t / zeta_sp)
    bps = [bp for bp in bps if bp >= 0 and bp <= 1]

    # Final calculations
    log_ratio = calculate_integral_ratio(arg_func_up=upstairs, arg_func_down=downstairs,
                                         pow_up=p_upstairs, pow_down=p, v=v,
                                         rel_loc_error=rel_loc_error, break_points=bps,
                                         lamb=lamb)
    return exp(log_prefactor + log_ratio)


def calculate_one_1D_prior_in_2D(zeta_a, V_pi, sigma2):
    """
    Calculate the prior for a projection of zeta_a for any convention for one bin in 2D

    Input:
    zeta_a: float; `a` projection on the selected axis
    """
    dim = 2

    zeta_a = np.asarray(zeta_a)
    n_pi = n_pi_func(dim)
    rel_loc_error = n_pi * V_pi / (4 * sigma2) if sigma2 > 0 else np.inf

    # Parameter combinations
    p = n_pi - 2
    p_upstairs = p + 1 / 2
    log_prefactor = - log(pi) / 2

    def upstairs(l):
        return 1 + zeta_a**2

    def downstairs(l):
        return 1

    log_ratio = calculate_integral_ratio(arg_func_up=upstairs, arg_func_down=downstairs,
                                         pow_up=p_upstairs, pow_down=p, v=1,
                                         rel_loc_error=rel_loc_error, break_points=[], lamb=0)
    r = log_prefactor + log_ratio
    return exp(r)


def get_lambda_MAP(zeta_t, zeta_sp):
    """
    Calculate the maximum a posteriori value of lambda for given zetas in 1D and 2D
    """
    lamb = np.dot(zeta_t, zeta_sp) / np.linalg.norm(zeta_sp)**2

    # Limit to the region where the prior is defined
    if lamb > 1:
        lamb = 1
    if lamb < 0:
        lamb = 0

    return lamb
