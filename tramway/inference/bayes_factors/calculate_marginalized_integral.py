# -*- coding: utf-8 -*-

# Copyright © 2018, Alexander Serov

import logging

import numpy as np
from numpy import exp, log
from scipy import integrate
from scipy.special import gamma, gammainc, gammaln

# Constants
atol = 1e-8


def calculate_marginalized_integral(zeta_t, zeta_sp, p, v, E, rel_loc_error, zeta_a=[0, 0], factor_za=0, lamb='int'):
    """
    Calculate the marginalized lambda integral
    >>>
    Integrate[gamma_inc[p, arg * rel_loc_error] * arg ** (-p), {lambda, 0, 1}]
    >>>
    for the given values of zeta_t and zeta_sp.

    Here:
    arg = (v + factor_za * zeta_a**2 + E * (zeta_t - zeta_a - lambda * zeta_sp)**2);
    IMPORTANT: gamma_inc --- is the normalized lower incomplete gamma function;
    rel_loc_error = n * V / (2 * dim * sigma_L^2) --- inverse relative localization error,


    Input:
    zeta_t and zeta_sp are vectors of length dim (x, y, z).
    All other parameters are scalars.
    abs_tol --- absolute tolerance for integral calculations.
    lamb = float in [0, 1] or 'int'. If a float is given, the integrated function is just evaluated at the given value of lambda, without integrating.
    """

    zeta_t, zeta_sp, zeta_a = map(np.asarray, [zeta_t, zeta_sp, zeta_a])
    if zeta_t.ndim > 1 or zeta_sp.ndim > 1:
        raise ValueError("zeta_t and zeta_sp should be 1D vectors")

    zeta_a_norm_sq = np.sum(zeta_a**2)

    def arg(l):
        """lambda-dependent argument of the gamma function and the second term."""
        diff = zeta_t - zeta_a - l * zeta_sp
        # the @ operator raises a syntax error in Py2
        return v + factor_za * zeta_a_norm_sq + E * np.matmul(diff, diff.T)

    def get_integrate_me():
        """Function to integrate with and without localization error."""
        def no_error(l):
            return arg(l) ** (-p)

        def with_error(l):
            return gammainc(p, arg(l) * rel_loc_error) * arg(l) ** (-p)

        if rel_loc_error == np.inf:
            return no_error
        else:
            return with_error

    integrate_me = get_integrate_me()

    # Skip integration if zeta_sp is close to 0
    if np.isclose(np.linalg.norm(zeta_sp), 0, atol=atol):
        return integrate_me(0)
    # Skip integration if a fixed-lambda convention is used
    if lamb is not 'int':
        return integrate_me(lamb)

    # Calculate break points
    with np.errstate(invalid='ignore', divide='ignore'):
        lambda_breaks = [lb for lb in np.divide(
            zeta_t - zeta_a, zeta_sp) if lb >= 0.0 and lb <= 1.0]

    # Perform integration
    result = integrate.quad(integrate_me, 0.0, 1.0, points=lambda_breaks,
                            full_output=False, epsabs=atol)
    return result[0]


def calculate_any_lambda_integral(func, break_points=[]):
    """
    DEPRECATED: calculate ratios instead.
    This function allows to calculate more complicated lambda integrals, such as 1D posteriors in 2D.

    Input:
    func(lambda) - function of lambda to integrate,
    break_points - special integration points (i.e. zeta_t/zeta_sp), optional.
    """
    #
    # zeta_t, zeta_sp, zeta_a = map(np.asarray, [zeta_t, zeta_sp, zeta_a])
    # if zeta_t.ndim > 1 or zeta_sp.ndim > 1:
    #     raise ValueError("zeta_t and zeta_sp should be 1D vectors")
    #
    # zeta_a_norm_sq = np.sum(zeta_a**2)

    # def arg(l):
    #     """lambda-dependent argument of the gamma function and the second term."""
    #     diff = zeta_t - zeta_a - l * zeta_sp
    #     # the @ operator raises a syntax error in Py2
    #     return v + factor_za * zeta_a_norm_sq + E * np.matmul(diff, diff.T)
    #
    # def get_integrate_me():
    #     """Function to integrate with and without localization error."""
    #     def no_error(l):
    #         return arg(l) ** (-p)
    #
    #     def with_error(l):
    #         return gammainc(p, arg(l) * rel_loc_error) * arg(l) ** (-p)
    #
    #     if rel_loc_error == np.inf:
    #         return no_error
    #     else:
    #         return with_error
    #
    # integrate_me = get_integrate_me()

    # # Skip integration if zeta_sp is close to 0
    # if np.isclose(np.linalg.norm(zeta_sp), 0, atol=atol):
    #     return integrate_me(0)
    # # Skip integration if a fixed-lambda convention is used
    # if lamb is not 'int':
    #     return integrate_me(lamb)

    # Verify break points
    with np.errstate(invalid='ignore', divide='ignore'):
        lambda_breaks = [lb for lb in break_points if lb >= 0.0 and lb <= 1.0]

    # Perform integration
    result = integrate.quad(func, 0.0, 1.0, points=lambda_breaks,
                            full_output=False, epsabs=atol)
    return result[0]


def calculate_integral_ratio(arg_func_up, arg_func_down, pow_up, pow_down, v, rel_loc_error, break_points=[], lamb='marg'):
    """
    Calculate the ratio of two similar lambda integrals, each of which has the form
    \int_0^1 d\lambda arg_func(\lambda)**(-pow) * gammainc(pow, rel_loc_error * arg_func(\lambda))

    Return:
    Natural logarithm of the integral ratio

    Details:
    Calculations are performed differently for the case with and without localization error. This avoids overflow errors. In these cases, both the integrated equations and prefactors are different.
    The integrals are taken over lambda if lamb == 'marg', otherwise they are evaluated at the given lambda.
    """
    # %% loc_error != 0 case
    def one_term_with_loc_error(l, arg_func, pow, i):
        log_res = gammaln(pow) - gammaln(pow + 1 + i)
        log_res += - rel_loc_error * arg_func(l) + i * log(rel_loc_error * arg_func(l))
        return exp(log_res)

    def one_term_integral_with_loc_error(arg_func, pow, i):
        def f(l):
            return one_term_with_loc_error(l, arg_func, pow, i)
        if lamb is 'marg':
            return integrate.quad(f, 0, 1, points=break_points)[0]
        else:
            return f(lamb)

    def sum_series_with_loc_error(arg_func, pow):
        """
        Sums series of the form
        \sum_{i=0}^\infty one_term_integral_with_loc_error(arg_func, pow, i)
        """
        rtol = 1e-5
        max_terms = int(1e5)

        sum = term = one_term_integral_with_loc_error(arg_func, pow, 0)
        for i in range(1, max_terms + 1):
            term = one_term_integral_with_loc_error(arg_func, pow, i)
            sum += term
            if abs(term) <= rtol * abs(sum):  # and i >= min_terms:
                return sum
        logging.warning(
            'Gamma function series did not converge to the required precision. Achieved precision: {}'.format(abs(term / sum)))
        return sum

    # %% loc_error == 0
    def integral_no_error(arg_func, pow):
        def f(l):
            return exp(-pow * log(arg_func(l) / v))
        if lamb is 'marg':
            sum = integrate.quad(f, 0, 1, points=break_points)[0]
        else:
            sum = f(lamb)
        return sum

    # %% Compile the ratio
    if ~np.isinf(rel_loc_error):
        log_res = (log(sum_series_with_loc_error(arg_func_up, pow_up))
                   - log(sum_series_with_loc_error(arg_func_down, pow_down)))
        log_res += log(rel_loc_error) / 2
    else:
        log_res = (log(integral_no_error(arg_func_up, pow_up)) -
                   log(integral_no_error(arg_func_down, pow_down)))
        log_res += gammaln(pow_up) - gammaln(pow_down) - log(v) / 2

    return log_res
