# -*- coding: utf-8 -*-

# Copyright © 2018, Alexander Serov

import logging

import numpy as np
from numpy import exp, log
from scipy import integrate
from scipy.special import gamma, gammainc, gammaln

# Constants

# prec = 1e-32
# max_terms = int(1e5)


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
    rel_loc_error = n * V / (4 * sigma_L^2) --- inverse relative localization error,


    Input:
    zeta_t and zeta_sp are vectors of length dim (x, y, z).
    All other parameters are scalars.
    abs_tol --- absolute tolerance for integral calculations.
    lamb = float in [0, 1] or 'int'. If a float is given, the integrated function is just evaluated at the given value of lambda, without integrating.
    """
    atol = 1e-16

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
    if lamb != 'int':
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


def calculate_integral_ratio(arg_func_up, arg_func_down, pow_up, pow_down, v, rel_loc_error, break_points=[], lamb='marg', rtol=1e-6, atol=1e-32):
    """
    Calculate the ratio of two similar lambda integrals, each of which has the form
    \int_0^1 d\lambda arg_func(\lambda)**(-pow) * gammainc(pow, rel_loc_error * arg_func(\lambda))

    Input:
    v --- zeta-independent term inside the argument functions. Must be the same upstairs and downstairs. In Bayes factor calculations, this is v := (1 + n_pi * V_pi / n / V).

    Return:
    Natural logarithm of the integral ratio

    Details:
    Calculations are performed differently for the case with and without localization error. This avoids overflow errors. In these cases, both the integrated equations and prefactors are different.
    The integrals are taken over lambda if lamb == 'marg', otherwise they are evaluated at the given lambda.
    """

    x0u = arg_func_up(lamb) if lamb != 'marg' else arg_func_up(0.5)
    x0d = arg_func_down(lamb) if lamb != 'marg' else arg_func_down(0.5)

    if np.any(np.array([pow_up, pow_down]) <= 0) or np.any(np.isnan([pow_up, pow_down])):
        logging.error(
            'Incorrect powers supplied for integral ratio calculations: {}'.format([pow_up, pow_down]))
        return np.nan

    # %% loc_error > 0
    def get_f_with_loc_error(arg_func, pow, x0):
        """Prepare the function for lambda integration.
        Switch summation algorithm based on the argument value at lambda = lamb or 1/2 for marginalized.

        Return:
        f --- a function to integrate,
        ln_prefactor --- a scaling prefactor
        """

        def term(i, l):
            q = arg_func(l)
            res = gammaln(pow) - gammaln(pow + 1 + i) + i * \
                log(q * rel_loc_error) - q * rel_loc_error
            return exp(res)

        q_eval = arg_func(lamb) if lamb != 'marg' else arg_func(0.5)
        q_eval *= rel_loc_error
        if q_eval < pow + 1:
            # print('Low x regime', pow, q_eval)

            def f(l):
                return sum_series(term, rtol, l)
            ln_prefactor = pow * log(rel_loc_error)
            return f, ln_prefactor
        else:
            def f(l):
                q = arg_func(l)
                res = -pow * (log(q) - log(x0)) + log(gammainc(pow, q * rel_loc_error))
                return exp(res)
            ln_prefactor = gammaln(pow) - pow * log(x0)
            return f, ln_prefactor

    def ln_integral_with_loc_error(arg_func, pow, x0):
        """Calculate the following expression for a marginalized or fixed lambda:
        v**k / Gamma[k] * \int_0^1 d l q(l)**(-k) gammainc(k, q(l) * rel_loc_error),
        with q(l) = arg_func(l)
        """
        f, ln_prefactor = get_f_with_loc_error(arg_func, pow, x0)
        if lamb == 'marg':
            ln_res = log(integrate.quad(f, 0, 1, points=break_points, epsrel=rtol)[0])
        else:
            ln_res = log(f(lamb))
        return ln_res + ln_prefactor

    # %% loc_error == 0
    def ln_integral_no_error(arg_func, pow, x0):
        ln_prefactor = gammaln(pow) - pow * log(x0)

        def ln_f(l):
            return -pow * log(arg_func(l) / x0)

        def f(l):
            return exp(ln_f(l))
        if lamb == 'marg':
            ln_res = integrate.quad(f, 0, 1, points=break_points, epsrel=rtol)[0]
            ln_res = log(ln_res)
        else:
            ln_res = ln_f(lamb)
        return ln_res + ln_prefactor

    # %% Compile the ratio
    if ~np.isinf(rel_loc_error):
        log_res = (ln_integral_with_loc_error(arg_func_up, pow_up, x0u)
                   - ln_integral_with_loc_error(arg_func_down, pow_down, x0d))
    else:
        log_res = (ln_integral_no_error(arg_func_up, pow_up, x0u) -
                   ln_integral_no_error(arg_func_down, pow_down, x0d))
    return log_res


def sum_series(term_func, rtol, *args):
    """Sum any series with the first argument of the term_func being the term number.
    Input: function, return - float
    """
    max_terms = int(1e5)
    sum = term = term_func(0, *args)
    for i in range(1, max_terms + 1):
        term = term_func(i, *args)
        sum += term
        if abs(term) <= rtol * abs(sum):
            return sum
    logging.warning(
        'Gamma function series did not converge to the required precision. Achieved precision: {}'.format(abs(term / sum)))
    return sum
