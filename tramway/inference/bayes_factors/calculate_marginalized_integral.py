# -*- coding: utf-8 -*-

# Copyright © 2018, Alexander Serov

import numpy as np
from scipy import integrate
from scipy.special import gammainc


def calculate_marginalized_integral(zeta_t, zeta_sp, p, v, E, rel_loc_error):
    """
    Calculate the marginalized lambda integral
    >>>
    Integrate[gamma_inc[p, arg * rel_loc_error] * arg ** (-p), {lambda, 0, 1}]
    >>>
    for the given values of zeta_t and zeta_sp.

    Here:

    arg = (v + E * (zeta_t - lambda * zeta_sp)**2);
    gamma_inc --- is the normalized lower incomplete gamma function;
    rel_loc_error = n * V / (2 * dim * sigma_L^2) --- inverse relative localization error,

    Input:
    zeta_t and zeta_sp are vectors of length dim (x, y, z).
    All other parameters are scalars.
    abs_tol --- absolute tolerance for integral calculations.
    """

    # Constants
    atol = 1e-8

    zeta_t, zeta_sp = map(np.asarray, [zeta_t, zeta_sp])
    if zeta_t.ndim > 1 or zeta_sp.ndim > 1:
        raise ValueError("zeta_t and zeta_sp should be 1D vectors")

    def arg(l):
        """lambda-dependent argument of the gamma function and the second term."""
        diff = l * zeta_sp - zeta_t
        return v + E * np.matmul(diff, diff.T) # the @ operator raises a syntax error in Py2

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

    # Calculate break points
    with np.errstate(invalid='ignore', divide='ignore'):
        lambda_breaks = [lb for lb in np.divide(zeta_t, zeta_sp) if lb >= 0.0 and lb <= 1.0]

    # Perform integration
    result = integrate.quad(integrate_me, 0.0, 1.0, points=lambda_breaks,
                            full_output=False, epsabs=atol,)
    return result[0]
