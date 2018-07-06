# -*- coding: utf-8 -*-

# Copyright © 2018, Institut Pasteur
#   Contributor: Alexander Serov

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
from scipy import integrate


def calculate_marginalized_integral(zeta_t, zeta_sp, p, v, E):
    """
    Calculate the marginalized lambda integral
    >>>
    Integrate[(v + E * (zeta_t - lambda * zeta_sp) **2) ** (-p), {lambda, 0, 1}]
    <<<
    for the given values of zeta_t and zeta_sp.

    Input:
    zeta_t and zeta_sp are vectors of length 2 (x, y).
    All other parameters are scalars.
    """

    # Constants
    tol = 1e-8

    # Convert to numpy
    zeta_t = np.asarray(zeta_t)
    zeta_sp = np.asarray(zeta_sp)

    # If norm(zeta_sp) ~ 0, no need to actually integrate
    if np.linalg.norm(zeta_sp) < tol:
        result = (v + E * (zeta_t @ zeta_t.T)) ** (-p)
        return (result)

    # Check if break points need to be added
    # lambda_breaks_candidates = []
    # ignore if 0 is divided by 0
    with np.errstate(invalid='ignore', divide='ignore'):
        # print(zeta_t, zeta_sp)
        lambda_breaks_candidates = np.divide(zeta_t, zeta_sp)
        # print(lambda_breaks_candidates)
    lambda_breaks = []
    for lb in lambda_breaks_candidates:
        if lb >= 0.0 and lb <= 1.0:
            lambda_breaks.append(lb)

    # Define the integrand function
    def integrate_me(l):
        diff = (l * zeta_sp - zeta_t)
        return ((v + E * (diff @ diff.T)) ** (-p))

    # Perform integration
    result = integrate.quad(integrate_me, 0.0, 1.0,
                            points=lambda_breaks, full_output=0)

    return(result[0])
