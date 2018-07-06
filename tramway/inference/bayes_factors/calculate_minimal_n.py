# -*- coding: utf-8 -*-

# Copyright © 2018, Institut Pasteur
#   Contributor: Alexander Serov

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .calculate_marginalized_integral import calculate_marginalized_integral
from .convenience_functions import *
import numpy as np
import scipy


def calculate_minimal_n(zeta_t, zeta_sp, n0, V, V_pi):
    """
    Calculate the minimal number of jumps per bin to obtain strong evidence for a conservative or a spurious force in the Bayes factor.

    Input:
    zeta_t, zeta_sp --- vectors of length 2 (x, y) in one bin
    n0 --- initial number of jumps (evidence already available). This way the "next" strong evidence can be found, i.e. the minimal number of data points to support the current conclusion

    Output:
    min_n --- minimal number of jumps to obtain strong evidence for the conservative force model.
    Return -1 if unable to find the min_n
    """
    # print("Hello!!!")
    # Local constants
    dim = 2
    n_pi = n_pi_func(dim)
    B_threshold = 10.0
    increase_factor = 2
    max_attempts = 40
    xtol = 0.1
    rtol = 0.1

    # Initialize
    lg_B_threshold = np.log10(B_threshold)
    u = V_pi / V

    def eta(n):
        return np.sqrt(n_pi / (n + n_pi))

    # def p(n):
    # 	return dim * (n + n_pi + 1) / 2 - 2

    def v0(n):
        return 1.0 + n_pi / n * u

    # Define the Bayes factor
    def lg_B(n):
        upstairs = calculate_marginalized_integral(
            zeta_t=zeta_t, zeta_sp=zeta_sp, p=p(n, dim), v=v0(n), E=eta(n)**2.0)
        downstairs = calculate_marginalized_integral(
            zeta_t=zeta_t, zeta_sp=zeta_sp, p=p(n, dim), v=v0(n), E=1.0)
        lg_B = dim * np.log10(eta(n)) + \
            np.log10(upstairs) - np.log10(downstairs)
        return lg_B

    # test1 = [lg_B(n) for n in range(1, 1000)]
    # print(test1)

    # Find initial search interval
    bl_found = False
    for attempt in range(max_attempts):
        # print(attempt)
        n = n0 - 1 + increase_factor ** attempt
        if lg_B(n) >= lg_B_threshold:
            bl_found = True
            break

    if not bl_found:
        return -1

    if attempt == 0:
        return n

    # Construct the initial interval
    n_interval = n0 - 1 + increase_factor ** np.asarray([0.0, attempt])
    lg_B_interval = [lg_B(n) for n in n_interval]

    # Identify whether the spurious or the conservative force model is favored by this n
    sign = np.sign(lg_B_interval[1])

    # Define the function being optimized
    def solve_me(n):
        return lg_B(n) - sign * lg_B_threshold

    # Find root
    min_n = scipy.optimize.brentq(solve_me, n_interval[0], n_interval[1])

    # Round
    min_n = np.int(np.ceil(min_n))
    # print(min_n)

    return min_n
