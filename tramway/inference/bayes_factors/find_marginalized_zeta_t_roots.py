# -*- coding: utf-8 -*-

# Copyright © 2018, Alexander Serov

# from log_C import log_C as log_C_func
import numpy as np
from scipy import optimize

from .calculate_marginalized_integral import calculate_marginalized_integral
from .convenience_functions import p


def find_marginalized_zeta_t_roots(zeta_sp_par, n, n_pi, B, u, dim, zeta_t_perp, sigma2):
    """
    Find marginalized roots zeta_t.
    I have proven that the min B for the marginalized inference is achieved at zeta_t = zeta_sp/2 (for the parallel component and under the condition that the orthogonal component is 0).
    sigma2 --- localization error. Same units as variance. Set to 0 if localization error can be ignored;
    """

    # Constants
    max_init_search_attempts = 50
    increase_factor = 2.0

    # # Check input
    # if not isinstance(zeta_sp, list):
    # 	raise TypeError("'zeta_sp' must be a 1D list.")

    eta = np.sqrt(n_pi / (n + n_pi))
    # p = dim * (n + n_pi + 1.0) / 2.0 - 2.0
    pow = p(n, dim)

    # # Define v function
    # def v(s):
    #     return(1.0 + n_pi / n * u + (dim - 1.0) * s * zeta_t_perp ** 2)
    v = 1.0 + n_pi / n * u

    rel_loc_error = n * V / (4 * sigma2) if sigma2 > 0 else np.inf

    # Function to optimize
    def solve_me(zeta_t_par):
        zeta_sp = np.asarray([zeta_sp_par, 0])
        zeta_t = np.asarray([zeta_t_par, zeta_t_perp])

        E = eta ** 2.0
        # print([zeta_t_cur], [zeta_sp], p, v(E), E)
        # calculate_marginalized_integral takes [2,1] vectors as input that include the parallel and othogonal components
        upstairs = calculate_marginalized_integral(zeta_t, zeta_sp, pow, v, E, rel_loc_error)

        E = 1.0
        downstairs = calculate_marginalized_integral(zeta_t, zeta_sp, pow, v, E, rel_loc_error)
        return upstairs * eta ** dim - B * downstairs

    # Guess sign change intervals centered around zeta_t = zeta_sp / 2
    zeta_t_lims = np.array([0.0, 0.5, 1.0]) * zeta_sp_par
    zeta_t_width = zeta_t_lims[-1] - zeta_t_lims[0]
    min_value = solve_me(zeta_t_lims[1])

    # If the minimized function is negative for the middle point we won't be able to find roots
    # print(min_value)
    # print([solve_me(zeta_t_lims[0]), solve_me(zeta_t_lims[-1])])
    if min_value > 0:
        # print("Warning: No marginalized zeta_t roots exist for zeta_sp = %.2f. Min Bayes factor value (%.3g) is always greater than the requested value (%.3g) for any zeta_t." % (zeta_sp, min_value + K, K))
        return [np.nan, np.nan]

    attempt = 0
    bl_interval_found = False
    while(attempt < max_init_search_attempts):
        # Check if the end points have a different sign from the middle
        end_values = [solve_me(zeta_t_lims[0]), solve_me(zeta_t_lims[-1])]

        if end_values[0] * min_value < 0 and end_values[1] * min_value < 0:
            # The interval found
            bl_interval_found = True
            break
        else:
            # Increase interval
            zeta_t_width = zeta_t_width * increase_factor
            zeta_t_lims[0] = zeta_t_lims[1] - 0.5 * zeta_t_width
            zeta_t_lims[-1] = zeta_t_lims[1] + 0.5 * zeta_t_width
        attempt += 1

    # Check if the interval was found
    if not bl_interval_found:
        raise AssertionError(
            "Failed to converge to initial search interval for the marginalized roots")

    # print (end_values, min_value)
    # print(zeta_t_lims)
    # print(attempt)

    # If the initial intervals found, find the two roots
    zeta_t_roots = []

    # Find the smaller root
    for i in [0, 1]:
        zeta_t_roots.append(optimize.brentq(
            solve_me, zeta_t_lims[i], zeta_t_lims[i + 1]))

    # Sort roots
    zeta_t_roots.sort()
    # print(zeta_t_roots)

    return(zeta_t_roots)
