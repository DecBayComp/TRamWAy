# -*- coding: utf-8 -*-

# Copyright © 2018, Alexander Serov


import warnings

# from multiprocessing import Pool
import numpy as np
import scipy.optimize
from .calculate_marginalized_integral import calculate_marginalized_integral
from .convenience_functions import n_pi_func
from .convenience_functions import p as pow
from .stopwatch import stopwatch

# from scipy.special import gammainc


try:
    from tqdm import trange  # for graphical estimate of the progress
except ImportError:
    trange = range


# def calculate_bayes_factors_for_cells(cells, loc_error, dim=2, B_threshold=10, verbose=True):
#     """Calculate Bayes factors for an iterable ensemble of cells."""
#
#     check_dimensionality(dim)
#
#     # M = len(cells)
#     lg_Bs, forces, min_ns = [], [], []
#     for i, cell in enumerate(cells):
#         lg_Bs[i], forces[i], min_ns[i] = calculate_bayes_factors_for_one_cell(
#             cell, loc_error, dim=dim, B_threshold=B_threshold, verbose=verbose)
#         cell.lg_B, cell.force, cell.min_n = lg_Bs[i], forces[i], min_ns[i]
#
#     return [lg_Bs, forces, min_ns]


def calculate_bayes_factors_for_one_cell(cell, loc_error, dim=2, B_threshold=10, verbose=True):
    """Calculate Bayes factors for one cell.
    The results are saved to cell properties.
    """

    check_dimensionality(dim)

    cell.lg_B, cell.force, cell.min_n = _calculate_one_bayes_factor(
        zeta_t=cell.zeta_total,
        zeta_sp=cell.zeta_spurious,
        n=cell.n,
        V=cell.V,
        V_pi=cell.V_prior,
        loc_error=loc_error,
        dim=dim,
        bl_need_min_n=True)

    return [cell.lg_B, cell.force, cell.min_n]


def calculate_bayes_factors(zeta_ts, zeta_sps, ns, Vs, Vs_pi, loc_error, dim=2, B_threshold=10, verbose=True):
    """
    Calculate the Bayes factor for a set of bins given a uniform localization error.

    Input:
    zeta_ts --- signal-to-noise ratios for the total force = dx_mean / sqrt(var(dx)) in bins. Size: M x 2,
    zeta_sps --- signal-to-noise ratios for the spurious force = grad(D) / sqrt(var(dx)) in bins. Size: M x 2,
    ns --- number of jumps in each bin. Size: M x 1,
    Vs --- jump variance in bins = E((dx - dx_mean) ** 2). Size: M x 1,
    Vs_pi --- jump variance in all other bins relative to the current bin. Size: M x 1,
    loc_error --- localization error. Same units as variance. Set to 0 if localization error can be ignored;
    dim --- dimensionality of the problem;
    B_threshold --- the values of Bayes factor for thresholding.

    Output:
    Bs, forces, min_ns

    lg_Bs --- log_10 Bayes factor values in the bins. Size: M x 1,
    forces --- Returns 1 if there is strong evidence for the presence of a conservative forces,
    -1 for strong evidence for 	a spurious force, and 0 if the is not enough evidence. Size: M x 1;
    min_ns --- minimum number of data points in a bin to support the currently favored model at the required evidence level. Size: M x 1.

    Notation:
    M --- number of bins.
    """
    check_dimensionality(dim)

    # Convert to numpy
    vars = [zeta_ts, zeta_sps, Vs, Vs_pi, ns]
    vars = map(np.asarray, vars)

    # Check that the 2nd dimension has size 2
    if np.shape(zeta_ts)[1] != 2 or np.shape(zeta_sps)[1] != 2:
        raise ValueError("zeta_ts and zeta_sps must be matrices of size (M x 2)")

    M = len(ns)

    # Calculate
    lg_Bs = np.zeros_like(ns) * np.nan
    forces = np.zeros_like(ns) * np.nan
    min_ns = np.zeros_like(ns, dtype=int) * np.nan
    with stopwatch("Bayes factor calculation", verbose):
        for i in trange(M):
            lg_Bs[i], forces[i], min_ns[i] = _calculate_one_bayes_factor(
                zeta_ts[i, :], zeta_sps[i, :], ns[i], Vs[i], Vs_pi[i], loc_error, dim)

        return [lg_Bs, forces, min_ns]


def _calculate_one_bayes_factor(zeta_t, zeta_sp, n, V, V_pi, loc_error, dim, B_threshold=10, bl_need_min_n=True):
    """Calculate the Bayes factor for one bin."""

    # Parameter combinations
    n_pi = n_pi_func(dim)
    p = pow(n, dim)
    u = V_pi / V
    v = 1.0 + n_pi / n * u
    eta = np.sqrt(n_pi / (n + n_pi))

    if loc_error > 0:
        rel_loc_error = n * V / (2 * dim * loc_error)
    else:
        rel_loc_error = np.inf

    upstairs = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                               p=p, v=v, E=eta**2, rel_loc_error=rel_loc_error)
    downstairs = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                                 p=p, v=v, E=1.0, rel_loc_error=rel_loc_error)
    lg_B = (dim * np.log10(eta) + np.log10(upstairs) - np.log10(downstairs))

    if bl_need_min_n:
        min_n = calculate_minimal_n(
            zeta_t=zeta_t, zeta_sp=zeta_sp, n0=n, V=V, V_pi=V_pi, loc_error=loc_error, dim=dim, B_threshold=B_threshold)
    else:
        min_n = None

    # Threshold
    force = 1 * (lg_B >= np.log10(B_threshold)) - 1 * (lg_B <= -np.log10(B_threshold))

    return [lg_B, force, min_n]


def calculate_minimal_n(zeta_t, zeta_sp, n0, V, V_pi, loc_error, dim=2, B_threshold=10):
    """
    Calculate the minimal number of jumps per bin needed to obtain strong evidence for the active force or the spurious force model.

    Input:
    zeta_t, zeta_sp - -- vectors of length 2 for one bin
    n0 - -- initial number of jumps(evidence already available). This way the "next" strong evidence can be found, i.e. the minimal number of data points to support the current conclusion

    Output:
    min_n - -- minimal number of jumps to obtain strong evidence for the conservative force model.
    Return - 1 if unable to find the min_n
    """

    # Local constants
    increase_factor = 2  # initial search interval increase with each iteration
    max_attempts = 40
    xtol = 1.0
    rtol = 0.001

    lg_B_threshold = np.log10(B_threshold)

    # Define the Bayes factor
    def lg_B(n):
        """A wrapper for the Bayes factor as a function of n."""
        lg_B, _, _ = _calculate_one_bayes_factor(
            zeta_t=zeta_t, zeta_sp=zeta_sp, n=n, V=V, V_pi=V_pi, loc_error=loc_error, dim=dim, bl_need_min_n=False)
        return lg_B

    if lg_B(n0) >= lg_B_threshold:
        return n0

    # Find the initial search interval
    bl_found = False
    n = n0
    for attempt in range(max_attempts):
        n = n0 - 1 + increase_factor ** attempt
        if lg_B(n) >= lg_B_threshold:
            bl_found = True
            break

    if not bl_found:
        warnings.warn("Unable to find the minimal number of data points to provide strong evidence.")
        return -1

    # Find a more accurate location
    n_interval = [n0, n]
    sign = np.sign(lg_B(n))

    def solve_me(n):
        return lg_B(n) - sign * lg_B_threshold

    min_n = scipy.optimize.brentq(solve_me, n_interval[0], n_interval[1], xtol=xtol, rtol=rtol)
    min_n = np.int(np.ceil(min_n))

    return min_n


def check_dimensionality(dim):
    if dim not in [2]:
        raise ValueError("Bayes factor calculations in {dim}D not supported yet.".format(dim=dim))
