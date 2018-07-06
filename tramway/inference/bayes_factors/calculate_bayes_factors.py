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
from .calculate_minimal_n import calculate_minimal_n
from .convenience_functions import *
import numpy as np
from tqdm import *  # for graphical estimate of the progress


def calculate_bayes_factors(zeta_ts, zeta_sps, ns, Vs, Vs_pi, B_threshold=10):
    """
    Calculate the marginalized Bayes factor for the presence of the conservative force in a set of bins.

    For usage and parameters see README.rst
    """

    # Constants
    dim = 2
    n_pi = n_pi_func(dim)
    # B_threshold = 10  # corresponds to strong evidence for the conservative force

    # Convert input to numpy
    zeta_ts = np.asarray(zeta_ts)
    zeta_sps = np.asarray(zeta_sps)
    Vs = np.asarray(Vs)
    Vs_pi = np.asarray(Vs_pi)
    ns = np.asarray(ns)

    # Check that the 2nd dimension has size 2
    if np.shape(zeta_ts)[1] != 2 or np.shape(zeta_sps)[1] != 2:
        raise VsalueError(
            "zeta_ts and zeta_sps must be matrices of size (M x 2)")

    # Initialize
    M = len(ns)
    etas = np.sqrt(n_pi / (ns + n_pi))
    us = np.divide(Vs_pi, Vs)
    v0s = 1.0 + np.multiply(n_pi / ns, us)
    # pows = dim * (ns + n_pi + 1.0) / 2.0 - 2.0
    pows = p(ns, dim)

    # Calculate
    lg_Bs = np.zeros((M, 1)) * np.nan
    min_ns = np.zeros((M, 1), dtype=int) * np.nan
    for i in trange(M):
        try:
            upstairs = calculate_marginalized_integral(zeta_t=zeta_ts[i, :], zeta_sp=zeta_sps[i, :], p=pows[i],
                                                       v=v0s[i], E=etas[i]**2.0)
            # print(upstairs)
            downstairs = calculate_marginalized_integral(zeta_t=zeta_ts[i, :], zeta_sp=zeta_sps[i, :], p=pows[i],
                                                         v=v0s[i], E=1.0)
            # print(downstairs)
            lg_Bs[i] = dim * np.log10(etas[i]) + \
                np.log10(upstairs) - np.log10(downstairs)
            # print(lg_Bs)
            min_ns[i] = calculate_minimal_n(zeta_ts[i, :], zeta_sp=zeta_sps[i, :], n0=ns[i], V=Vs[i],
                                            V_pi=Vs_pi[i])
            # print(min_ns)
        except:
            print("Warning: Detected data error in bin %i. Skipping bin." % i)

    # Threshold into 3 categories: strong evidence for either of the models and insufficient evidence
    forces = 1 * (lg_Bs >= np.log10(B_threshold)) - \
        1 * (lg_Bs <= -np.log10(B_threshold))
    # forces = np.zeros_like(lg_Bs, dtype = int)
    # forces [lg_Bs	>= np.log10(B_threshold)] = 1
    # forces [lg_Bs	<= - np.log10(B_threshold)] = -1

    return (10.0 ** lg_Bs, forces, min_ns)
