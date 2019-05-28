# -*- coding:utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#    Contributor: Maxime Duval

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""
This module regroups random walk simulations of many different kinds.
"""

import numpy as np
import numpy.matlib as npm
import pandas as pd

from .distribution import *
from .rw_misc import *


# Directory of saved fractal patterns (to avoid generating them each time
# we want a random walk).
FRACTAL_DIR = ('Z:\\LAB_shared_stuff\\Maxime_Duval\\single_rw\\'
               'Fractal_pattern\\patterns')
TREE_DIR = ('Z:\\LAB_shared_stuff\\Maxime_Duval\\single_rw\\DLA\\patterns')


# Continous time random walks

def RW_brownian(T_max=1, dt=1e-2, D=0.1, nb_short=1, v=None, X_init=0, dim=2):
    """
    Generates simple pure diffusive random walk.

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    D : float, the diffusivity of the random walk.
    nb_short : int,  the number of points used between each point output, to
        avoid accumulating noise.
    X_init : float or list, initial position of the random walk.
    dim : int, in {1,2,3}, the dimension of the random walk.
    """
    nb_point = int(T_max/dt)
    nb_tot = (nb_point-1) * nb_short
    dt_short = dt/nb_short
    Xi = np.random.randn(nb_tot, dim)
    X_init = normalize_init(X_init, dim)
    drift = np.array(normalize_init(v, dim)) * dt * np.ones((nb_point-1, dim))
    X = np.cumsum(np.sqrt(2*D*dt_short)*Xi, axis=0) + np.array(X_init) + drift
    X = X[::nb_short]
    X = np.insert(X, 0, X_init, axis=0)
    t = np.arange(nb_point)*dt
    data = np.stack((t,) + tuple(X[:, i] for i in range(dim))).T
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:dim])


def RW_exp_dist(T_max=1, dt=1e-2, d_l=0.1, v=None, X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space="exp", d_l=d_l,
                distribution_time="cst", v=v, X_init=X_init, dim=dim)


def RW_const_dist(T_max=1, dt=1e-2, d_l=0.1, v=None, X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space="uni", d_l=d_l,
                distribution_time="cst", v=v, X_init=X_init, dim=dim)


def RW_gauss_dist(T_max=1, dt=1e-2, d_l=0.1, v=None, X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space="gauss", d_l=d_l,
                distribution_time="cst", v=v, X_init=X_init, dim=dim)


def RW_exp_dist_exp_temps(T_max=1, dt=1e-2, d_l=0.1,
                          d_tau=1e-2, v=None, X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space="exp", d_l=d_l,
                distribution_time="exp", d_tau=d_tau, v=v, X_init=X_init,
                dim=dim)


def RW_anomalous_dist(T_max=1, dt=1e-2, alpha=1.5, d_scale=0.1,
                      c_scale_alpha_stable=1, nature_distribution="lomax",
                      v=None, X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space=nature_distribution,
                c_scale_alpha_stable_x=c_scale_alpha_stable,
                d_l=d_scale, alpha_space=alpha,
                distribution_time="cst", v=v, X_init=X_init, dim=dim)


def RW_anomalous_with_cut_dists(T_max=1, dt=1e-2, alpha=1.5, d_scale=0.1,
                                d_jump_max=1, v=None,
                                nature_distribution="constant_tail_cut",
                                X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space=nature_distribution,
                d_l=d_scale, alpha_space=alpha, d_jump_max=d_jump_max,
                distribution_time="cst", v=v, X_init=X_init, dim=dim)


def RW_exp_dist_anomalous_time(T_max=1, dt=1e-2, d_l=0.1, alpha=1.5, d_tau=0.1,
                               c_scale_alpha_stable=1, v=None,
                               nature_distribution="lomax", X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space="exp", d_l=d_l,
                distribution_time=nature_distribution, alpha_time=alpha,
                d_tau=d_tau, v=v, X_init=X_init, dim=dim,
                c_scale_alpha_stable_t=c_scale_alpha_stable)


def RW_exp_dist_anomalous_with_cut_time(T_max=1, dt=1e-2, d_l=0.1, alpha=1.5,
                                        d_tau=0.1, d_tau_max=1,
                                        nature_distribution="lomax_cut",
                                        v=None, X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space="exp", d_l=d_l,
                distribution_time=nature_distribution, alpha_time=alpha,
                d_tau=d_tau, d_tau_max=d_tau_max, v=v, X_init=X_init, dim=dim)


def RW_gauss_dist_anomalous_with_cut_time(T_max=1, dt=1e-2, d_l=0.1, alpha=1.5,
                                          d_tau=0.1, d_tau_max=1,
                                          nature_distribution="lomax_cut",
                                          v=None, X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space="gauss", d_l=d_l,
                distribution_time=nature_distribution, alpha_time=alpha,
                d_tau=d_tau, d_tau_max=d_tau_max, v=v, X_init=X_init, dim=dim)


def RW_const_dist_anomalous_time(T_max=1, dt=1e-2, d_l=0.1, alpha=1.5,
                                 d_tau=0.1, d_tau_max=1,
                                 c_scale_alpha_stable=1,
                                 nature_distribution="lomax",
                                 v=None, X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space="uni", d_l=d_l,
                distribution_time=nature_distribution, alpha_time=alpha,
                d_tau=d_tau, d_tau_max=d_tau_max, v=v, X_init=X_init, dim=dim,
                c_scale_alpha_stable_t=c_scale_alpha_stable)


def RW_const_dist_anomalous_with_cut_time(T_max=1, dt=1e-2, d_l=0.1, alpha=1.5,
                                          d_tau=0.1, d_tau_max=1,
                                          nature_distribution="lomax_cut",
                                          v=None, X_init=0, dim=2):
    return CTRW(T_max=T_max, dt=dt, distribution_space="uni", d_l=d_l,
                distribution_time=nature_distribution, alpha_time=alpha,
                d_tau=d_tau, d_tau_max=d_tau_max, v=v, X_init=X_init, dim=dim)


def CTRW(T_max=1, dt=1e-2,
         distribution_space="exp",  alpha_space=1.5, d_l=0.1, d_jump_max=1,
         c_scale_alpha_stable_x=1,
         distribution_time="cst", alpha_time=1.5, d_tau=0.02, d_tau_max=1,
         c_scale_alpha_stable_t=1,
         v=None, X_init=0, dim=2):
    """
    Generates diffusive RW from chosen distribution of distances and waiting
    times.

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    distribution_space : string, the type of the distance distribution.
    alpha_space : float, parameter used for long tail distributions.
    d_l : float, the characteristic distance of the random walk.
    d_jump_max : float, the maximum distance of a jump for long tail
        distributions with a cut.
    c_scale_alpha_stable_x : scale parameter in case distances follow an alpha
        levy stable distribution.
    distribution_time : string, the type of the waiting time distribution.
    alpha_time : float, parameter used for long tail distributions.
    d_tau : float, the characteristic waiting time of the random walk.
    d_tau_max : float, the maximum distance of a jump for long tail
        distributions with a cut.
    c_scale_alpha_stable_x : scale parameter in case waiting times follow an
        alpha levy stable distribution.
    v : drift, list of size dim.
    X_init : float or list, initial position of the random walk.
    dim : int, in {1,2,3}, the dimension of the random walk.
    """
    nb_point = int(T_max/dt)
    if distribution_time != "cst":
        nb_point *= 10
        T_max *= 10
    # Generates a vector t such that t[-1] > T ==> len(t) >= nb_point
    t = generate_times(distribution_time, nb_point, T_max, dt,
                       alpha=alpha_time,
                       d_tau=d_tau, d_tau_max=d_tau_max, dim=dim,
                       c_scale_alpha_stable_t=c_scale_alpha_stable_t)
    raw_length = len(t)
    distances = generate_distribution(distribution_space,
                                      raw_length - 1, alpha=alpha_space,
                                      d_l=d_l, d_max=d_jump_max, dim=dim,
                                      c_alpha_levy=c_scale_alpha_stable_x)
    random_angles = distrib_random_angle(raw_length-1, dim=dim)
    X = apply_angle_dists(distances, random_angles, dim)
    normed_pos_init = np.array(normalize_init(X_init, dim))
    drift = np.array(normalize_init(v, dim)) * dt * np.ones(X.shape)
    X = X.cumsum(axis=0) + normed_pos_init + drift
    X = np.concatenate((np.expand_dims(normed_pos_init, axis=0), X), axis=0)
    if distribution_time != "cst":
        t_regularized = np.linspace(0, T_max, nb_point, endpoint=False)
        X, t = regularize_times(X, t, t_regularized)
    data = np.concatenate((np.expand_dims(t, axis=1), X), axis=1)
    if distribution_time != "cst":
        random_start = np.random.randint(0, int(nb_point * 0.9))
        data = data[random_start:random_start+int(nb_point * 0.1)]
        data[:, 0] = np.linspace(0, T_max*0.1, int(nb_point*0.1),
                                 endpoint=False)
        data[:, [1, 2]] -= data[0, [1, 2]]
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:dim])

# Fractional random walks.


def RW_FBM(T_max=1, dt=1e-2, corr_type="frac", sigma=0.5, H=0.55,
           v=None, X_init=0, dim=2):
    """
    Generates fractal diffusive RW.

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    corr_type : string, type of correlation. Should be in {"frac", "exp",
        "wood_chan"}.
    sigma : float, a scaling parameter.
    H : fractionnarity.
        ! If corr_type = "exp", H controls the fractionarity with a
        different range of values : H need to get into the thousands to
        have the same fractionarity as H ~ 0.3 with corr_type = "frac".
    v : drift, list of size dim.
    X_init : float or list, initial position of the random walk.
    dim : int, in {1,2,3}, the dimension of the random walk.
    """
    nb_point = int(T_max/dt)
    if corr_type == "frac":
        W = [fractional_correlated_noise(dt, H, nb_point-1) * sigma
             for _ in range(dim)]
    elif corr_type == "exp":
        W = [exponential_correlated_noise(dt, H, nb_point-1) * sigma
             for _ in range(dim)]
    elif corr_type == "wood_chan":
        W = [fractional_correlated_noise_wood_chan(dt, sigma, H, nb_point-1)
             for _ in range(dim)]
    else:
        raise ValueError(f'{corr_type} : unrecognized correlation type.')
    W = np.atleast_2d(W)[:, :nb_point-1]
    normed_pos_init = np.array(normalize_init(X_init, dim))
    drift = np.array(normalize_init(v, dim)) * dt * np.ones((nb_point-1, dim))
    W = W.cumsum(axis=1).T + normed_pos_init + drift
    W = np.concatenate((np.expand_dims(normed_pos_init, axis=0), W), axis=0)
    t = np.linspace(0, nb_point*dt, nb_point, endpoint=False)
    data = np.concatenate((np.expand_dims(t, axis=1), W), axis=1)
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:dim])


# Confined Random Walk

def Ornstein_Uhlenbeck_integral(T_max=1, dt=1e-2, D=0.1, d_confinement=0.2,
                                nb_short=1, X_init=0, X_target=0, dim=2,
                                V_confinement=4):
    """
    Generate confined random walk following ornstein uhlenbeck process.
    follow equilibrium rule (D = kbT/gamma)
    center of the trap will be X_target to be modified if necessary
    avoid starting r_ini too far from x_init
    integral version.

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    D : float, the diffusivity of the random walk.
    d_confinement: typical scale of confinement corresponding to 4kbT.
    nb_short : int,  the number of points used between each point output, to
        avoid accumulating noise.
    X_init : float or list, initial position of the random walk.
    X_target : center of the confined walk.
    dim : int, in {1,2,3}, the dimension of the random walk.
    """
    # parameters definition
    gamma = 1 / D
    k_eff = 2 * V_confinement / (np.power(d_confinement, 2) * gamma)
    sigma = np.sqrt(2*D)
    nb_point = int(T_max/dt)
    nb_tot = (nb_point) * nb_short
    dt_short = dt/nb_short
    X_init = normalize_init(X_init, dim)

    t = np.arange(nb_tot)*dt_short
    X_target = np.array(normalize_init(X_target, dim))
    exp_k_eff = np.exp(-k_eff*t)
    exp_k_eff_2 = np.tile(exp_k_eff, (dim, 1)).T
    one_exp_k_eff_2 = np.ones((nb_tot, dim)) - exp_k_eff_2

    exp_2k_eff_minus_1 = np.exp(2 * k_eff * t) - 1.
    exp_2k_eff_minus_1_2 = np.tile(exp_2k_eff_minus_1, (dim, 1)).T
    exp_2k_eff_minus_1_2_diff = np.sqrt(np.diff(exp_2k_eff_minus_1_2, axis=0))

    term_gaussian = np.random.randn(nb_tot-1, dim)/np.sqrt(2*k_eff)
    joined_term = exp_2k_eff_minus_1_2_diff * term_gaussian
    joined_term_0 = np.insert(joined_term, 0, 0, axis=0)

    term1 = np.array(X_init)*exp_k_eff_2
    term2 = X_target*one_exp_k_eff_2
    term3 = sigma * exp_k_eff_2 * np.cumsum(joined_term_0, axis=0)
    X = term1 + term2 + term3
    data = np.stack((t,) + tuple(X[:, i] for i in range(dim))).T
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:dim])


def Ornstein_Uhlenbeck_update(T_max=1, dt=1e-2, D=0.1, d_confinement=0.2,
                              nb_short=1, X_init=0, X_target=0, dim=2,
                              V_confinement=4):
    """
    Generate confined random walk following ornstein uhlenbeck process.
    follow equilibirum rule (D = kbT/gamma)
    center of the trap will be X_target to be modified if necessary
    avoid starting r_ini too far from x_init
    classical update version

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    D : float, the diffusivity of the random walk.
    d_confinement: typical scale of confinement corresponding to 4kbT
    nb_short : int,  the number of points used between each point output, to
        avoid accumulating noise.
    X_init : float or list, initial position of the random walk.
    X_target : center of the confined walk.
    dim : int, in {1,2,3}, the dimension of the random walk.
    """
    # parameters definition
    gamma = 1 / D
    k_eff = 2 * V_confinement / (np.power(d_confinement, 2) * gamma)
    sigma = np.sqrt(2 * D)
    nb_point = int(T_max/dt)
    nb_tot = (nb_point - 1) * nb_short
    dt_short = dt / nb_short
    k_eff_dt_short = k_eff * dt_short
    X_init = normalize_init(X_init, dim)
    t = np.arange(nb_tot)*dt_short
    X_target = np.array(normalize_init(X_target, dim))
    X_noise = np.sqrt(2*D*dt_short)*np.random.randn(nb_tot, dim)
    X = np.empty((nb_tot, dim))
    X[0] = X_init
    for i in range(1, nb_tot):
        X[i] = X[i-1] - k_eff_dt_short * (X[i-1] - X_target) + X_noise[i-1]
    data = np.stack((t,) + tuple(X[:, i] for i in range(dim))).T
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:dim])


# Hidden Markov Model of diffusion, brownian movement.

def RW_HMM(T_max=1, dt=1e-2, D=np.array([0.001, 0.1]),
           T=np.array([[0.9, 0.1], [0.1, 0.9]]),
           p_init=np.array([0.5, 0.5]), v=None,
           nb_short=1, X_init=0, dim=2):
    """
    Generate diffusive random walk with diffusion following a markov model.

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    D : float, the k possible diffusivities of the random walk.
    T : transition matrix from one diffusive state to the other.
        Must have shape k x k and each row must sum to 1.
    p_init : initial state probability distribution, shape : k.
    v : speed, drift parameter. If not None, should be an array of size
        k x dim.
    nb_short : int,  the number of points used between each point output, to
        avoid accumulating noise.
    X_init : float or list, initial position of the random walk.
    dim : int, in {1,2,3}, the dimension of the random walk.
    """
    nb_point = int(T_max/dt)
    nb_tot = (nb_point - 1) * nb_short
    dt_short = dt/nb_short
    X = np.zeros((nb_point, dim)) + np.array(normalize_init(X_init, dim))
    states = np.zeros((nb_point, 1))
    Xi = X[0].copy()
    nstate = len(p_init)
    state = np.random.choice(nstate, size=1, p=p_init).item()
    states[0] = state
    if v is None:
        v = np.zeros((nstate, dim))
    for i in range(1, nb_tot+1):
        Xi += (np.sqrt(2*D[state]*dt_short)*np.random.randn(dim) +
               v[state]*dt_short)
        if i % nb_short == 0:
            X[i//nb_short] = Xi
            state = np.random.choice(nstate, size=1, p=T[state]).item()
            states[i//nb_short] = state
    t = np.linspace(0, nb_point*dt, nb_point, endpoint=False)
    data = np.concatenate((np.expand_dims(t, axis=1), X, states), axis=1)
    return pd.DataFrame(data=data, columns=(['t'] + SPACE_COLS[:dim] +
                                            ['state']))


# Confined movement.

def RW_circular_confinement(T_max=1, dt=1e-2, D=0.1, d_wall=0.2, nb_short=1,
                            X_init=0, dim=2, sigma_boundary=0.02,
                            V_confinement=10):
    # parameters definition
    # sigma_boundary : important parameter for regularization
    gamma_D = 1 / D
    nb_point = int(T_max / dt)
    nb_tot = (nb_point - 1) * nb_short
    dt_short = dt/nb_short
    X_init = normalize_init(X_init, dim)

    t = np.arange(nb_tot)*dt_short
    X_noise = np.sqrt(2*D*dt_short)*np.random.randn(nb_tot, dim)
    X = np.empty((nb_tot, dim))
    X[0] = X_init

    if dim == 1:
        for i in range(1, nb_tot):
            f_local = local_border_forces_1D(
                X[i-1], d_wall, sigma_boundary, V_confinement) * 1/gamma_D
            X[i] = X[i-1] + f_local*dt_short + X_noise[i-1]

    if dim == 2:
        for i in range(1, nb_tot):
            r, theta = cart2pol(X[i-1, 0], X[i-1, 1])
            f_local = local_border_forces_2D(
                r, d_wall, theta, sigma_boundary, V_confinement) * 1/gamma_D
            X[i] = X[i-1] + f_local*dt_short + X_noise[i-1]

    if dim == 3:
        for i in range(1, nb_tot):
            r, phi, theta = cart2spher(X[i-1, 0], X[i-1, 1],  X[i-1, 2])
            f_local = (local_border_forces_3D(
                r, d_wall, theta, phi, sigma_boundary, V_confinement) *
                1/gamma_D)
            X[i] = X[i-1] + f_local*dt_short + X_noise[i-1]

    X = X[::nb_short]
    t = t[::nb_short]
    data = np.stack((t,) + tuple(X[:, i] for i in range(dim))).T
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:dim])


# Diffusing diffusivity

def RW_DD(T_max=1, dt=1e-2, nb_short=10, X_init=0, dim=2,
          mode_diffusion="chubynsky_slater", sigma_bruit=0.1, g_bruit=0.1,
          D_init=0.05, tau_bruit=5, n_dimension=2):
    """
    Generate diffusive random walk with diffusion itself following a random
    process.

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    nb_short : int,  the number of points used between each point output, to
        avoid accumulating noise.
    X_init : float or list, initial position of the random walk.
    dim : int, in {1,2,3}, the dimension of the random walk.
    mode_diffusion : string, the type of random process that diffusion follows
        Must be in {"chubynsky_slater", "langevin_square", "CIR"}.
    sigma_bruit : parameter equivalent to the diffusion in the random process
        of the diffusion.
    g_bruit : drift parameter in the random process of the diffusion.
    D_init : initial value taken by the diffusion of the random walk.
    tau_bruit : controls how much diffusion is above 0.
    n_dimension : dimension of the diffusion returned in case we want to
        generate a multi dimensional Diffusion process (with the langevin
        square distribution).
    """
    nb_point = int(T_max/dt)
    nb_tot = (nb_point-1) * nb_short
    dt_short = dt/nb_short
    Xi = np.random.randn(nb_tot, dim)
    X_init = normalize_init(X_init, dim)
    if mode_diffusion == "chubynsky_slater":
        D = give_diffusion_chubynsky_slater(nb_tot, dt_short, sigma_bruit,
                                            g_bruit, D_init)
    if mode_diffusion == "langevin_square":
        D = give_diffusion_langevin_square(n_dimension, nb_tot, dt_short,
                                           tau_bruit, sigma_bruit, D_init)
    if mode_diffusion == "CIR":
        D = give_diffusion_CIR(nb_tot, dt_short, tau_bruit, sigma_bruit,
                               D_init)
    D = npm.repmat(D, dim, 1).T
    X = np.cumsum(np.sqrt(2*D*dt_short)*Xi, axis=0) + np.array(X_init)
    X = X[::nb_short]
    X = np.insert(X, 0, X_init, axis=0)
    t = np.arange(nb_point)*dt
    data = np.stack((t,) + tuple(X[:, i] for i in range(dim))).T
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:dim])


# SELF AVOIDING WALK
# code form: Computational investigations of folded self-avoiding
# walks related to protein folding
# Jacques Bahi, Christophe Guyeux, Kamel Mazouzi, Laurent Philippe
# Electronic Notes in Discrete Mathematics
# Volume 59, June 2017, Pages 37-50
# (slight modification)


# MISC functions with a global prm

def construction_sommets_graphe(pt_base, pt, d):
    global Graphe
    if d == 0:
        pt.reverse()
        pt = tuple(pt)
        if pt not in Dico_Des_Points:
            Graphe[pt] = []
    else:
        for dep in [-1, 0, 1]:
            construction_sommets_graphe(pt_base, pt + [pt_base[d - 1] + dep],
                                        d - 1)


def construction_aretes_graphe(dim):
    global Graphe
    for v1 in Graphe:
        for v2 in Graphe:
            if norme_manhattan(moinss(v1, v2, dim)) == 1:
                Graphe[v1].append(v2)
                Graphe[v2].append(v1)
    return Graphe


def calcul_nombres_infinis(dim, Deplacements):
    global Graphe
    pt_courant = Liste_Des_Points[-1]
    num_voisin = 0
    Graphe = {}
    resul = []
    while num_voisin < 2 * dim:
        candidat = pluss(pt_courant, Deplacements[num_voisin], dim)
        som = -1
        if candidat not in Dico_Des_Points:
            som = 0
            for ind in range(dim):
                petit_pt = enleve_coord(candidat, ind)
                if petit_pt not in Les_Minmax[ind]:
                    som += 2
                elif (candidat[ind] < Les_Minmax[ind][petit_pt][0] or
                      (candidat[ind] > Les_Minmax[ind][petit_pt][1])):
                    som += 1
            if som == 0 and candidat not in Dico_Ariane:
                if Graphe == {}:
                    construction_sommets_graphe(pt_courant, [], dim)
                    Graphe = construction_aretes_graphe(dim)
                    dest = intersection(Graphe, Dico_Ariane)
                chem = chemin(candidat, dest, Graphe)
                if chem is None:
                    som = -1
        resul.append(som)
        num_voisin += 1
    return resul


# Self Avoiding Random Walk useful functions


def nouveau_fil(pt_base):
    global Liste_Ariane
    global Dico_Ariane
    Liste_Ariane = [pt_base]
    Dico_Ariane = {pt_base: 0}


def racourcir_fil(index):
    while len(Liste_Ariane) > index:
        del Dico_Ariane[Liste_Ariane.pop()]


def ajout_dans_fil(pt):
    Dico_Ariane[tuple(pt)] = len(Liste_Ariane)
    Liste_Ariane.append(tuple(pt))


def insert_point(pt, dim):
    Liste_Des_Points.append(pt)
    Dico_Des_Points[pt] = None
    for index in range(dim):
        petit_pt = enleve_coord(pt, index)
        if petit_pt not in Les_Minmax[index]:
            Les_Minmax[index][petit_pt] = [pt[index], pt[index]]
        elif pt[index] < Les_Minmax[index][petit_pt][0]:
            Les_Minmax[index][petit_pt][0] = pt[index]
        elif pt[index] > Les_Minmax[index][petit_pt][1]:
            Les_Minmax[index][petit_pt][1] = pt[index]


def ajout_point(bias, dim, Deplacements):
    global Graphe
    pt_courant = Liste_Des_Points[-1]
    les_probas = calcul_proba(calcul_nombres_infinis(dim, Deplacements),
                              bias, dim)
    num_voisin = np.random.choice(les_probas)
    candidat = pluss(pt_courant, Deplacements[num_voisin], dim)
    for index in range(dim):
        petit_pt = enleve_coord(candidat, index)
        if ((petit_pt not in Les_Minmax[index]) or
                (candidat[index] > Les_Minmax[index][petit_pt][1])):
            insert_point(candidat, dim)
            nouveau_fil(voisin(candidat, index, 1))
            break
        elif candidat[index] < Les_Minmax[index][petit_pt][0]:
            insert_point(candidat, dim)
            nouveau_fil(voisin(candidat, index, -1))
            break
    else:
        if candidat in Dico_Ariane:
            insert_point(candidat, dim)
            racourcir_fil(Dico_Ariane[candidat])
        else:
            dest = intersection(Graphe, Dico_Ariane)
            chem = chemin(candidat, dest, Graphe)
            insert_point(candidat, dim)
            racourcir_fil(Dico_Ariane[chem[0]] + 1)
            for pt in chem[1: -1]:
                ajout_dans_fil(pt)


def RW_SAW(T_max=1, dt=1e-2, dr=1e-2, dim=2, bias=-1):
    """Generates a self avoiding random walk. This random walk nevers crosses a
    path where it has been.

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    dr : float, scale of the random walk.
    dim : int, in {1,2,3}, the dimension of the random walk.
    bias : scalar, somewhat corresponds to drift.
    """
    nb_point = int(T_max/dt)
    global Liste_Des_Points
    global Dico_Des_Points
    global Liste_Ariane
    global Dico_Ariane
    global Les_Minmax

    sigma = dr/4

    Origine = tuple([0] * dim)     # (0,..., 0)
    Liste_Des_Points = [Origine]
    Dico_Des_Points = {Origine: None}
    Petite_Origine = tuple([0] * (dim - 1))

    Les_Minmax = create_minmax(dim, Petite_Origine)

    Point_Un = list(Origine)
    Point_Un[0] = 1
    Point_Un = tuple(Point_Un)     # (1, 0,..., 0)

    Liste_Ariane = [Point_Un]
    Dico_Ariane = {Point_Un: 0}

    Deplacements = create_deplacement(dim)
    Graphe = {}

    Liste_Des_Points = [Origine]
    Dico_Des_Points = {Origine: None}
    Les_Minmax = []
    for index in range(dim):
        Les_Minmax.append({Petite_Origine: [0, 0]})
    Liste_Ariane = [Point_Un]
    Dico_Ariane = {Point_Un: 0}
    for nb in range(nb_point):
        ajout_point(bias, dim, Deplacements)

    noise_ = sigma*np.random.randn(nb_point, dim)
    X = np.asarray(Liste_Des_Points)[:-1] * dr + noise_
    t = np.arange(nb_point)*dt
    data = np.stack((t,) + tuple(X[:, i] for i in range(dim))).T
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:dim])


def load_motif_into_memory(fractal_name):
    global MOTIF
    try:
        MOTIF = np.load(f'{FRACTAL_DIR}\{fractal_name}.npy')
    except:
        print('Could not load fractal, try to create fractal with pattern {}'
              .format(fractal_name))


def RW_on_fractal_pattern(T_max=1, dt=1e-2, dr=1e-2, X_init=0, v=None,
                          pattern=None, noise_frac=0.2, **kwargs):
    """Generates a random walk which moves on a fractal pattern. We add noise
    to make it look more realistic

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    dr : float, scale of the random walk.
    X_init : float or list, initial position of the random walk.
    v : drift, list of size dim.
    noise_frac : float > 0. The fraction of the movement which is due to noise.
        In practise, we add a gaussian noise with standard deviation equal to
        dr * noise_frac.

    Note : dim is fixed at 2 at the moment, but could be generalized to higher
        dimensions by changing patterns and the function update_position.
    """
    nb_point = int(T_max/dt)
    X = np.zeros((nb_point, 2), dtype=int)
    X[0] = int(np.floor(pattern.shape[0]/2))
    for i in range(1, nb_point):
        i_init = X[i-1, 0]
        j_init = X[i-1, 1]
        k, l, keep_going = update_position(pattern, i_init, j_init)
        if keep_going:
            X[i, 0] = k
            X[i, 1] = l
        else:
            break
    noise_ = noise_frac * dr * np.random.randn(nb_point, 2)
    X = (X - X[0, :] + normalize_init(X_init, 2)) * dr + noise_
    drift = np.array(normalize_init(v, 2)) * dt * np.ones((nb_point-1, 2))
    X[1:] += drift.cumsum(axis=0)
    t = np.arange(nb_point)*dt
    data = np.stack((t,) + tuple(X[:, i] for i in range(2))).T
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:2])


# Diffusion-limited aggregation


def load_tree_into_memory(tree_name, DIR=TREE_DIR):
    global TREE
    try:
        TREE = np.load(f'{DIR}\\{tree_name}.npy')
    except:
        print('Could not load fractal, try to create fractal with pattern {}'
              .format(tree_name))


def RW_on_tree(T_max=1, dt=1e-2, dr=1e-2, X_init=0, v=None, tree=None,
               noise_frac=0.2, **kwargs):
    """Generates a random walk which moves on a tree. See Diffusion-limited
    aggregation.

    Parameters
    ----------
    T_max : float, the total time computed for which the random walk is
        computed.
    dt : float, time between each point output.
    dr : float, scale of the random walk.
    X_init : float or list, initial position of the random walk.
    v : drift, list of size dim.
    noise_frac : float > 0. The fraction of the movement which is due to noise.
        In practise, we add a gaussian noise with standard deviation equal to
        dr * noise_frac.

    Note : dim is fixed at 2 at the moment, but could be generalized to higher
        dimensions by changing patterns and the function update_position.
    """
    assert tree is not None, 'No tree given'
    nb_point = int(T_max/dt)
    X = np.zeros((nb_point, 2), dtype=int)
    X[0] = int(np.floor(tree.shape[0]/2))
    for i in range(1, nb_point):
        i_init = X[i-1, 0]
        j_init = X[i-1, 1]
        k, l, keep_going = update_position(tree, i_init, j_init)
        if keep_going:
            X[i, 0] = k
            X[i, 1] = l
        else:
            break
    noise_ = noise_frac * dr * np.random.randn(nb_point, 2)
    X = (X - X[0, :] + normalize_init(X_init, 2)) * dr + noise_
    drift = np.array(normalize_init(v, 2)) * dt * np.ones((nb_point-1, 2))
    X[1:] += drift.cumsum(axis=0)
    t = np.arange(nb_point)*dt
    data = np.stack((t,) + tuple(X[:, i] for i in range(2))).T
    return pd.DataFrame(data=data, columns=['t'] + SPACE_COLS[:2])
