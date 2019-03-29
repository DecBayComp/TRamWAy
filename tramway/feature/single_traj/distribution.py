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
This module regroups functions generating random distributions.
"""

import numpy as np


def distribution_exp(nb_point=100, lambda_=0.1):
    s = np.random.uniform(0, 1, nb_point)
    return -1 / lambda_ * np.log(1 - s)


def distrib_random_angle(nb_point=1000, dim=1):
    if dim == 1:
        return np.random.randint(2, size=nb_point) * 2 - 1
    elif dim == 2:
        return np.random.uniform(0, 2*np.pi, nb_point)
    elif dim == 3:
        theta = np.random.uniform(0, np.pi, nb_point)
        phi = np.random.uniform(0, 2*np.pi, nb_point)
        return theta, phi
    else:
        raise ValueError(f'dim = {dim} not supported')


def distrib_long_tail_LOMAX_time(alpha=1., lambda_=2., nb_point=100):
    y = np.random.uniform(0, 1, nb_point)
    tau = lambda_*(-1 + np.power(1-y, -1/alpha))
    return tau


def distribution_long_tail_LOMAX_timecut(alpha=1., lambda_=2., tau_max=100,
                                         nb_point=100):

    loc = 1 + tau_max/lambda_
    beta = 1 / (1 - np.power(loc, -alpha))
    y = np.random.uniform(0, 1, nb_point)
    tau = lambda_*(-1 + np.power(beta - y, -1/alpha))
    tau[tau <= 0] = 0
    return tau


def distribution_t_student(df=1, nb_point=1000):
    return np.random.standard_t(df, size=nb_point)


def distribution_uniform_low_max(low_value=0, max_value=1, nb_point=1000):
    s = np.random.uniform(low_value, max_value, 1000)
    return s


def distribution_uniform_low_max_sobol(low_value=0, max_value=1,
                                       nb_point=1000, dim=2):

    s = sobol_seq.i4_sobol_generate(dim, nb_point)
    return low_value + s * (max_value - low_value)


def distrib_long_tail_cst_before_cut(tau_scale=1, alpha=0.25, nb_point=100):
    """
    distribution normalisee (qaund elle est definie cest a dire alpha>1) de
    points avec valeur constante jusqu'à un cut (tau_scale) et ensuite une
    distribution qui chute en t^{-alpha}.
    Distrbution qui permet de ne pas trop définir.
    """
    y = np.random.uniform(0, 1, nb_point)
    y[y <= alpha/(1.+alpha)] = 0.
    y[y > alpha/(1.+alpha)] = 1.
    V_max = 1./(1+alpha)
    # first part
    tau_1 = np.random.uniform(0, tau_scale, nb_point)
    # second part
    y_loc = np.random.uniform(0, V_max, nb_point)
    tau_2 = tau_scale*np.power(1 - (1+alpha)*(V_max - y_loc), -1/alpha)

    tau = (1-y)*tau_1 + y*tau_2
    return tau, tau_1, tau_2


def distrib_long_tail_with_cut_cst_other_cut(tau_scale=0.5, tau_cut=10,
                                             alpha=0.25, nb_point=100):
    """
    distribution normalisee (et definie partout) avec du cut value
    la premiere pour la partie constante de la distribution
    la seconde pour le cut de la partie en  t^{-alpha}
    distrbution qui permet de faire des cas realistes avec des lois d'échelles
    mais un cut evitant les emmerdes et les non normalizations
    tau_scale < tau_cut
    """
    constante1 = tau_scale - np.power(tau_scale, 1. + alpha)/alpha * (
        np.power(tau_cut, -alpha*1.) - np.power(tau_scale, -alpha*1.))
    constante2 = 1/constante1
    constante3 = constante2 * tau_scale
    V_max = -constante2*tau_scale/alpha * \
        (np.power(tau_cut/tau_scale, -alpha*1.)-1)
    y = np.random.uniform(0, 1, nb_point)
    y[y <= constante3] = 0.
    y[y > constante3] = 1.

    # first part
    tau_1 = np.random.uniform(0, tau_scale, nb_point)

    # second part
    y_loc = np.random.uniform(0, V_max, nb_point)
    tau_2 = tau_scale * \
        np.power(1 - alpha/(constante2*tau_scale)*(V_max-y_loc), -1/alpha)

    tau = (1-y)*tau_1 + y*tau_2
    return tau, tau_1, tau_2


def distribution_alpha_levy_stable(alpha=1.5, beta=1, c=1, mu=0,
                                   nb_point=1000):
    """
    distribution alpha-levy stable
    alpha expoenent of the tail part
    beta [-1: 1] skewness parameter
    c [0.. infty[ scale parameter
    mu ]-infty .. infty[
    """

    U = np.random.uniform(-np.pi/2., np.pi/2., nb_point)
    W = distribution_exp(nb_point, 1)
    ksi = -beta * np.tan(np.pi*alpha/2.)

    if alpha != 1:
        eta = 1./alpha*np.arctan(-ksi)

        term1 = np.power(1+ksi*ksi, 1/(2. * alpha))
        term2 = np.sin(alpha*(U + eta))/np.power(np.cos(U), 1./alpha)
        term3 = np.power(np.cos(U - alpha * (U + eta))/W, (1. - alpha)/alpha)

        tau = term1 * term2 * term3
        tau = c*tau + mu

    else:
        eta = np.pi/2.
        term1 = 1./eta
        term2 = (np.pi/2. + beta * U)*np.tan(U)
        term3 = beta*np.log((np.pi/2*W*np.cos(U))/(np.pi/2 + beta*U))

        tau = term1*(term2 - term3)
        tau = c*tau + 2/np.pi*beta*c*np.log(c) + mu

    return tau


def exp_noise(d_max=0.1, nb_point=1000):
    s = np.random.uniform(0, 1, nb_point)
    return -1 / d_max * np.log(1 - s)


def exponential_correlated_noise(dt=0.025, lambda_=1, nb_point=100):
    t = np.arange(nb_point) * dt
    delta_t = np.abs(t[np.newaxis, :] - t[:, np.newaxis])
    covariance = np.exp(-lambda_ * delta_t)
    y = np.random.randn(nb_point)
    L = np.linalg.cholesky(covariance)
    return np.dot(L, y)


def fractional_correlated_noise(dt=0.025, H=0.55, nb_point=100, matrix=False):
    HH = 2 * H
    t = np.arange(nb_point) * dt
    d = np.abs(t[:, np.newaxis] - t[np.newaxis, :])
    covariance = (np.abs(d-dt)**HH + (d+dt)**HH - 2*d**HH) * 0.5
    y = np.random.randn(nb_point)
    L = np.linalg.cholesky(covariance)
    return np.dot(L, y)


def correlation_fractional(sigma=1, H=0.25, t=0, s=1):
    HH = 2 * H
    constante = sigma * sigma / 2
    term1 = np.power(np.absolute(t-s-1), HH)
    term2 = np.power(np.absolute(t-s), HH)
    term3 = np.power(np.absolute(t-s+1), HH)
    return constante * (term1 - 2 * term2 + term3)


def correlation_fractional_t_s(dt=0.025, sigma=1, H=0.25, t_s=0):
    # t_s is an integer
    HH = 2 * H
    constante = sigma * sigma / 2 * np.power(dt, HH)
    term1 = np.power(np.absolute(t_s-1), HH)
    term2 = np.power(np.absolute(t_s), HH)
    term3 = np.power(np.absolute(t_s+1), HH)
    return constante * (term1 - 2*term2 + term3)


def correlation_exponential(sigma=1, lambda_=1, t=0, s=1):
    return sigma_ * np.exp(-lambda_ * np.absolute(t-s))


def fractional_correlated_noise_wood_chan(dt=0.025, sigma=0.5,
                                          H=0.25, nb_point=100):
    nu = int(np.floor(np.log(nb_point) / np.log(2)) + 1)
    m = int(2 * np.power(2, nu))
    m_2 = int(np.floor(m/2))

    e = np.linspace(0, m_2*1, m_2, endpoint=False)
    g = np.linspace(0, (m_2+1)*1, m_2+1, endpoint=False)
    g = g[::-1]
    g = g[0:m_2]
    c2 = np.concatenate((e, g), axis=0)
    c = correlation_fractional_t_s(dt, sigma, H, c2)

    eigen = np.fft.fft(c)

    U = np.random.randn(m_2)
    V = np.random.randn(m_2)

    W = 1j*np.zeros((m,))
    W[0] = U[0]
    W[m_2] = V[0]

    W[1:m_2] = 1 / np.sqrt(2) * (U[1:m_2] + 1j * V[1:m_2])
    W2 = 1 / np.sqrt(2.)*(U[1:m_2] - 1j*V[1:m_2])
    W3 = W2[::-1]
    W[m_2+1:m] = W3

    W2 = 1 / np.sqrt(m) * np.sqrt(eigen) * W
    X = np.fft.fft(W2)
    return X.real


def noisify_gaussian_static(data_in, sigma=0.01):
    return data_in + np.random.randn(data_in.shape) * sigma


def generate_distribution(distribution, nb_point, **kwargs):
    """
    This function is used to generate distributions.
    """
    if distribution == 'exp':
        return distribution_exp(nb_point, 1 / kwargs['d_l'])
    elif distribution == 'uni':
        return np.random.uniform(0, 2 * kwargs['d_l'], nb_point)
    elif distribution == 'gauss':
        if kwargs['dim'] == 1:
            return np.random.normal(0, kwargs['d_l'], nb_point)
        else:
            cst = np.sqrt(np.pi / 2)
            # cst = 1
            return np.abs(np.random.normal(0, kwargs['d_l'] * cst, nb_point))
    elif distribution == 'lomax':
        alpha = kwargs['alpha']
        lambda_ = kwargs['d_l'] * (alpha - 1) if alpha > 1 else 1
        return distrib_long_tail_LOMAX_time(alpha, lambda_, nb_point)
    elif distribution == 'constant_and_tail':
        alpha = kwargs['alpha']
        d_scale = kwargs['d_l']
        tau_scale = d_scale * (1 + alpha) / (alpha * (1/2 + 1 / (alpha - 1)))
        return distrib_long_tail_cst_before_cut(tau_scale, alpha, nb_point)[0]
    elif distribution == "alpha_stable":
        args = (kwargs['alpha'], 0, kwargs['c_alpha_levy'], 0, nb_point)
        return np.abs(distribution_alpha_levy_stable(*args))
    elif distribution == 'lomax_cut':
        prms = (kwargs['alpha'], kwargs['d_l'], kwargs['d_max'], nb_point)
        return distribution_long_tail_LOMAX_timecut(*prms)
    elif distribution == 'constant_tail_cut':
        prms = (kwargs['alpha'], kwargs['d_l'], kwargs['d_max'], nb_point)
        return distrib_long_tail_with_cut_cst_other_cut(*prms)[0]
    elif distribution == 'cst':
        return np.ones(nb_point) * kwargs['d_l']


def generate_times(distribution, nb_point, T_max, dt, alpha=1.5,
                   d_tau=0.02, d_tau_max=1, dim=2, c_scale_alpha_stable_t=1):
    if distribution == "cst":
        t = np.linspace(0, nb_point*dt, nb_point, endpoint=False)
    else:
        t = generate_distribution(distribution, nb_point - 1, alpha=alpha,
                                  d_l=d_tau, d_max=d_tau_max, dim=dim,
                                  c_alpha_levy=c_scale_alpha_stable_t)
        while np.sum(t) < T_max:
            t_add = generate_distribution(distribution, nb_point - 1,
                                          alpha=alpha,
                                          d_l=d_tau, d_max=d_tau_max, dim=dim,
                                          c_alpha_levy=c_scale_alpha_stable_t)
            t = np.concatenate((t, t_add))
        t = np.insert(t.cumsum(), 0, 0)
    return t


# For diffusing diffusivity

def give_diffusion_chubynsky_slater(nb_point=1000, dt=0.025,
                                    sigma_bruit=0.1, g_bruit=0.1,
                                    D_init=0.05):
    # equaton dD/dt = -g + sigma*xi(t)
    # reflecting boundary conditions at D = 0
    # equilibirum diffusion D* = sigma^2/(2*g)
    # equilibirum distribution 1/D*exp(-D/D*)
    noise_ = sigma_bruit*np.sqrt(dt)*np.random.randn(nb_point, 1)
    D = np.zeros((nb_point,))
    D[0] = D_init
    drift = -g_bruit*dt
    for i in range(1, nb_point):
        D[i] = D[i-1] + drift + noise_[i]
        if D[i] < 0:
            D[i] *= -1
    return D


def give_diffusion_langevin_square(n_dimension=1, nb_point=1000, dt=0.025,
                                   tau_bruit=10, sigma_bruit=0.1, D_init=0.05):
    # 1D most likely in our case could be extended to ndimention : to our
    # application a bit pointless
    # Y is auxiliary variable
    noise_ = sigma_bruit*np.sqrt(dt)*np.random.randn(nb_point, n_dimension)
    Y = np.zeros((nb_point, n_dimension))
    Y[0, :] = 1/np.sqrt(n_dimension)*np.sqrt(D_init)*np.ones((1, n_dimension))
    const = (1 - dt/tau_bruit)
    for i in range(1, nb_point):
        Y[i, :] = const * Y[i-1, :] + noise_[i-1, :]
        Y[i, Y[i, :] < 0] = -Y[i,  Y[i, :] < 0]
    D = np.power(np.linalg.norm(Y, axis=1), 2)
    return D


def give_diffusion_CIR(nb_point=1000, dt=0.025, tau_bruit=10, sigma_bruit=0.1,
                       D_init=0.05):
    # modele de Cox–Ingersoll–Ross utilise par grebenkov dans
    # J. Phys. A: Math. Theor. 51 145602
    noise_ = sigma_bruit*np.sqrt(dt)*np.random.randn(nb_point, 1)
    D = np.zeros((nb_point,))
    D[0] = D_init
    const = (1. - dt/tau_bruit)
    drift = D_init/tau_bruit*dt
    for i in range(1, nb_point):
        D[i] = const*D[i-1] + drift + np.sqrt(2*D[i-1])*noise_[i-1]
        if D[i] < 0:
            D[i] *= -1
    return D
