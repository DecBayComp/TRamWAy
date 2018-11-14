# -*- coding: utf-8 -*-

# Copyright © 2018, Institut Pasteur
#   Contributor: Jean-Baptiste Masson

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
import random
#import pylab
#import math
import collections


############################################################
############################################################


class generate_trajectories:


    def __init__(self, lambda_ = 1, sigma_ = 0.4, sigma_noise_    = 0.010,  dt_  = 25e-3,
       n_short_ = 1000, N_mean_ = 7, n_trajs_  = 100, L_ = 0.5, density_  = 2. , D0_   = 0.1,
       amplitude_D_ = 0.025, amplitude_V_ = 0,
       mode_D="linear", mode_V="potential_force", mode_gamma="equilibrium", name_file_out="trajectories.txt", verbose=True):

       self.lambda_         = lambda_
       self.sigma_          = sigma_
       self.sigma_noise_    = sigma_noise_
       self.dt_             = dt_
       self.dt_space_       = 10 * self.dt_
       self.n_short_        = n_short_
       self.N_mean_         = N_mean_
       self.n_trajs_        = n_trajs_
       self.L_              = L_
       self.density_        = density_
       self.D0_             = D0_
       self.gamma_          = 1 / D0_
       self.n_total_size_max_   = n_trajs_ * 3 * N_mean_
       self.Nb_tot_loc_     = density_ * L_ * L_
       self.N_duration_tot_ = np.round(n_trajs_/self.Nb_tot_loc_)
       self.dt_short_       = dt_ / n_short_

       self.amplitude_D_    = amplitude_D_
       self.amplitude_V_    = amplitude_V_


       self.mode_D          = mode_D
       self.mode_V          = mode_V
       self.mode_gamma      = mode_gamma

       self.name_file_out   = name_file_out

       self.verbose = verbose

############################################################
############################################################
    def  random_start(self):
         L_ = self.L_
         x = -L_ + np.random.random()*2*L_
         y = -L_ + np.random.random()*2*L_
         return x , y
############################################################
############################################################
    def traj_duration(self):
        N_mean_ = self.N_mean_
        N_duration_ = -N_mean_*np.log(np.random.random())
        return N_duration_
############################################################
############################################################
    def noisify(self):
        sigma_noise_  = self.sigma_noise_
        x_            = self.x_
        y_            = self.y_

        nn_            = x_.size

        sigma1_       = np.random.randn(nn_)
        sigma2_       = np.random.randn(nn_)

        x_            = x_ + sigma_noise_*sigma1_
        y_            = y_ + sigma_noise_*sigma2_

        self.x_      = x_
        self.y_      = y_
############################################################
############################################################
    def diffusion_gaussian_trap(self, x, y ):

        r2_           = x*x + y*y;
        D0_           = self.D0_
        sigma_        = self.sigma_
        amplitude_D_  = self.amplitude_D_
        sigma_2_      = sigma_ * sigma_
        D_            = D0_ - amplitude_D_*np.exp(-r2_/(2*sigma_2_))
        grad_D_x_     = amplitude_D_*x/(2*sigma_2_) * np.exp(-r2_/(2*sigma_2_))
        grad_D_y_     = amplitude_D_*y/(2*sigma_2_) * np.exp(-r2_/(2*sigma_2_))

        return D_, grad_D_x_, grad_D_y_
############################################################
############################################################
    def diffusion_gaussian_bump(self, x, y ):

        r2_           = x*x + y*y;
        D0_          = self.D0_
        sigma_       = self.sigma_
        amplitude_D_ = self.amplitude_D_
        sigma_2_     = sigma_ * sigma_
        D_            = D0_ + amplitude_D_*np.exp(-r2_/(2*sigma_2_))
        grad_D_x_     = - amplitude_D_*x/(2*sigma_2_) * np.exp(-r2_/(2*sigma_2_))
        grad_D_y_     = - amplitude_D_*y/(2*sigma_2_) * np.exp(-r2_/(2*sigma_2_))

        return D_, grad_D_x_, grad_D_y_

############################################################
############################################################
    def diffusion_linear(self, x, y):


        D0_          = self.D0_
        amplitude_D_ = self.amplitude_D_
        L_           = self.L_

        D_           = D0_ - amplitude_D_/2 + amplitude_D_/(2*L_)*(x+L_)

        grad_D_x_ = amplitude_D_/(2*L_)
        grad_D_y_ = 0.
        return D_, grad_D_x_, grad_D_y_
############################################################
############################################################
    def potential_trap(self, x, y):


        amplitude_V_ = self.amplitude_V_
        r2_           = x*x + y*y;
        sigma_       = self.sigma_
        sigma_2_     = sigma_ * sigma_

        V_    = 0 -  amplitude_V_*np.exp(-r2_/(2*sigma_2_))
        fx_   = -amplitude_V_*x/(sigma_2_)* np.exp(-r2_/(2*sigma_2_))
        fy_   = -amplitude_V_*y/(sigma_2_)* np.exp(-r2_/(2*sigma_2_))

        return V_, fx_, fy_

############################################################
############################################################
    def potential_drift(self, x, y):

        amplitude_V_ = self.amplitude_V_
        L_           = self.L_

        V_    = 0. -  amplitude_V_/2 + amplitude_V_/(2*L_)*(x+L_)
        fx_   = -amplitude_V_/(2*L_)
        fy_   = 0.

        return V_, fx_, fy_
############################################################
############################################################
    def give_gamma_equilibirum(self, D):

        gamma  = 1/D
        return gamma

############################################################
############################################################
    def give_gamma_fixed_value(self):

        gamma = self.gamma_

        return gamma

############################################################
############################################################
    def give_diffusion_drift_gamma(self, u, v):

        mode_D     = self.mode_D
        mode_V     = self.mode_V
        mode_gamma = self.mode_gamma

        if mode_D == "linear":
            D_, grad_D_x_, grad_D_y_ = self.diffusion_linear(u, v)
        elif mode_D == "gaussian_trap":
            D_, grad_D_x_, grad_D_y_ = self.diffusion_gaussian_trap( u, v )
        elif mode_D == "gaussian_bump":
            D_, grad_D_x_, grad_D_y_ = self.diffusion_gaussian_bump( u, v )

        if mode_V == "potential_force":
            V_, fx_, fy_ = self.potential_trap(u, v)
        elif mode_V == "potential_linear":
            V_, fx_, fy_ = self.potential_drift( u, v)

        if mode_gamma    == "equilibrium":
            gamma_       = self.give_gamma_equilibirum( D_)
        elif  mode_gamma == "fixed":
            gamma_       = self.give_gamma_fixed_value()

        return     D_ , grad_D_x_ , grad_D_y_ , gamma_ , V_ , fx_ , fy_


############################################################
############################################################


    def create_traj(self):
        nb_       = self.nb_
        x_        = self.x_
        y_        = self.y_
        t_        = self.t_
        traj      = pd.DataFrame(collections.OrderedDict([("n", nb_), ("x", x_), ("y", y_), ("t", t_)]))
        self.traj = traj

############################################################
############################################################
    def print_traj(self):
        traj           = self.traj
        name_file_out  = self.name_file_out
        traj.to_csv(name_file_out, sep='\t', index=False, header=False)
############################################################
############################################################
    def update_position(self, u_ , v_ , t_ ):
        dt_short_ = self.dt_short_
        lambda_   = self.lambda_
        D_, grad_D_x_, grad_D_y_, gamma_, V_, fx_, fy_ = self.give_diffusion_drift_gamma( u_, v_)

        u_  = u_ + (fx_/gamma_ + lambda_ *grad_D_x_)*dt_short_ +  np.sqrt(2 * D_ * dt_short_) * np.random.randn()
        v_  = v_ + (fy_/gamma_ + lambda_ *grad_D_y_)*dt_short_ +  np.sqrt(2 * D_ * dt_short_) * np.random.randn()
        t_  = t_ + dt_short_
        return u_, v_, t_
############################################################
############################################################
    def generate_the_actual_trajectories(self):

        n_total_size_max_ = self.n_total_size_max_
        n_trajs_          = self.n_trajs_
        n_short_          = self.n_short_
        dt_space_         = self.dt_space_

        nb_               = []#np.empty(n_total_size_max_, dtype="int")
        x_                = []#np.empty(n_total_size_max_, dtype="float16")
        y_                = []#np.empty(n_total_size_max_, dtype="float16")
        t_                = []#np.empty(n_total_size_max_, dtype="float16")


        indice_ = 0
        t       = 0
        for i in range(1,n_trajs_ + 1):
            if self.verbose:
                print(i)
            t                 = t + dt_space_
            u_, v_            = self.random_start()
            #nb_[indice_]      = i
            #x_[indice_]       = u_
            #y_[indice_]       = v_
            #t_[indice_]       = t
            nb_.append(i)
            x_.append(u_)
            y_.append(v_)
            t_.append(t)
            indice_           = indice_ + 1
            N_duration_       = self.traj_duration()
            N_duration_       = np.round(N_duration_)
            for j in range(1, int( N_duration_ * n_short_ + 1) ):
                #D_ , grad_D_x_ , grad_D_y_ , gamma_ , V_ , fx_ , fy_ = self.give_diffusion_drift_gamma( u_, v_)
                u_, v_, t  = self.update_position( u_ , v_ , t )
                if j%n_short_ == 0:
                    #nb_[indice_]      = i
                    #x_[indice_]       = u_
                    #y_[indice_]       = v_
                    #t_[indice_]       = t
                    nb_.append(i)
                    x_.append(u_)
                    y_.append(v_)
                    t_.append(t)
                    indice_           = indice_ + 1


        #nb_               = nb_[:indice_]
        #x_                = x_[:indice_]
        #y_                = y_[:indice_]
        #t_                = t_[:indice_]
        nb_              = np.array(nb_)
        x_              = np.array(x_)
        y_              = np.array(y_)
        t_              = np.array(t_)

        self.nb_         = nb_
        self.x_          = x_
        self.y_          = y_
        self.t_          = t_



        self.indice_final_ = indice_




############################################################
############################################################


def random_walk_2d(*args, **kwargs):
    """
    Generate 2D trajectories.

    Arguments:

        lambda (float): stochastic integral lambda = 0 ito, 1/2 stratonovich, 1 klimintovitch(?)
        sigma (float):  width of the trap
        sigma_noise (float):   positioning noise
        dt (float):     time between frames
        n_short (int):  number of small time steps between the returned time steps
        N_mean (int):   mean duration of the trajectories in number of points
        n_trajs (int):  number of individual trajectories
        L (float):      size of the trap in microns (half)
        density (float):       effective density of the random walks
        D0 (float):     local diffusivity
        amplitude_D (float):   amplitude of diffusivity variation
        amplitude_V (float):   ampltiude of the trap or slope
        mode_D (str):   any of '*linear*', '*gaussian_trap*', '*gaussian_bump*'
        mode_V (str):   any of '*potential_force*', '*potential_linear*'
        mode_gamma (str):       any of '*equilibrium*', '*fixed*'
        verbose (bool): print trajectory index (default is False)

    Returns:

        pandas.DataFrame:
            column '*n*' is trajectory index, columns '*x*' and '*y*' are spatial
            coordinates (in micrometers) and column '*t*' is time (in seconds).

    See also :class:`generate_trajectories` for default values.

    """
    # rename the arguments with a trailing underscore
    for arg in list(kwargs.keys()):
        if arg in ('lambda', 'sigma', 'sigma_noise', 'dt', 'n_short', 'N_mean', 'n_trajs',
                'L', 'density', 'D0', 'amplitude_D', 'amplitude_V'):
            kwargs[arg+'_'] = kwargs.pop(arg)
    # turn verbosity off by default
    kwargs['verbose'] = kwargs.get('verbose', False)
    # generate trajectories
    gen = generate_trajectories(*args, **kwargs)
    gen.generate_the_actual_trajectories()
    gen.noisify()
    # return a DataFrame
    gen.create_traj()
    return gen.traj


############################################################
############################################################


if __name__ == '__main__':

    #“”””
    #One imagine that some of those words were attached to actual meaning of some sort.
    #“””

    import warnings
    warnings.filterwarnings('ignore')

    # check the parameters np and pandas for weird bug
    print(np.__version__)
    print(pd.__version__)

    # parameters to be changed to generate
    lambda_         = 1 # stochastic integral lambda = 0 ito, 1/2 strato , 1 klimintovitvh
    sigma_          = 0.4 # width of the trap
    sigma_noise_    = 0.010 # positioning noise
    dt_             = 25e-3 # time between frames
    dt_space_       = 10 * dt_ # time between trajectories
    n_short_        = 1000 # number of small time steps between the time stepds oututed
    N_mean_         = 5 # mean duration of the traj in number of poitns
    n_trajs_        = 200 # number of individual trajs
    L_              = 0.5 # size of the trap in microns (half)
    dt_short_       = dt_ /n_short_ # short time between steps in the generation trajs
    density_        = 2. # effective desnity of RW
    Nb_tot_loc_     = density_*L_*L_ #nombre tot
    N_duration_tot_ = np.round(n_trajs_ /Nb_tot_loc_) # estimate duration
    #############################################
    D0_             = 0.1 # local diffusivity
    amplitude_D_    = 0.02 # amplitude of diffusivity variation
    amplitude_V_    = 5 # ampltidue of the trap
    #############################################
    mode_D          ="gaussian_bump"
    mode_V          ="potential_force"
    mode_gamma      = "equilibrium"
    name_file_out   ="trajectories.txt"

    gen = generate_trajectories(lambda_ = lambda_,
        sigma_ = sigma_,
        sigma_noise_ = sigma_noise_,
        dt_ = dt_,
        #dt_space_ = dt_space_,
        n_short_ = n_short_,
        N_mean_ = N_mean_,
        n_trajs_ = n_trajs_,
        L_ = L_,
        #dt_short_ = dt_short_,
        density_ = density_,
        #Nb_tot_loc_ = Nb_tot_loc_,
        #N_duration_tot_ = N_duration_tot_,
        D0_ = D0_,
        amplitude_D_ = amplitude_D_,
        amplitude_V_ = amplitude_V_,
        mode_D = mode_D,
        mode_V = mode_V,
        mode_gamma = mode_gamma,
        name_file_out = name_file_out )
    gen.generate_the_actual_trajectories()
    gen.noisify()
    gen.create_traj()
    gen.print_traj()

    #print(gen.nb_.size)
    #print(gen.x_.size)
    #print(gen.y_.size)
    #print(gen.t_.size)

