# -------------------------------
# Libraries
# -------------------------------
import os.path
from math import *
import numpy as np
from tramway.helper import *
from tramway.helper.simulation import *
from tramway.helper.tessellation import *
from tramway.helper.inference import *
from tramway.plot.mesh import plot_delaunay
#from scipy.stats import skellam
#from scipy.optimize import fmin,minimize

#--------------------------------
# Functions
#--------------------------------
def D_field(r,t):
	'''Function defining the diffusivity field.'''
    mu_r=np.array([0.5,0.5])
    sig_r=0.05
    mu_t=0.5
    sig_t=0.05
    return D - (D-D0)*np.exp(-np.sum((r/r_scale-mu_r)**2)/sig_r) #*np.exp(-(t-mu_t)**2/(sig_t))

def new_tracers(t):
	'''Function drawing the number of tracers to appear each time-step from a Poisson distribution.'''
    return np.random.poisson(mu_on)

#--------------------------------
# Constants:
#--------------------------------
# Size of the bounding box:
r_scale = 10. # [um]
# number of trajectories:
M = 100
# time step:
dt = 0.05 # [s]
# Simulation duration:
T = 1. # [s]
# average trajectory lifetime:
tau = 0.25
# baseline diffusivity:
D = 0.5 # [um^2.s^-1]
# diffusivity at the bottom of the sink:
D0 = 0.05 # [um^2.s^-1]
# position noise:
sigma = 0. # [um]
# Reference distance to set mesh size:
ref_distance = 1. # [mu]

# Tesselation method and name:
tessellation_method = 'hexagon'
mesh_label = '{}'.format(tessellation_method) #, location_count, min_location_count)
min_loc_count = 0
#--------------------------------
# Simulate:
#--------------------------------
## generate the trajectories
nxyt = random_walk(diffusivity=D_field, duration=T, lifetime_tau=tau, single=True, reflect=True, 
                  trajectory_mean_count=M, box=[0.,0.,10.,10.], )
nxyt = nxyt.dropna()
xyt = nxyt.drop('n',axis=1)

# Tesselate:
cells = tessellate(positions, tessellation_method, ref_distance=ref_distance, min_location_count=0,
           strict_min_location_count=min_loc_count,
           force=True, output_label=mesh_label)
