# --------------------------------------------------------------------------------------------------
# Libraries
# --------------------------------------------------------------------------------------------------
from sys import argv
from time import time
import matplotlib.pylab as plt
import os.path
from math import *
import numpy as np
from tramway.helper import *
from tramway.helper.simulation import *
from tramway.helper.tessellation import *
from tramway.helper.inference import *
from tramway.plot.mesh import plot_delaunay
from numpy.random import seed

#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------
def D_field(r,t):
    '''Defining the diffusivity field: Gaussian in space, constant in time.'''
    return D - (D-D0)*np.exp(-np.sum((r/r_scale-mu_r)**2)/sig_r) 

def new_tracers(t):
    '''Number of tracers to appear in each frame n ~ Poisson(mu_on).'''
    return np.random.poisson(mu_on)

def inner_cells(cells_):
    indices  = cells_.tessellation.cell_vertices
    vertices = cells_.tessellation.vertices

    cells_to_include = []

    for i in range(len(indices)):
        cells_to_include.append( not ((vertices[indices[i]]<=0)|(vertices[indices[i]]>=r_scale)).any() )

    return cells_to_include

#---------------------------------------------------------------------------------------------------
# Constants and parameters:
#---------------------------------------------------------------------------------------------------
# Tesselation method and name:
tessellation_method = 'hexagon'
mesh_label = '{}'.format(tessellation_method)
min_loc_count = 0

# Optimization parameters:
nt_method = ('nT_test02_hex', 'NM')
tol = 1e-2

#            D0,   D  , tau, dt,   sig_r, r_scale, T,   M,  sigma, r_cell, Q        
# argv = [0. , 0.03, 0.3, 0.2, 0.04, 0.05,  5.,      0.25, 10, 0.,    1.,     2]
# Normalized placement and width of diffusivity well:
mu_r=np.array([0.5,0.5]) 
sig_r=float(argv[5])
print(f"sig_r={sig_r}")

# Size of the bounding box:
r_scale = float(argv[6]) # [um]
print(f"r_scale={r_scale}")

# Reference distance to set mesh size:
ref_distance = float(argv[10]) # [mu]
print(f"ref_distance={ref_distance}")

# baseline diffusivity:
D = float(argv[2]) # [um^2.s^-1]
print(f"D={D}")

# diffusivity at the bottom of the sink:
D0 = float(argv[1]) # [um^2.s^-1]
print(f"D0={D0}")

# number of trajectories:
M = int(argv[8])
print(f"M={M}")

# time step:
dt = float(argv[4]) # [s]
print(f"dt={dt}")

# average trajectory lifetime (and p_off and mu_on):
tau = float(argv[3])
p_off = dt/tau
mu_on = p_off*M
print(f"tau={tau}")

# Simulation duration:
T = float(argv[7]) # [s]
print(f"T={T}")

# position noise:
sigma = float(argv[9]) # [um]
print(f"sigma={sigma}")

# Number of ensembles to simulate:
Q = int(argv[11])
print(f"Q={Q}")

#---------------------------------------------------------------------------------------------------
# Simulate:
#---------------------------------------------------------------------------------------------------
seed(1234567890)

# Empty lists for results:
Ds_nxyt = []
Ds_xyt = []

# Loop over repetitions:
for q in range(Q):
    print(f"\n--------------------------------------------")
    print(f"Simulation no.: {q+1}/{Q}")
    print(f"--------------------------------------------")
    
    # Simulate trajectories:
    data_nxyt  = random_walk(diffusivity=D_field, duration=T, lifetime_tau=tau, single=True, reflect=True, 
                      trajectory_mean_count=M, box=[0.,0.,r_scale,r_scale], )

    data_nxyt  = data_nxyt.dropna()
    
    # tesselate for trajectories:
    cells_nxyt = tessellate(data_nxyt, tessellation_method, ref_distance=ref_distance, min_location_count=0,
                strict_min_location_count=min_loc_count, force=True, output_label=mesh_label)
    
    # tesselate for positions:
    data_xyt   = data_nxyt.drop('n',axis=1)

    cells_xyt  = tessellate(data_xyt, tessellation_method, ref_distance=ref_distance, min_location_count=0,
                strict_min_location_count=min_loc_count, force=True, output_label=mesh_label)
    
    # Select only cells that are entirely within box for further treatment:
    cells_to_include = inner_cells(cells_xyt)    

    cells_xyt.tessellation.cell_label  = np.array(cells_to_include)
    cells_nxyt.tessellation.cell_label = np.array(cells_to_include)

    # --------------------------------------------------------------------------------------------
    # infer in D mode without spatial regularization
    # --------------------------------------------------------------------------------------------
    # Create map with true values of D in cell centers:
    centers = cells_nxyt.tessellation.cell_centers

    D_true  = pd.DataFrame([D_field(c_,0.) for c_ in centers],columns=['D'])
    
    #--- Infer using true trajectories: ---
    maps_nxyt = infer(cells_nxyt, 'D', sigma=sigma, verbose=False)

    # Save analysis-tree as .rwa file:
    analysis_tree_nxyt = Analyses(data_nxyt)
    analysis_tree_nxyt.add(cells_nxyt, label='mesh')
    analysis_tree_nxyt['mesh'].add(maps_nxyt, label='diffusivity')

    save_rwa(f'analyses_nxyt,D=[{D0},{D}],tau={tau},dt={dt},sig_r={sig_r},box={r_scale},T={T},M={M},r_cell={ref_distance},q={q}.rwa', 
             analysis_tree_nxyt, force=True)

    
    # Add inferred values to array:
    Ds_nxyt.append(maps_nxyt['diffusivity']['diffusivity'].values)
    
    #--- Infer from positions using BP: ---
    maps_xyt = infer(cells_xyt, nt_method[0], new_cell=Locations, dt=dt, p_off=p_off, mu_on=mu_on,
                method=nt_method[1], tol=tol)
    
    # Save analysis-tree as .rwa file:
    analysis_tree_xyt = Analyses(data_xyt)
    analysis_tree_xyt.add(cells_xyt, label='mesh')
    analysis_tree_xyt['mesh'].add(maps_xyt, label='diffusivity')
    
    save_rwa(f'analyses_xyt,D=[{D0},{D}],tau={tau},dt={dt},sig_r={sig_r},box={r_scale},T={T},M={M},r_cell={ref_distance},q={q}.rwa', 
             analysis_tree_xyt, force=True)
    
    # Add inferred values to array:
    Ds_xyt.append(maps_xyt['D']['D'].values)

# Save inferred values as text:
np.save(f'D_nxyt,D=[{D0},{D}],tau={tau},dt={dt},sig_r={sig_r},box={r_scale},T={T},M={M},r_cell={ref_distance},Q={Q}.txt', Ds_nxyt)
np.save(f'D_xyt,D=[{D0},{D}],tau={tau},dt={dt},sig_r={sig_r},box={r_scale},T={T},M={M},r_cell={ref_distance},Q={Q}.txt', Ds_xyt)

print("Done!")