import numpy as np
from numpy import array, reshape
from numpy.random import rand
#=============================================================================
# Functions
#=============================================================================
def simulate_BM(L, rho, D, dt, p_off, mu_on, nF, N):
    '''Generates sequence of distance matrices between all points in each pair 
    of succesive frames as well as the distances corresponding to the true
    matchings.'''
    drs_ij=[]
    drs_true=[]
    Ns=[N]
    Ns_true=[]
    for frame in range(nF):
        x=array([L*rand(N)-L/2.,L*rand(N)-L/2]) # Initialize positions inside square
        dr_true=reshape(sqrt(2*D*dt)*randn(2*N),[2,N]) # Particle displacements
        y_true=x+dr_true # Positions in subsequent frame
        #--- Simulate Blinking: ------------------------------------------------------
        # Particles disappearing:
        i_off=[]
        for i in range(N):
            if rand()<p_off:
                i_off.append(i)
        n_off=len(i_off)
        dr_true=array([np.delete(dr_true[0],i_off),np.delete(dr_true[1],i_off)])
        drs_true.append(dr_true)
        # Particles appearing:
        m_on=np.random.poisson(mu_on)
        y_new=array([L*rand(m_on)-L/2.,L*rand(m_on)-L/2])+reshape(sqrt(2*D*dt)*randn(2*m_on),[2,m_on])
        #--- Calculate observed positions: -------------------------------------------
        y=array([np.append(np.delete(y_true[0],i_off),y_new[0]),np.append(np.delete(y_true[1],i_off),y_new[1])])
        N_true=N-n_off
        Ns_true.append(N_true)
        M=N_true+m_on
        Delta=M-N
        # True assignement matrix:
        pi_ij_true=zeros([N,M])
        for i_,j_ in zip(np.delete(arange(N),i_off),range(N_true)):
            pi_ij_true[i_,j_]=1.
        mpa_row_true=np.delete(arange(N),i_off)
        mpa_col_true=np.arange(N_true)
        # Matrix of distances:
        dr_ij=distance_matrices(x,y)
        drs_ij.append(dr_ij)
        # Update for next frame and add data to arrays:
        N=M
        Ns.append(N)
    return (Ns_true, drs_true, Ns, drs_ij) 
