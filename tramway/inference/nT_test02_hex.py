
from .base import *
import numpy as np
import pandas as pd
from time import time
from numpy import array, reshape, sum, zeros, ones, arange, dot, max, argmax, log, exp, sqrt
from scipy import optimize as optim
from scipy.special import iv
from scipy.special import gamma as g
from scipy.special import gammaln as lng
from scipy.misc import logsumexp
from scipy.optimize import linear_sum_assignment as kuhn_munkres
from scipy.stats import skellam
from scipy.optimize import minimize


setup = {
    'infer': 'nT_test02_hex', 
    'sampling': 'group' ,
    'arguments': {'cell_index': {}, 'dt': {}, 'p_off': {}, 's2': {}, 'D_bounds': {}, 'D0': {}, 'method': {},\
                  'tol': {}, 'times': {}}
}


#-----------------------------------------------------------------------------
# Main functions
#-----------------------------------------------------------------------------
def nT_test02_hex(cells, dt=0.04, p_off=0., mu_on=0., s2=0.0025, D0=np.array([0.2,0.2]), method='NM',tol=1e-3,times=[0.]):
    #index, _, _, _, D0, _, _, _ = smooth_infer_init(cells)
    index = list(cells.keys())
    D=[]
    for i in index:
        D.append(estimate_D_2zones(i,cells,dt,p_off,mu_on,s2,D0,method,tol,times))
    D_out=pd.DataFrame(np.array(D), index=index, columns=['D'])
#    print(D_out)
    return D_out

def estimate_D_2zones(i,cells,dt,p_off,mu_on,s2,D0,method,tol,times):
    assert cells[i]
    cell = cells[i]

    # Positions in cell i:
    r_in = cell.r
    t_in = cell.t
    S_in = cell.volume

    # Return 0 if no positions are in cell:
    if len(t_in)==0:
        return 0.

    # Positions in neighboring cells:
    r_out=[] 
    t_out=[] 
    S_out=0.
    for j in cells.neighbours(i):
        cell = cells[j]
        r = cell.r
        t = cell.t
        S = cell.volume
        r_out.extend(r) 
        t_out.extend(t) 
        S_out += S
    r_out=np.array(r_out)
    t_out=np.array(t_out)

    # Positions in cell i and neighbors:
    r_total=list(r_in.copy())
    r_total.extend(r_out)
    t_total=list(t_in.copy())
    t_total.extend(t_out)
    r_total=np.array(r_total)
    t_total=np.array(t_total)
    S_total=S_in+S_out

    # List of times corresponding to frames:
    times=np.arange(min(t_total),max(t_total+dt/100.),dt)

    # Build lists of frames:
    frames=rt_to_frames(r_total,t_total,dt,times)
    N_t=[len(frame) for frame in frames]
    poss_links_total=sum([min([N_t[n],N_t[n+1]]) for n in range(len(N_t)-1)])
    frames_in = rt_to_frames(r_in,t_in,dt,times)
    N_in=[len(frame) for frame in frames_in]
    poss_links_in=sum([min([N_in[n],N_in[n+1]]) for n in range(len(N_in)-1)])
    frames_out = rt_to_frames(r_out,t_out,dt,times)
    N_out=[len(frame) for frame in frames_out]
    poss_links_out=sum([min([N_out[n],N_out[n+1]]) for n in range(len(N_out)-1)])
    print("\nCell no.:", i, ", frames:", len(frames)) 
    print("Inner cell: avg(N):", np.average(N_in), "S:", S_in, "mu^2")
    print("Outer cells: avg(N):", np.average(N_out), ", S:", S_total, "mu^2")
    print("Possible links:", poss_links_total, ", avg(N):", np.average(N_t))

    # List of distance matrices:
    drs_ij=frame_stack_to_distance_matrices(frames) 

    # Average number of particles disappearing per frame:
    mu_off=np.average(N_t)*p_off

    #--- Fit: ---
    fittime=time()
    if method=='NM':
        fit = minimize(marginal_minusLogLikelihood_multiFrame,x0=D0,args=(drs_ij,dt,S_total,mu_off,mu_on,N_t,N_in),\
                       method='Nelder-Mead',tol=tol)        
    D_i=abs((fit.x)[0])
    print(fit)
#    D_i_ub=max([1.5*(D_i-s2/dt),0.])
#    print("D corrected for motion blur and localization error:", D_i_ub)
    print(time()-fittime, "s")

    return D_i

#-----------------------------------------------------------------------------
# Generate stack of frames
#-----------------------------------------------------------------------------
def rt_to_frames(r,t,dt,times):
#    times=np.arange(min(t),max(t+dt/100.),dt)
    frames=[]
    for t_ in times:
        frames.append(np.array(r[(t<t_+dt/100.)&(t>t_-dt/100.)]))
    return frames

def frame_stack_to_distance_matrices(frames):
    return [positions_to_distance_matrix(frames[n],frames[n+1]) for n in range(len(frames)-1)]

#-----------------------------------------------------------------------------
# Helper functions for tracking/non-tracking
#-----------------------------------------------------------------------------
def positions_to_distance_matrix(x,y):
    '''Matrix of distances between all measured points in two succesive frames.'''
    N_=len(x)
    M_=len(y)
    dr_ij=np.zeros([2,N_,M_])
    for i in range(N_):
        dr_ij[0,i]=y[:,0]-x[i,0]
        dr_ij[1,i]=y[:,1]-x[i,1]
    return dr_ij

def lnp_ij(dr,D,dt,minlnL=-100):
    '''Matrix of log-probabilities for each distance.'''
    lnp=-log(4*np.pi*D*dt)-(dr[0]**2+dr[1]**2)/(4.*D*dt)
    lnp[lnp<minlnL]=minlnL
    return lnp

def Q_ij(S,N,n_off,m_on,lnp,minlnL=-100):
    '''Matrix of log-probabilities for assignments with appearances and disappearances.'''
    M = N-n_off+m_on
    out = zeros([m_on+N,m_on+N])
    LM=ones([m_on+N,m_on+N])*S
    out[:N,:M]=lnp-log(LM[:N,:M])
    out[:N,M:]=-log(LM[:N,M:])
    out[N:,:M]=-log(LM[N:,:M])
    out[N:,M:]=2.*minlnL 
    return out

def p_m(x,Delta,mu_on,mu_off):
    '''Poissonian probability for m_on particles to appear between frames given Delta=M-N 
    and blinking parameters mu_on and mu_off.'''
    return (mu_on*mu_off)**(x-Delta/2.)/(g(x+1.)*g(x+1.-Delta)*iv(Delta,2*sqrt(mu_on*mu_off)))

#-----------------------------------------------------------------------------
# Sum-product BP
#-----------------------------------------------------------------------------
def initialize_messages_zeros(N):
    '''Initializes BP to zero'''
    hij=zeros([N,N])
    hji=zeros([N,N])
    return hij,hji

def boolean_matrix(N):
    '''Boolean matrix for parralel implementation of BP.'''
    bool_array=ones([N,N])
    for i in range(N):
        bool_array[i,i]=0.
    return bool_array

def bethe_free_energy(hij,hji,Q):
    '''Bethe free energy given BP messages hij and hji and log-probabilities Q.'''
    return + sum( log(   1. + exp(Q+hij+hji)    ) )\
           - sum( log( sum( exp(Q+hji), axis=1) ) )\
           - sum( log( sum( exp(Q+hij), axis=0) ) )

def sum_product_BP(bool_array,gamma,epsilon,max_it,Q,hij,hji):
    '''Sum-product (BP) free energy (minus-log-likelihood) and BP messages.'''
    hij_old=hij
    hji_old=hji
    F_BP_old=bethe_free_energy(hij_old,hji_old,Q)
    for n in range(max_it):
        hij_new=-log( dot(exp(Q+hji_old),bool_array) )
        hji_new=-log( dot(bool_array,exp(Q+hij_old)) )
        hij=(1.-gamma)*hij_new + gamma*hij_old
        hji=(1.-gamma)*hji_new + gamma*hji_old
        # Stopping condition:
        F_BP=bethe_free_energy(hij,hji,Q)
        if abs(F_BP-F_BP_old)<epsilon:
            break
        # Update old values of energy and messages:    
        F_BP_old=F_BP
        hij_old=hij
        hji_old=hji
    return F_BP,hij,hji,n

def sum_product_energy(D_in,D_out,dt,dr,gamma,epsilon,max_it,S,N,n_off,m_on,N_in,M_in):
    '''Bethe free energy of given graph.'''
    M=N-n_off+m_on
    Ds=zeros([N,M])
    Ds[:N_in]=ones([N_in,M])*D_in
    Ds[N_in:]=ones([N-N_in,M])*D_out
    # If a single particle has been recorded in each frame and tracers are permanent, the link is known:
    if (N==1)&(n_off==0)&(m_on==0):
        F_BP=-lnp_ij(dr,Ds,dt)
    # Else, perform BP to obtaine the Bethe free energy:
    else:
        bool_array=boolean_matrix(m_on+N)
        Q=Q_ij(S,N,n_off,m_on,lnp_ij(dr,Ds,dt))
        hij,hji=initialize_messages_zeros(m_on+N)
        F_BP,hij,hji,n=sum_product_BP(bool_array,gamma,epsilon,max_it,Q,hij,hji)
    return F_BP

def marginal_minusLogLikelihood(D_in,D_out,dt,dr,gamma,epsilon,max_it,S,mu_off,mu_on,N,M,N_in,M_in):
    '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
    Delta=M-N
    if (mu_off==0.)&(mu_on==0.)&(Delta==0):
        mlnL=sum_product_energy(D_in,D_out,dt,dr,gamma,epsilon,max_it,S,N,0,0,N_in,M_in)
    elif (M==0) or (N==0):
        mlnL=0.
    elif N_in==0:
        mlnL=0.
    else:
        mlnL=-logsumexp([log(p_m(m,Delta,mu_on,mu_off))+lng(M+1-m)-lng(M+1)-lng(N+1)\
                        -sum_product_energy(D_in,D_out,dt,dr,gamma,epsilon,max_it,S,N,m-Delta,m,N_in,M_in)\
                             for m in range(max([0,Delta]),M)])
    return mlnL    

def marginal_minusLogLikelihood_multiFrame(D_,drs,dt,S,mu_off,mu_on,Ns,N_in,gamma=0.8,\
                                           epsilon=1e-8,max_it=100000):
    '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
    mlnL=0
    for n,dr in enumerate(drs):
        mlnL += marginal_minusLogLikelihood(abs(D_[0]),abs(D_[1]),dt,dr,gamma,epsilon,max_it,S,mu_off,mu_on,Ns[n],Ns[n+1],\
                                            N_in[n],N_in[n+1])
    return mlnL    
