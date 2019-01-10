
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
from scipy.optimize import minimize_scalar, minimize


setup = {
    'infer': 'nT_test01', 
    'sampling': 'group' ,
    'arguments': {'cell_index': {}, 'dt': {}, 'S': {}, 'p_off': {}, 'D_bounds': {}, 'D0': {}, 'method': {}}
}

s2=0.0001

#-----------------------------------------------------------------------------
# Main functions
#-----------------------------------------------------------------------------
def nT_test01(cells, cell_index=0, dt=0.04, S=10., p_off=0., mu_on=0., D_bounds=[0.001,5.], D0=0.2, method='NM'):
    #index, _, _, _, D0, _, _, _ = smooth_infer_init(cells)
    index = list(cells.keys())
    i = cell_index
    print "Cell number:", i
    D=[estimate_D_1zone(i,cells,dt,S,p_off,mu_on,D_bounds,D0,method)]
    return pd.DataFrame(np.array(D)) #, index=index, columns=['D'])

def estimate_D_1zone(i,cells,dt,S,p_off,mu_on,D_bounds,D0,method):
    assert cells[i]
    cell = cells[i]
    r = cell.r
    t = cell.t
    if len(t)==0:
        return 0.
    frames = rt_to_frames(r,t,dt)
    N_t=[len(frame) for frame in frames]
#    print 'N_t =', N_t, np.sum(N_t)
    drs_ij=frame_stack_to_distance_matrices(frames) 
#    print 'dr_ij:', [dr_.shape for dr_ in drs_ij]
    mu_off=np.average(N_t)*p_off
    fittime=time()
    if method=='bounded':
        fit = minimize_scalar(marginal_minusLogLikelihood_multiFrame,bounds=D_bounds,args=(dt,drs_ij,S,mu_off,mu_on,N_t),\
                       method='bounded')
    if method=='NM':
        fit = minimize(marginal_minusLogLikelihood_multiFrame,x0=D0,args=(dt,drs_ij,S,mu_off,mu_on,N_t),\
                       method='Nelder-Mead')        
    D_i=fit.x
    print fit
    print time()-fittime, "s"
    return D_i

#-----------------------------------------------------------------------------
# Generate stack of frames
#-----------------------------------------------------------------------------
def rt_to_frames(r,t,dt):
    times=np.arange(min(t),max(t)+dt/100.,dt)
    frames=[]
    for t_ in times:
        frames.append(np.array(r[t==t_]))
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
        F_BP_old=F_BP
        hij_old=hij
        hji_old=hji
    return F_BP,hij,hji,n

def sum_product_energy(D,dt,dr,gamma,epsilon,max_it,S,N,n_off,m_on):
    '''Bethe free energy of given graph.'''
    if (N==1)&(n_off==0)&(m_on==0):
        F_BP=-lnp_ij(dr,D,dt)
    else:
        bool_array=boolean_matrix(m_on+N)
        Q=Q_ij(S,N,n_off,m_on,lnp_ij(dr,D,dt))
        hij,hji=initialize_messages_zeros(m_on+N)
        F_BP,hij,hji,n=sum_product_BP(bool_array,gamma,epsilon,max_it,Q,hij,hji)
#    print('N,n_off,m_on,F_BP =',N,n_off,m_on, F_BP)
    return F_BP

def marginal_minusLogLikelihood(D,dt,dr,gamma,epsilon,max_it,S,mu_off,mu_on,N,M):
    '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
    Delta=M-N
#    print("N,M,Delta:",N,M,Delta)
    if (mu_off==0.)&(mu_on==0.)&(Delta==0):
        mlnL=sum_product_energy(D,dt,dr,gamma,epsilon,max_it,S,N,0,0)
    elif (M==0) or (N==0):
        mlnL=0.
    else:
        mlnL=-logsumexp([log(p_m(m,Delta,mu_on,mu_off))+lng(M+1-m)-lng(M+1)-lng(N+1)\
                                                 -sum_product_energy(D,dt,dr,gamma,epsilon,max_it,S,N,m-Delta,m)\
                                                       for m in range(max([0,Delta]),M)])
    return mlnL

def marginal_minusLogLikelihood_multiFrame(D,dt,drs,S,mu_off,mu_on,Ns,gamma=0.8,epsilon=1e-8,max_it=100000):
    '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
    mlnL=0
    for n,dr in enumerate(drs):
        mlnL += marginal_minusLogLikelihood(D,dt,dr,gamma,epsilon,max_it,S,mu_off,mu_on,Ns[n],Ns[n+1])
#    print D,mlnL
    return mlnL    
