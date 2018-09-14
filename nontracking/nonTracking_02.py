#=============================================================================
# Import libraries
#=============================================================================
from time import time
from sys import argv
import numpy as np
from matplotlib import pylab as plt
from numpy.random import seed, rand, randn
from numpy import array, reshape, sum, zeros, ones, arange, dot, max, argmax
from numpy import log, exp, sqrt
from scipy import optimize as optim
from scipy.special import iv
from scipy.special import gamma as g
from scipy.special import gammaln as lng
from scipy.misc import logsumexp
from scipy.optimize import linear_sum_assignment as kuhn_munkres
from scipy.stats import skellam

#=============================================================================
# Parameters of the BP algorithm:
#=============================================================================
minlnL=-100. # Cut-off for log-likelihood (to avoid numerical underflow)
gamma=0.8 # Damping factor
epsilon=1e-8 # Error tolerance
max_it=100000 # Maximum number of BP iterations per evaluation of F

#=============================================================================
# Functions
#=============================================================================
#-----------------------------------------------------------------------------
# Functions for simulating positions of Brownian tracers
#-----------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------
# Log-likelihood for the true assignment
#-----------------------------------------------------------------------------
def true_energy(D,dt,dr,rho):
    return sum( -log(rho) + log(4*np.pi*D*dt)+(dr[0]**2+dr[1]**2)/(4.*D*dt) )

def true_energy_multiframe(D,dt,drs,rho):
    return sum( [true_energy(D,dt,dr,rho) for dr in drs] ) 

#-----------------------------------------------------------------------------
# Helper functions for tracking/non-tracking
#-----------------------------------------------------------------------------
def distance_matrices(x,y):
    '''Matrix of distances between all measured points in two succesive frames.'''
    N=x.shape[1]
    M=y.shape[1]
    dr_ij=zeros([2,N,M])
    for i in range(N):
        dr_ij[0,i]=y[0]-x[0,i]
        dr_ij[1,i]=y[1]-x[1,i]
    return dr_ij

def lnp_ij(dr,D,dt,minlnL=-100):
    '''Matrix of log-probabilities for each distance.'''
    lnp=-log(4*np.pi*D*dt)-(dr[0]**2+dr[1]**2)/(4.*D*dt)
    lnp[lnp<minlnL]=minlnL
    return lnp

def Q_ij(L,N,n_off,m_on,lnp,minlnL=-100):
    '''Matrix of log-probabilities for assignments with appearances and disappearances.'''
    M = N-n_off+m_on
    out = zeros([m_on+N,m_on+N])
    LM=ones([m_on+N,m_on+N])*L
    out[:N,:M]=lnp-2.*log(LM[:N,:M])
    out[:N,M:]=-2.*log(LM[:N,M:])
    out[N:,:M]=-2.*log(LM[N:,:M])
    out[N:,M:]=2.*minlnL 
    return out

def p_m(x,Delta,mu_on,mu_off):
    '''Poissonian probability for m_on particles to appear between frames given Delta=M-N 
    and blinking parameters mu_on and mu_off.'''
    return (mu_on*mu_off)**(x-Delta/2.)/(g(x+1.)*g(x+1.-Delta)*iv(Delta,2*sqrt(mu_on*mu_off)))

#-----------------------------------------------------------------------------
# Kuhn-Munkres MPA
#-----------------------------------------------------------------------------
def MPA(D,dt,dr,L,N,n_off,m_on):
    Q=Q_ij(L,N,n_off,m_on,lnp_ij(dr,D,dt))
    mpa_row,mpa_col=kuhn_munkres(-Q)
    return mpa_row,mpa_col

def MPA_energy(D,dt,dr,L,N,n_off,m_on,mpa_row,mpa_col):
    Q=Q_ij(L,N,n_off,m_on,lnp_ij(dr,D,dt))
    return -sum(Q[mpa_row,mpa_col])

def mMPAscore(D,dt,dr,L,mu_on,mu_off,N,M):
    Delta=M-N
    if(mu_off==0.)&(mu_on==0.)&(Delta==0):
        mpa_row,mpa_col=MPA(D,dt,dr,L,N,0,0)
        max_score=-MPA_energy(D,dt,dr,L,N,0,0,mpa_row,mpa_col)
    else:
        scores_=[]
        for m_ in arange(max([0,Delta]),M):
            mpa_row,mpa_col=MPA(D,dt,dr,L,N,m_-Delta,m_)
            logp_m=log(p_m(m_,Delta,mu_on,mu_off))
            E_MPA=MPA_energy(D,dt,dr,L,N,m_-Delta,m_,mpa_row,mpa_col)
            score=logp_m-E_MPA+lng(M+1-m_)-lng(M+1)-lng(N+1)
            scores_.append((score))
        max_score=max(scores_)
    return -max_score
   
def MPA_minusLogLikelihood_multiFrame(D,dt,drs,L,mu_off,mu_on,Ns):
    mlnL=0
    for n,dr in enumerate(drs):
        mlnL += mMPAscore(D,dt,dr,L,mu_on,mu_off,Ns[n],Ns[n+1])
    return mlnL

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

def sum_product_energy(D,dt,dr,gamma,epsilon,max_it,L,N,n_off,m_on):
    '''Bethe free energy of given graph.'''
    bool_array=boolean_matrix(m_on+N)
    Q=Q_ij(L,N,n_off,m_on,lnp_ij(dr,D,dt))
    hij,hji=initialize_messages_zeros(m_on+N)
    F_BP,hij,hji,n=sum_product_BP(bool_array,gamma,epsilon,max_it,Q,hij,hji)
    return F_BP

def marginal_minusLogLikelihood(D,dt,dr,gamma,epsilon,max_it,L,mu_off,mu_on,N,M):
    '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
    Delta=M-N
    if(mu_off==0.)&(mu_on==0.)&(Delta==0):
        mlnL=sum_product_energy(D,dt,dr,gamma,epsilon,max_it,L,N,0,0)
    else:
        mlnL=-logsumexp([log(p_m(m,Delta,mu_on,mu_off))+lng(M+1-m)-lng(M+1)-lng(N+1)\
                                                 -sum_product_energy(D,dt,dr,gamma,epsilon,max_it,L,N,m-Delta,m)\
                                                       for m in range(max([0,Delta]),M)])
    return mlnL    

def marginal_minusLogLikelihood_multiFrame(D,dt,drs,L,mu_off,mu_on,Ns,gamma=0.8,epsilon=1e-8,max_it=100000):
    '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
    mlnL=0
    for n,dr in enumerate(drs):
        mlnL += marginal_minusLogLikelihood(D,dt,dr,gamma,epsilon,max_it,L,mu_off,mu_on,Ns[n],Ns[n+1])
    return mlnL    
