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
epsilon=0.01 # Error tolerance
max_it=100 # Maximum number of BP iterations per evaluation of F

#=============================================================================
# Functions
#=============================================================================
def distance_matrices(x,y):
    '''Matrix of distances between all measured points in two succesive frames.'''
    N_=len(x[0])
    M_=len(y[0])
    dr_ij=zeros([2,N_,M_])
    for i in xrange(N_):
        dr_ij[0,i]=y[0]-x[0,i]
        dr_ij[1,i]=y[1]-x[1,i]
    return dr_ij

def lnp_ij(dr,D,dt):
    '''Matrix of log-probabilities for each distance.'''
    lnp=-log(4*np.pi*D*dt)-(dr[0]**2+dr[1]**2)/(4.*D*dt)
    lnp[lnp<minlnL]=minlnL
    return lnp

#-----------------------------------------------------------------------------
# Sum-product BP
#-----------------------------------------------------------------------------
def initialize_messages_rand(N):
    '''Randomly initializes BP messages.'''
    hij=reshape(rand(N**2),[N,N])
    hji=reshape(rand(N**2),[N,N])
    return hij,hji

def initialize_messages_Q(N,Q):
    '''Initializes BP to log p'''
    hij=-Q
    hji=-Q
    return hij,hji

def initialize_messages_Qrand(N,Q):
    '''Randomly initializes BP messages.'''
    hij=-Q*reshape(rand(N**2),[N,N])
    hji=-Q*reshape(rand(N**2),[N,N])
    return hij,hji

def initialize_messages_logrand(N):
    '''Randomly initializes BP messages.'''
    hij=reshape(log(rand(N**2)),[N,N])
    hji=reshape(log(rand(N**2)),[N,N])
    return hij,hji

def initialize_messages_Qlogrand(N,Q):
    '''Randomly initializes BP messages.'''
    hij=-Q + reshape(log(rand(N**2)),[N,N])
    hji=-Q + reshape(log(rand(N**2)),[N,N])
    return hij,hji

def boolean_matrix(N):
    '''Boolean matrix for parralel implementation of BP.'''
    bool_array=ones([N,N])
    for i in xrange(N):
        bool_array[i,i]=0.
    return bool_array

def Q_ij(L,N,n_off,m_on,lnp,nL):
    '''Matrix of log-probabilities for assignments with appearances and disappearances.'''
    M = N-n_off+m_on
    out = zeros([m_on+N,m_on+N])
    LM=ones([m_on+N,m_on+N])*L*(1+nL*np.reshape(rand((m_on+N)**2)-0.5,[m_on+N,m_on+N]))
    out[:N,:M]=lnp-2.*log(LM[:N,:M])
    out[:N,M:]=-2.*log(LM[:N,M:])
    out[N:,:M]=-2.*log(LM[N:,:M])
    out[N:,M:]=2.*minlnL #*(1+nL*np.reshape(rand(n_off*m_on)-0.5,[m_on,n_off]))
    return out

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

def sum_product_energy(D,dt,dr,gamma,epsilon,max_it,L,N,n_off,m_on,tries,nL):
    '''Bethe free energy of given graph.'''
    bool_array=boolean_matrix(m_on+N)
    Q=Q_ij(L,N,n_off,m_on,lnp_ij(dr,D,dt),nL)
    F_BPs=zeros(tries)
    for t in xrange(tries):
        hij,hji=initialize_messages_Qlogrand(m_on+N,Q)
        F_BP,hij,hji,n=sum_product_BP(bool_array,gamma,epsilon,max_it,Q,hij,hji)
        F_BPs[t]=F_BP
    F_BP=max(F_BPs)    
    return F_BP

def p_m(x,Delta,mu_on,mu_off):
    '''Poissonian probability for m_on particles to appear between frames given Delta=M-N 
    and blinking parameters mu_on and mu_off.'''
    return (mu_on*mu_off)**(x-Delta/2.)/(g(x+1.)*g(x+1.-Delta)*iv(Delta,2*sqrt(mu_on*mu_off)))

def marginal_minusLogLikelihood(D,dt,dr,gamma,epsilon,max_it,L,mu_off,mu_on,N,M,tries,nL):
    '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
    Delta=M-N
    mlnL=-logsumexp([log(p_m(m,Delta,mu_on,mu_off))+lng(M+1-m)-lng(M+1)-lng(N+1)\
                                                 -sum_product_energy(D,dt,dr,gamma,epsilon,max_it,L,N,m-Delta,m,tries,nL)\
                                                       for m in range(max([0,Delta]),M)])
    return mlnL    

def marginal_minusLogLikelihood_multiframe(D,dt,drs,gamma,epsilon,max_it,L,mu_off,mu_on,Ns,tries,nL):
    '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
    mlnL=0
    for n,dr in enumerate(drs):
        mlnL += marginal_minusLogLikelihood(D,dt,dr,gamma,epsilon,max_it,L,mu_off,mu_on,Ns[n],Ns[n+1],tries,nL)
    return mlnL    

#-----------------------------------------------------------------------------
# Kuhn-Munkres MPA
#-----------------------------------------------------------------------------
def MPA(D,dt,dr,L,N,n_off,m_on,nL):
    Q=Q_ij(L,N,n_off,m_on,lnp_ij(dr,D,dt),nL)
    mpa_row,mpa_col=kuhn_munkres(-Q)
    return mpa_row,mpa_col

def MPA_energy(D,dt,dr,L,N,n_off,m_on,mpa_row,mpa_col,nL):
    Q=Q_ij(L,N,n_off,m_on,lnp_ij(dr,D,dt),nL)
    return -sum(Q[mpa_row,mpa_col])

def mMPAscore(D,dt,dr,L,mu_on,mu_off,N,M,nL):
    Delta=M-N
    scores_=[]
    for m_ in arange(max([0,Delta]),M,nL,dtype=int):
        mpa_row,mpa_col=MPA(D,dt,dr,L,N,m_-Delta,m_,nL)
        logp_m=log(p_m(m_,Delta,mu_on,mu_off))
        E_MPA=MPA_energy(D,dt,dr,L,N,m_-Delta,m_,mpa_row,mpa_col,nL)
        score=logp_m-E_MPA+lng(M+1-m_)-lng(M+1)-lng(N+1)
        scores_.append((score))
    return -max(scores_)
   
def MPA_minusLogLikelihood_multiframe(D,dt,drs,L,mu_off,mu_on,Ns,nL):
    mlnL=0
    for n,dr in enumerate(drs):
        mlnL += mMPAscore(D,dt,dr,L,mu_on,mu_off,Ns[n],Ns[n+1],nL)
    return mlnL

#-----------------------------------------------------------------------------
# Log-likelihood for the true assignment
#-----------------------------------------------------------------------------
def true_energy(D,dt,dr,rho):
    return sum( -log(rho) + log(4*np.pi*D*dt)+(dr[0]**2+dr[1]**2)/(4.*D*dt) )

def true_energy_multiframe(D,dt,drs,rho):
    return sum( [true_energy(D,dt,dr,rho) for dr in drs] ) 
