
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
    'infer': 'cell_neighbours', 
    'sampling': 'group' ,
    'arguments': {'cell_index': {}, 'show_N_t': {}, 'dt': {}, 'show_possible_links': {}}
}

#-----------------------------------------------------------------------------
# Main functions
#-----------------------------------------------------------------------------
def cell_neighbours(cells, cell_index=0, show_N_t=False, dt=0.04, show_possible_links=False):
    i = cell_index
    assert cells[i]
    cell = cells[i]
    r = cell.r
    t = cell.t
    frames = rt_to_frames(r,t,dt)
    N_t=[len(frame) for frame in frames]
    poss_links=sum([min([N_t[n],N_t[n+1]]) for n in range(len(N_t)-1)])
    print "Cell no.:", i, "; sum(N):", np.sum(N_t), ", possible links:", poss_links
    print "--- Neighbours: ---"
    for j in cells.neighbours(i):
        cell = cells[j]
        r = cell.r
        t = cell.t
        frames = rt_to_frames(r,t,dt)
        N_t=[len(frame) for frame in frames]
        poss_links=sum([min([N_t[n],N_t[n+1]]) for n in range(len(N_t)-1)])
        print "  ", j, "; sum(N):", np.sum(N_t), ", possible links:", poss_links
    if show_N_t:
        frames = rt_to_frames(r,t,dt)
        N_t=[len(frame) for frame in frames]
        print 'N_t =', N_t, np.sum(N_t)
    if show_possible_links:
        frames = rt_to_frames(r,t,dt)
        N_t=[len(frame) for frame in frames]
        poss_links=sum([min([N_t[n],N_t[n+1]]) for n in range(len(N_t)-1)])
        print 'Total number of possible links:', poss_links
    index=[i]
    D=0.
    return pd.DataFrame(np.array(D), index=index, columns=['D'])

def rt_to_frames(r,t,dt):
    times=np.arange(min(t),max(t+dt/100.),dt)
    frames=[]
    for t_ in times:
        frames.append(np.array(r[(t<t_+dt/100.)&(t>t_-dt/100.)]))
    return frames

