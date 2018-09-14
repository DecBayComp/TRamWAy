
from .base import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize


setup = {
    'infer': 'nontracking_1zone', 
    'sampling': 'group' ,
    'arguments': {'cell_index': {}}
}

s2=0.0001

def nontracking_1zone(cells, cell_index=0):
        #index, _, _, _, D0, _, _, _ = smooth_infer_init(cells)
        index = list(cells.keys())
        D0 = 0.
        i = cell_index
        print i
        D=estimate_D_1zone(D, i, cells)
        return pd.DataFrame(np.array(D)) #, index=index, columns=['D'])
    
def rt_to_frames(r,t,dt):
    times=arange(min(t),max(t)+dt/100.,dt)
    frames=[]
    for time in times:
        frames.append(r[t==[time]]
    return frames

def estimate_D_1zone(D, i, cells):
        assert cells[i]
        cell = cells[i]
        r = cell.r
        t = cell.t
        dt = cell.dt
        frames = tr_to_frames(r,t,dt)
        print(frames)
        fit = 0.
        D_i=fit.x
        return D_i
