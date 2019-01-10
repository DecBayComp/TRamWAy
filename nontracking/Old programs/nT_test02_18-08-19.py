
from .base import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize


setup = {
    'infer': 'nT_test02', 
    'sampling': 'group' ,
    'arguments': {'cell_index': {}}
}

s2=0.00001

def nT_test02(cells, cell_index=0):
        #index, _, _, _, D0, _, _, _ = smooth_infer_init(cells)
        index = list(cells.keys())
        D0 = 0.
        i = cell_index
        print i
        res = minimize(minimize_me2, D0, args=(i, cells))
        D=res.x
        return pd.DataFrame(np.array(D)) #, index=index, columns=['D'])
    
def minimize_me2(D, i, cells):
        assert cells[i]
        cell = cells[i]
        r = cell.r
        t = cell.t
#        for j in cells.neighbours(i):
        print(r.size,t.size)
        print(t)
#            r.append(cells[j].r)
        r = np.vstack(r) 
        return np.sum(np.log(abs(D)+s2) + r*r/(abs(D)+s2) )