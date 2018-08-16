# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import ChainArray
from .base import *
from warnings import warn
from math import pi, log
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import OrderedDict


setup = {'name': ('dd', 'ddrift'),
        'arguments': OrderedDict((
                ('localization_error',  ('-e', dict(type=float, default=0.03, help='localization error'))),
                ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
                ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed'))))}


def dd_neg_posterior(x, dd, cell, squared_localization_error, jeffreys_prior, dt_mean, min_diffusivity):
        dd.update(x)
        D, drift = dd['D'], dd['drift']
        if D < min_diffusivity:
                warn(DiffusivityWarning(D, min_diffusivity))
        noise_dt = squared_localization_error
        n = len(cell) # number of translocations
        denominator = 4. * (D * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
        dr_minus_drift_dt = cell.dr - np.outer(cell.dt, drift)
        # non-directional squared displacement
        ndsd = np.sum(dr_minus_drift_dt * dr_minus_drift_dt, axis=1)
        neg_posterior = n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)
        if jeffreys_prior:
                neg_posterior += 2. * log(D * dt_mean + squared_localization_error)
        return neg_posterior


def infer_DD(cells, localization_error=0.03, jeffreys_prior=False, min_diffusivity=None, **kwargs):
        if isinstance(cells, Distributed): # multiple cells
                if min_diffusivity is None:
                        if jeffreys_prior:
                                min_diffusivity = 0.01
                        else:
                                min_diffusivity = 0
                elif min_diffusivity is False:
                        min_diffusivity = None
                args = (localization_error, jeffreys_prior, min_diffusivity)
                index, inferred = [], []
                for i in cells:
                        cell = cells[i]
                        index.append(i)
                        inferred.append(infer_DD(cell, *args, **kwargs))
                any_cell = cell
                inferred = pd.DataFrame(np.stack(inferred, axis=0), \
                        index=index, \
                        columns=[ 'diffusivity' ] + \
                                [ 'drift ' + col for col in any_cell.space_cols ])
                return inferred
        else: # single cell
                cell = cells
                if not bool(cell):
                        raise ValueError('empty cells')
                if not np.all(0 < cell.dt):
                        warn('translocation dts are non-positive', RuntimeWarning)
                        cell.dr[cell.dt < 0] *= -1.
                        cell.dt[cell.dt < 0] *= -1.
                dt_mean = np.mean(cell.dt)
                D_initial = np.mean(cell.dr * cell.dr) / (2. * dt_mean)
                initial_drift = np.zeros(cell.dim, dtype=D_initial.dtype)
                dd = ChainArray('D', D_initial, 'drift', initial_drift)
                if min_diffusivity is not None:
                        if 'bounds' in kwargs:
                                print(kwargs['bounds'])
                        kwargs['bounds'] = [(min_diffusivity, None)] + [(None, None)] * cell.dim
                #cell.cache = None # no cache needed
                sle = localization_error * localization_error # sle = squared localization error
                result = minimize(dd_neg_posterior, dd.combined, \
                        args=(dd, cell, sle, jeffreys_prior, dt_mean, min_diffusivity), \
                        **kwargs)
                #dd.update(result.x)
                #return (dd['D'], dd['drift'])
                return result.x # needless to split dd.combined into D and drift

