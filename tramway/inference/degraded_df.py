# -*- coding: utf-8 -*-

# Copyright © 2017-2018, Institut Pasteur
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


setup = {'name':    'degraded.df',
        'provides': 'df',
        'arguments': OrderedDict((
        ('localization_error',  ('-e', dict(type=float, help='localization precision (see sigma; default is 0.03)'))),
        ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
        ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')),
        ('debug',       dict(action='store_true')))),
        'cell_sampling':    'individual'}


def df_neg_posterior(x, df, cell, sigma2, jeffreys_prior, dt_mean, min_diffusivity):
    """
    Adapted from InferenceMAP's *dfPosterior* procedure:

    .. code-block:: c++

        for (int i = 0; i < ZONES[CURRENT_ZONE].translocations; i++) {
            const double dt = ZONES[CURRENT_ZONE].dt[i];
            const double dx = ZONES[CURRENT_ZONE].dx[i];
            const double dy = ZONES[CURRENT_ZONE].dy[i];

            D_bruit = LOCALIZATION_ERROR*LOCALIZATION_ERROR/dt;

            result += - log(4.0*PI*(FxFyD[2]+D_bruit)*dt ) - pow(fabs(dx-FxFyD[2]*FxFyD[0]*dt ),2.0)/(4.0*(FxFyD[2]+D_bruit)*dt) - pow(fabs(dy-FxFyD[2]*FxFyD[1]*dt ),2.0)/(4.0*(FxFyD[2]+D_bruit)*dt);
        }

        if (JEFFREYS_PRIOR == 1) {
            result += 2.0*log(FxFyD[2]) - 2.0*log(FxFyD[2]*ZONES[CURRENT_ZONE].dtMean + LOCALIZATION_ERROR*LOCALIZATION_ERROR);
        }
        return -result;

    """
    df.update(x)
    D, F = df['D'], df['F']
    if D < min_diffusivity:
        warn(DiffusivityWarning(D, min_diffusivity))
    noise_dt = sigma2
    n = len(cell) # number of translocations
    D_dt = D * cell.dt
    denominator = 4. * (D_dt + noise_dt) # 4*(D+Dnoise)*dt
    dr_minus_drift_dt = cell.dr - np.outer(D_dt, F)
    # non-directional squared displacement
    ndsd = np.sum(dr_minus_drift_dt * dr_minus_drift_dt, axis=1)
    neg_posterior = n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)
    if jeffreys_prior:
        try:
            neg_posterior += 2. * (log(D * dt_mean + sigma2) - log(D))
        except ValueError as e: # math domain error
            warn(DiffusivityWarning(e))
    return neg_posterior


def infer_DF(cells, localization_error=None, jeffreys_prior=False, min_diffusivity=None, debug=False, \
        **kwargs):
    if isinstance(cells, Distributed): # multiple cells
        localization_error = cells.get_localization_error(kwargs, 0.03, True, \
                localization_error=localization_error)
        args = (localization_error, jeffreys_prior, min_diffusivity)
        index, inferred = [], []
        for i in cells:
            cell = cells[i]
            # sanity checks
            if not bool(cell):
                raise ValueError('empty cells')
            if cell.dr.shape[1] == 0:
                raise ValueError('translocation array has no column')
            if cell.dt.shape[1:]:
                raise ValueError('time deltas are structured in multiple dimensions')
            # ensure that translocations are properly oriented in time
            if not np.all(0 < cell.dt):
                warn('translocation dts are non-positive', RuntimeWarning)
                cell.dr[cell.dt < 0] *= -1.
                cell.dt[cell.dt < 0] *= -1.
            index.append(i)
            inferred.append(infer_DF(cell, *args, **kwargs))
        inferred = np.stack(inferred, axis=0)
        #D = inferred[:,0]
        #gradD = []
        #for i in index:
        #       gradD.append(cells.grad(i, D))
        #gradD = np.stack(gradD, axis=0)
        inferred = pd.DataFrame(inferred, \
            index=index, \
            columns=[ 'diffusivity' ] + \
                [ 'force ' + col for col in cells.space_cols ])
        #for j, col in enumerate(cells.space_cols):
        #       inferred['gradD '+col] = gradD[:,j]
        if debug:
            xy = np.vstack([ cells[i].center for i in index ])
            inferred = inferred.join(pd.DataFrame(xy, index=index, \
                columns=cells.space_cols))
        return inferred
    else: # single cell
        cell = cells
        dt_mean = np.mean(cell.dt)
        D_initial = np.mean(cell.dr * cell.dr) / (2. * dt_mean)
        F_initial = np.zeros(cell.dim, dtype=D_initial.dtype)
        df = ChainArray('D', D_initial, 'F', F_initial)
        if min_diffusivity is not False:
            if min_diffusivity is None:
                noise_dt = localization_error
                min_diffusivity = (1e-16 - noise_dt) / np.max(cell.dt)
            kwargs['bounds'] = [(min_diffusivity, None)] + [(None, None)] * cell.dim
        #cell.cache = None # no cache needed
        result = minimize(df_neg_posterior, df.combined, \
            args=(df, cell, localization_error, jeffreys_prior, dt_mean, min_diffusivity), \
            **kwargs)
        #df.update(result.x)
        #return (df['D'], df['F'])
        return result.x # needless to split df.combined into D and F

