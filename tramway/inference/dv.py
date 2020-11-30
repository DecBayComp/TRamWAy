# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import ChainArray
from .base import *
from .gradient import *
from warnings import warn
from math import pi, log
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import OrderedDict
import time


setup = {'arguments': OrderedDict((
        ('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
        ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
        ('potential_prior',     ('-v', dict(type=float, help='prior on the potential'))),
        ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
        ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')),
        ('max_iter',        dict(type=int, help='maximum number of iterations (~100)')),
        ('compatibility',       ('-c', '--inferencemap', '--compatible',
                    dict(action='store_true', help='InferenceMAP compatible'))),
        ('rgrad',       dict(help="alternative gradient for the regularization; can be 'delta0' or 'delta1'")),
        ('export_centers',      dict(action='store_true')),
        ('verbose',         ()))),
    'cell_sampling': 'connected'}
setup_with_grad_arguments(setup)


class DV(ChainArray):
    """
    Convenience class for handling the D+V parameter vector and related priors.

    Objects of this class are not supposed to be serialized in .rwa files.
    """
    __slots__ = ('_diffusivity_prior', '_potential_prior', 'minimum_diffusivity', 'prior_include')

    def __init__(self, diffusivity, potential, diffusivity_prior=None, potential_prior=None, \
        minimum_diffusivity=None, positive_diffusivity=None, prior_include=None):
        # positive_diffusivity is for backward compatibility
        ChainArray.__init__(self, 'D', diffusivity, 'V', potential)
        self._diffusivity_prior = diffusivity_prior
        self._potential_prior = potential_prior
        self.minimum_diffusivity = minimum_diffusivity
        if minimum_diffusivity is None and positive_diffusivity is True:
            self.minimum_diffusivity = 0
        self.prior_include = prior_include

    @property
    def D(self):
        return self['D']

    @property
    def V(self):
        return self['V']

    @D.setter
    def D(self, diffusivity):
        self['D'] = diffusivity

    @V.setter
    def V(self, potential):
        self['V'] = potential

    def diffusivity_prior(self, j):
        if self._diffusivity_prior and (self.prior_include is None or self.prior_include[j]):
            return self._diffusivity_prior
        else:
            return None

    def potential_prior(self, j):
        if self._potential_prior and (self.prior_include is None or self.prior_include[j]):
            return self._potential_prior
        else:
            return None



def dv_neg_posterior(x, dv, cells, sigma2, jeffreys_prior, dt_mean, \
        index, reverse_index, grad_kwargs, y0, verbose, posteriors):
    """
    Adapted from InferenceMAP's *dvPosterior* procedure modified:

    .. code-block:: c++

        for (int a = 0; a < NUMBER_OF_ZONES; a++) {
            ZONES[a].gradVx = dvGradVx(DV,a);
            ZONES[a].gradVy = dvGradVy(DV,a);
            ZONES[a].gradDx = dvGradDx(DV,a);
            ZONES[a].gradDy = dvGradDy(DV,a);
            ZONES[a].priorActive = true;
        }


        for (int z = 0; z < NUMBER_OF_ZONES; z++) {
            const double gradVx = ZONES[z].gradVx;
            const double gradVy = ZONES[z].gradVy;
            const double gradDx = ZONES[z].gradDx;
            const double gradDy = ZONES[z].gradDy;

            const double D = DV[2*z];

            for (int j = 0; j < ZONES[z].translocations; j++) {
                const double dt = ZONES[z].dt[j];
                const double dx = ZONES[z].dx[j];
                const double dy = ZONES[z].dy[j];
                const double  Dnoise = LOCALIZATION_ERROR*LOCALIZATION_ERROR/dt;

                result += - log(4.0*PI*(D + Dnoise)*dt) - ((dx-D*gradVx*dt)*(dx-D*gradVx*dt) + (dy-D*gradVy*dt)*(dy-D*gradVy*dt))/(4.0*(D+Dnoise)*dt);
            }

            if (ZONES[z].priorActive == true) {
                result -= V_PRIOR*(gradVx*gradVx*ZONES[z].areaX + gradVy*gradVy*ZONES[z].areaY);
                result -= D_PRIOR*(gradDx*gradDx*ZONES[z].areaX + gradDy*gradDy*ZONES[z].areaY);
                if (JEFFREYS_PRIOR == 1) {
                    result += 2.0*log(D*1.00) - 2.0*log(D*ZONES[z].dtMean + LOCALIZATION_ERROR*LOCALIZATION_ERROR);
            }
        }

    with ``dx-D*gradVx*dt`` and ``dy-D*gradVy*dt`` modified as ``dx+D*gradVx*dt`` and ``dy+D*gradVy*dt`` respectively.

    See also :ref:`DV <inference_dv>`.
    """
    if verbose:
        t = time.time()

    # extract `D` and `V`
    dv.update(x)
    D = dv.D
    V = dv.V
    #

    if dv.minimum_diffusivity is not None:
        observed_min = np.min(D)
        if observed_min < dv.minimum_diffusivity and not \
                np.isclose(observed_min, dv.minimum_diffusivity):
            warn(DiffusivityWarning(observed_min, dv.minimum_diffusivity))
    noise_dt = sigma2

    # for all cell
    raw_posterior = priors = 0.
    for j, i in enumerate(index):
        cell = cells[i]
        n = len(cell) # number of translocations

        # spatial gradient of the local potential energy
        gradV = cells.grad(i, V, reverse_index, **grad_kwargs)
        #print('{}\t{}\t{}\t{}\t{}\t{}'.format(i+1,D[j], V[j], -gradV[0], -gradV[1], result))
        #print('{}\t{}\t{}'.format(i+1, *gradV))
        if gradV is None:
            continue

        # various posterior terms
        #print(cell.dt)
        D_dt = D[j] * cell.dt
        denominator = 4. * (D_dt + noise_dt)
        dr_minus_drift = cell.dr + np.outer(D_dt, gradV)
        # non-directional squared displacement
        ndsd = np.sum(dr_minus_drift * dr_minus_drift, axis=1)
        res = n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)
        if np.isnan(res):
            #print('isnan')
            continue
        raw_posterior += res

        # priors
        potential_prior = dv.potential_prior(j)
        if potential_prior:
            priors += potential_prior * cells.grad_sum(i, gradV * gradV, reverse_index)
        diffusivity_prior = dv.diffusivity_prior(j)
        if diffusivity_prior:
            # spatial gradient of the local diffusivity
            gradD = cells.grad(i, D, reverse_index, **grad_kwargs)
            if gradD is not None:
                # `grad_sum` memoizes and can be called several times at no extra cost
                priors += diffusivity_prior * cells.grad_sum(i, gradD * gradD, reverse_index)
        #print('{}\t{}\t{}'.format(i+1, D[j], result))
    if jeffreys_prior:
        priors += 2. * np.sum(np.log(D * dt_mean + sigma2) - np.log(D))

    result = raw_posterior + priors
    posteriors.append([raw_posterior, result])

    if verbose:
        print('objective: {}\t time: {}ms'.format(result, int(round((time.time() - t) * 1e3))))

    return result - y0


def dv_neg_posterior1(x, dv, cells, sigma2, jeffreys_prior, dt_mean, \
        index, reverse_index, grad_kwargs, y0, verbose, posteriors):
    """
    Similar to :func:`dv_neg_posterior`.
    The smoothing priors feature an alternative spatial "gradient" implemented using
    :meth:`~tramway.inference.base.Distributed.local_variation` instead of
    :meth:`~tramway.inference.base.Distributed.grad`.
    """
    if verbose:
        t = time.time()

    # extract `D` and `V`
    dv.update(x)
    D = dv.D
    V = dv.V
    #

    if dv.minimum_diffusivity is not None:
        observed_min = np.min(D)
        if observed_min < dv.minimum_diffusivity and \
                not np.isclose(observed_min, dv.minimum_diffusivity):
            warn(DiffusivityWarning(observed_min, dv.minimum_diffusivity))
    noise_dt = sigma2

    # for all cell
    raw_posterior = priors = 0.
    for j, i in enumerate(index):
        cell = cells[i]
        n = len(cell) # number of translocations

        # spatial gradient of the local potential energy
        gradV = cells.grad(i, V, reverse_index, **grad_kwargs)
        #print('{}\t{}\t{}\t{}\t{}\t{}'.format(i+1,D[j], V[j], -gradV[0], -gradV[1], result))
        #print('{}\t{}\t{}'.format(i+1, *gradV))
        if gradV is None:
            continue

        # various posterior terms
        #print(cell.dt)
        D_dt = D[j] * cell.dt
        denominator = 4. * (D_dt + noise_dt)
        dr_minus_drift = cell.dr + np.outer(D_dt, gradV)
        # non-directional squared displacement
        ndsd = np.sum(dr_minus_drift * dr_minus_drift, axis=1)
        res = n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)
        if np.isnan(res):
            #print('isnan')
            continue
        raw_posterior += res

        # priors
        potential_prior = dv.potential_prior(j)
        if potential_prior:
            # spatial variation of the local potential
            deltaV = cells.local_variation(i, V, reverse_index, **grad_kwargs)
            if deltaV is not None:
                priors += potential_prior * cells.grad_sum(i, deltaV * deltaV, reverse_index)
        diffusivity_prior = dv.diffusivity_prior(j)
        if diffusivity_prior:
            # spatial variation of the local diffusivity
            deltaD = cells.local_variation(i, D, reverse_index, **grad_kwargs)
            if deltaD is not None:
                # `grad_sum` memoizes and can be called several times at no extra cost
                priors += diffusivity_prior * cells.grad_sum(i, deltaD * deltaD, reverse_index)
        #print('{}\t{}\t{}'.format(i+1, D[j], result))
    if jeffreys_prior:
        priors += 2. * np.sum(np.log(D * dt_mean + sigma2) - np.log(D))

    result = raw_posterior + priors
    if posteriors is not None:
        posteriors.append([raw_posterior, result])

    if verbose:
        print('objective: {}\t time: {}ms'.format(result, int(round((time.time() - t) * 1e3))))

    return result - y0


def inferDV(cells, diffusivity_prior=None, potential_prior=None, \
    jeffreys_prior=False, min_diffusivity=None, max_iter=None, epsilon=None, \
    export_centers=False, verbose=True, compatibility=False, \
    D0=None, V0=None, rgrad=None, **kwargs):

    localization_error = cells.get_localization_error(kwargs, 0.03, True)

    # initial values
    index, reverse_index, n, dt_mean, D_initial, min_diffusivity, D_bounds, border = \
        smooth_infer_init(cells, min_diffusivity=min_diffusivity, jeffreys_prior=jeffreys_prior,
        sigma2=localization_error)
    # V initial values
    if V0 is None:
        try:
            if compatibility:
                raise Exception # skip to the except block
            volume = [ cells[i].volume for i in index ]
        except:
            V_initial = -np.log(n / np.max(n))
        else:
            density = n / np.array([ np.inf if v is None else v for v in volume ])
            density[density == 0] = np.min(density[0 < density])
            V_initial = np.log(np.max(density)) - np.log(density)
    else:
        if np.isscalar(V0):
            V_initial = np.full(D_initial.size, V0)
        elif V0.size == D_initial.size:
            V_initial = V0
        else:
            raise ValueError('wrong size for V0')
    if D0 is not None:
        if np.isscalar(D0):
            D_initial[...] = D0
        elif D0.size == D_initial.size:
            D_initial = D0
        else:
            raise ValueError('wrong size for D0')

    dv = DV(D_initial, V_initial, diffusivity_prior, potential_prior, min_diffusivity)
    posteriors = []

    # gradient options
    grad_kwargs = get_grad_kwargs(kwargs, epsilon=epsilon, compatibility=compatibility)

    # parametrize the optimization algorithm
    default_lBFGSb_options = dict(maxiter=1e3, maxfun=1e10, maxcor=dv.combined.size, ftol=1e-8)
    # in L-BFGS-B the number of iterations is usually very low (~10-100) while the number of
    # function evaluations is much higher (~1e4-1e5);
    # with maxfun defined, an iteration can stop anytime and the optimization may terminate
    # with an error message
    if min_diffusivity is None:
        bounds = None
        options = {}
    else:
        V_bounds = [(None, None)] * V_initial.size
        bounds = D_bounds + V_bounds
        options = dict(default_lBFGSb_options)
    options.update(kwargs.pop('options', {}))
    options.update(**kwargs) # for backward compatibility
    if max_iter:
        options['maxiter'] = max_iter
    if verbose:
        options['disp'] = verbose
    if options:
        _kwargs = dict(options = options)
    else:
        _kwargs = {}

    # posterior function
    if rgrad in ('delta','delta0','delta1'):
        fun = dv_neg_posterior1
    else:
        if rgrad not in (None, 'grad', 'grad1', 'gradn'):
            warn('unsupported rgrad: {}'.format(rgrad), RuntimeWarning)
        fun = dv_neg_posterior

    # posterior function input arguments
    args = (dv, cells, localization_error, jeffreys_prior, dt_mean,
            index, reverse_index, grad_kwargs)

    # get the initial posterior value so that it is subtracted from the further evaluations
    y0 = fun(dv.combined, *(args + (0., False, [])))
    if verbose:
        print('At X0\tactual posterior= {}\n'.format(y0))
    #y0 = 0.
    args = args + (y0, 1 < int(verbose), posteriors)

    # run the optimization routine
    result = minimize(fun, dv.combined, args=args, bounds=bounds, **_kwargs)
    if not (result.success or verbose):
        warn('{}'.format(result.message), OptimizationWarning)

    y = np.array(result.x)

    # collect the result
    dv.update(y)
    D, V = dv.D, dv.V
    if np.any(V < 0):
        V -= np.min(V)
    DVF = pd.DataFrame(np.stack((D, V), axis=1), index=index, \
        columns=[ 'diffusivity', 'potential'])

    # derivate the forces
    index_, F = [], []
    for i in index:
        gradV = cells.grad(i, V, reverse_index, **grad_kwargs)
        if gradV is not None:
            index_.append(i)
            F.append(-gradV)
    if F:
        F = pd.DataFrame(np.stack(F, axis=0), index=index_, \
            columns=[ 'force ' + col for col in cells.space_cols ])
    else:
        warn('not any cell is suitable for evaluating the local force', RuntimeWarning)
        F = pd.DataFrame(np.zeros((0, len(cells.space_cols)), dtype=V.dtype), \
            columns=[ 'force ' + col for col in cells.space_cols ])
    DVF = DVF.join(F)

    # add extra information if required
    if export_centers:
        xy = np.vstack([ cells[i].center for i in index ])
        DVF = DVF.join(pd.DataFrame(xy, index=index, \
            columns=cells.space_cols))
        #DVF.to_csv('results.csv', sep='\t')

    # format the posteriors
    posteriors = pd.DataFrame(np.array(posteriors), columns=['fit', 'total'])

    return DVF, posteriors

