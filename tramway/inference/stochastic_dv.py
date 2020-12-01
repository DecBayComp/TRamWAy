# -*- coding: utf-8 -*-

# Copyright © 2018-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
from .gradient import *
from .dv import DV
from .optimization import *
from tramway.core import parallel
from math import pi, log
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from collections import OrderedDict, deque
import time
from scipy.stats import trim_mean
import logging
import os


setup = {'name': ('stochastic.dv', 'stochastic.dv1'),
    'provides': 'dv',
    'infer': 'infer_stochastic_DV',
    'arguments': OrderedDict((
        ('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
        ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
        ('potential_prior',     ('-v', dict(type=float, help='prior on the potential energy'))),
        ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
        ('time_prior',          ('-t', dict(type=float, help='prior on the temporal variations of the diffusivity and potential energy (multiplies with `diffusivity_prior` and `potential_prior`'))),
        ('diffusivity_time_prior',('--time-d', dict(type=float, help='prior on the temporal variations of the diffusivity'))),
        ('potential_time_prior',('--time-v', dict(type=float, help='prior on the temporal variations of the potential'))),
        ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')),
        ('max_iter',            dict(type=int, help='maximum number of iterations')),
        ('compatibility',       ('-c', '--inferencemap', '--compatible',
                                dict(action='store_true', help='InferenceMAP compatible'))),
        ('gradient',            ('--grad', dict(help="spatial gradient implementation; any of 'grad1', 'gradn'"))),
        ('grad_epsilon',        dict(args=('--eps', '--epsilon'), kwargs=dict(type=float, help='if defined, every spatial gradient component can recruit all of the neighbours, minus those at a projected distance less than this value'), translate=True)),
        ('grad_selection_angle',('-a', dict(type=float, help='top angle of the selection hypercone for neighbours in the spatial gradient calculation (1= pi radians; if not -c, default is: {})'.format(default_selection_angle)))),
        ('rgrad',               dict(help="local spatial variation; any of 'delta0' (highly recommended), 'delta1'")),
        ('export_centers',      dict(action='store_true')),
        ('verbose',             ()))),
        #('region_size',         ('-s', dict(type=int, help='radius of the regions, in number of adjacency steps'))))),
    'cell_sampling': 'group'}


module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter('%(message)s'))
module_logger.addHandler(_console)


class LocalDV(DV):
    __slots__ = ('regions','_diffusivity_time_prior','_potential_time_prior',
            'prior_delay','_n_calls','_undefined_grad','_undefined_time_derivative',
            '_update_undefined_grad','_update_undefined_time_derivative','_logger')

    def __init__(self, diffusivity, potential,
        diffusivity_spatial_prior=None, potential_spatial_prior=None,
        diffusivity_time_prior=None, potential_time_prior=None,
        minimum_diffusivity=None, positive_diffusivity=None, prior_include=None,
        regions=None, prior_delay=None, logger=True):
        # positive_diffusivity is for backward compatibility
        DV.__init__(self, diffusivity, potential, diffusivity_spatial_prior, potential_spatial_prior,
            minimum_diffusivity, positive_diffusivity, prior_include)
        self.regions = regions
        self._diffusivity_time_prior = diffusivity_time_prior
        self._potential_time_prior = potential_time_prior
        self.prior_delay = prior_delay
        self._n_calls = 0.
        self._undefined_grad = set()
        self._undefined_time_derivative = set()
        self._update_undefined_grad = set()
        self._update_undefined_time_derivative = set()
        self._logger = logger

    @property
    def verbose(self):
        return self._logger is not None

    @verbose.setter
    def verbose(self, b):
        if b:
            if not self.verbose:
                self._logger = True
        else:
            if self.verbose:
                self._logger = None

    def region(self, i):
        return self.regions[i]

    def indices(self, cell_ids):
        if isinstance(cell_ids, (int, np.int_)):
            return np.array([ cell_ids, int(self.combined.size / 2) + cell_ids ])
        cell_ids = np.array(cell_ids)
        cell_ids.sort()
        return np.concatenate((cell_ids, int(self.combined.size / 2) + cell_ids))
    def diffusivity_indices(self, cell_ids):
        if isinstance(cell_ids, (int, np.int_)):
            cell_ids = [cell_ids]
        return np.array(cell_ids)
    def potential_indices(self, cell_ids):
        if isinstance(cell_ids, (int, np.int_)):
            cell_ids = [cell_ids]
        return int(self.combined.size / 2) + np.array(cell_ids)

    def potential_prior(self, i):
        if self.prior_delay:
            if self._n_calls < self.prior_delay:
                prior = None
            else:
                prior = DV.potential_prior(self, i)
            if self._diffusivity_prior:
                self._n_calls += .5
            else:
                self._n_calls += 1.
        else:
                prior = DV.potential_prior(self, i)
        return prior

    def potential_spatial_prior(self, i):
        return self.potential_prior(i)

    def potential_time_prior(self, i):
        return self._potential_time_prior

    def diffusivity_prior(self, i):
        if self.prior_delay:
            if self._n_calls < self.prior_delay:
                prior = None
            else:
                prior = DV.diffusivity_prior(self, i)
            if self._potential_prior:
                self._n_calls += .5
            else:
                self._n_calls += 1.
        else:
                prior = DV.diffusivity_prior(self, i)
        return prior

    def diffusivity_spatial_prior(self, i):
        return self.diffusivity_prior(i)

    def diffusivity_time_prior(self, i):
        return self._diffusivity_time_prior

    @property
    def logger(self):
        if self._logger is True:
            self._logger = module_logger
        return self._logger

    @logger.setter
    def logger(self, l):
        self._logger = l

    def undefined_grad(self, i, feature=''):
        if self.verbose and i not in self._undefined_grad:
            self._undefined_grad.add(i)
            self._update_undefined_grad.add(i)
            self.logger.debug('grad{}({}) is not defined'.format(feature, i))

    def undefined_time_derivative(self, i, feature=''):
        if self.verbose and i not in self._undefined_time_derivative:
            self._undefined_time_derivative.add(i)
            self._update_undefined_time_derivative.add(i)
            self.logger.debug('d{}({})/dt failed'.format(feature, i))

    def pop_workspace_update(self):
        try:
            return self._update_undefined_grad, self._update_undefined_time_derivative
        finally:
            self._update_undefined_grad, self._update_undefined_time_derivative = set(), set()

    def push_workspace_update(self, update):
        undefined_grad, undefined_time_derivative = update
        if isinstance(undefined_grad, set):
            self._undefined_grad.update(undefined_grad)
            self._undefined_time_derivative.update(undefined_time_derivative)

parallel.abc.WorkspaceExtension.register(LocalDV)


def make_regions(cells, index, reverse_index, size=1):
    A = cells.adjacency
    regions = []
    for i in index:
        if isinstance(i, int):
            j = set([i])
        else:
            j = set([i.tolist()])
        j_inner = set()
        for k in range(size):
            j_outer = j - j_inner
            j_inner = j
            for l in j_outer:
                neighbours = A.indices[A.indptr[l]:A.indptr[l+1]] # NOT cells.neighbours(l)
                j |= set(neighbours.tolist())
        j = reverse_index[list(j)]
        regions.append(j)
    return regions


def lookup_space_cells(cells):
    A = cells.temporal_adjacency
    available = { i for i in cells }
    space_cells = []
    while available:
        space_cell = set()
        front_cells = { available.pop() }
        while front_cells:
            more_cells = set()
            for i in front_cells:
                more_cells |= set(A.indices[A.indptr[i]:A.indptr[i+1]])
            space_cell |= front_cells
            if more_cells:
                front_cells = more_cells - space_cell
            else:
                break
        available -= space_cell
        space_cells.append(list(space_cell))
    return space_cells


def local_dv_neg_posterior(j, x, dv, cells, sigma2, jeffreys_prior,
    dt_mean, index, reverse_index, grad_kwargs,
    posterior_info=None, iter_num=None, verbose=False):
    """
    """

    # extract `D` and `V`
    #dv.update(x)
    #D = dv.D # slow(?)
    #V = dv.V
    #Dj = D[j]
    Dj = x[j]
    if np.any(np.isnan(Dj)):
        raise ValueError('D is nan')
    V = x[int(x.size/2):]
    #

    noise_dt = sigma2

    # for all cell
    i = index[j]
    cell = cells[i]
    n = len(cell) # number of translocations

    # spatial gradient of the local potential energy
    gradV = cells.grad(i, V, reverse_index, **grad_kwargs)
    #print('{}\t{}\t{}\t{}\t{}\t{}'.format(i+1,D[j], V[j], -gradV[0], -gradV[1], result))
    #print('{}\t{}\t{}'.format(i+1, *gradV))
    if gradV is None or np.any(np.isnan(gradV)):
        dv.undefined_grad(i, 'V')
        gradV = np.zeros(cell.dim)

    # various posterior terms
    #print(cell.dt)
    D_dt = Dj * cell.dt
    denominator = 4. * (D_dt + noise_dt)
    if np.any(denominator <= 0):
        raise ValueError('undefined posterior; local diffusion value: %s', Dj)
    dr_minus_drift = cell.dr + np.outer(D_dt, gradV)
    # non-directional squared displacement
    ndsd = np.sum(dr_minus_drift * dr_minus_drift, axis=1)
    raw_posterior = n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)

    if np.isnan(raw_posterior):
        raise ValueError('undefined posterior; local diffusion value: %s', Dj)

    # priors
    standard_priors, time_priors = 0., 0.
    V_prior = dv.potential_spatial_prior(j)
    if V_prior:
        deltaV = cells.local_variation(i, V, reverse_index, **grad_kwargs)
        if deltaV is not None:
            standard_priors += V_prior * cells.grad_sum(i, deltaV * deltaV, reverse_index)
    D_prior = dv.diffusivity_spatial_prior(j)
    if D_prior:
        D = x[:int(x.size/2)]
        # spatial gradient of the local diffusivity
        deltaD = cells.local_variation(i, D, reverse_index, **grad_kwargs)
        if deltaD is not None:
            # `grad_sum` memoizes and can be called several times at no extra cost
            standard_priors += D_prior * cells.grad_sum(i, deltaD * deltaD, reverse_index)
    #print('{}\t{}\t{}'.format(i+1, D[j], result))
    if jeffreys_prior:
        if Dj <= 0:
            raise ValueError('non positive diffusivity')
        standard_priors += jeffreys_prior * 2. * np.log(Dj * dt_mean[j] + sigma2) - np.log(Dj)

    D_time_prior = dv.diffusivity_time_prior(i)
    if D_time_prior:
        if not D_prior:
            D = x[:int(x.size/2)]
        dDdt = cells.temporal_variation(i, D, reverse_index)
        if dDdt is None:
            dv.undefined_time_derivative(i, 'D')
        else:
            # assume fixed-duration time window
            time_priors += D_time_prior * np.sum(dDdt * dDdt)
    V_time_prior = dv.potential_time_prior(i)
    if V_time_prior:
        dVdt = cells.temporal_variation(i, V, reverse_index)
        if dVdt is None:
            dv.undefined_time_derivative(i, 'V')
        else:
            time_priors += V_time_prior * np.sum(dVdt * dVdt)

    priors = standard_priors + time_priors
    result = raw_posterior + priors
    #if 1e6 < np.max(np.abs(gradV)):
    #    print((i, raw_posterior, standard_priors, Dj, V[j], n, gradV, [V[_j] for _j in cells.neighbours(i)]))

    if verbose:
        dv.logger.debug((i, raw_posterior, standard_priors, time_priors))
    if posterior_info is not None:
        if iter_num is None:
            info = [i, raw_posterior, result]
        else:
            info = [iter_num, i, raw_posterior, result]
        posterior_info.append(info)

    return result

def _local_dv_neg_posterior(*args, **kwargs):
    try:
        return local_dv_neg_posterior(*args, **kwargs)
    except ValueError:
        return np.inf


def infer_stochastic_DV(cells,
    diffusivity_spatial_prior=None, potential_spatial_prior=None,
    diffusivity_time_prior=None, potential_time_prior=None,
    jeffreys_prior=False, min_diffusivity=None, max_iter=None,
    compatibility=False,
    export_centers=False, verbose=True, superlocal=True, stochastic=True,
    D0=None, V0=None, x0=None, rgrad=None, debug=False, fulltime=False,
    diffusion_prior=None, diffusion_spatial_prior=None, diffusion_time_prior=None,
    prior_delay=None, return_struct=False, posterior_max_count=None,# deprecated
    diffusivity_prior=None, potential_prior=None, time_prior=None,
    allow_negative_potential=False,
    **kwargs):
    """
    Arguments:

        ...

        time_prior (float or tuple): one of two regularization coefficients that
            penalize the temporal derivative of diffusivity and potential energy;
            the first coefficient applies to diffusivity (and is multiplied by
            `diffusivity_prior`) and the second applies to potential energy (and
            is multiplied by `potential_prior`).

        diffusivity_spatial_prior/diffusion_spatial_prior: alias for `diffusivity_prior`.

        diffusivity_time_prior/diffusion_time_prior: penalizes the temporal derivative
            of diffusivity; this coefficient does NOT multiply with `diffusivity_prior`.

        potential_spatial_prior: alias for `potential_prior`.

        potential_time_prior: penalizes the temporal derivative of potential;
            this coefficient does NOT multiply with `potential_prior`.

        stochastic (bool): if ``False``, emulate standard *DV* with sparse gradient.

        D0 (float or ndarray): initial diffusivity value(s).

        V0 (float or ndarray): initial potential energy value(s).

        x0 (ndarray): initial parameter vector [diffusivities, potentials].

        rgrad (str): either 'grad'/'grad1', 'gradn' (none recommended), 'delta'/'delta0'
            or 'delta1';
            see the corresponding functions in module :mod:`~tramway.inference.gradient`.

        debug (bool or sequence of str): any subset of {'ncalls', 'f_history',
            'df_history', 'projg_history', 'error', 'diagnoses'}.

        xref (ndarray): reference parameter vector [diffusivities, potentials];
            required to compute the 'error' debug variable.

        diagnosis (callable): function of the iteration number, current component and
            candidate updated component;
            required to compute the 'diagnoses' debug variable.

        prior_delay/return_struct/posterior_max_count: all deprecated.

        allow_negative_potential (bool): do not offset the whole potential map towards
            all-positive values.

        ...

    See also :func:`~tramway.inference.optimization.minimize_sparse_bfgs`.
    """

    # initial values
    if stochastic and not (superlocal or potential_prior or potential_spatial_prior):
        raise ValueError('spatial regularization is required for the potential energy')
    localization_error = cells.get_localization_error(kwargs, 0.03, True)
    index, reverse_index, n, dt_mean, D_initial, _min_diffusivity, D_bounds, border = \
        smooth_infer_init(cells, min_diffusivity=min_diffusivity, jeffreys_prior=jeffreys_prior,
        sigma2=localization_error)
    # V initial values
    if x0 is None:
        if V0 is None:
            try:
                if compatibility:
                    raise Exception # skip to the except block
                volume = [ cells[i].volume for i in index ]
            except:
                V_initial = -np.log(n / np.max(n))
            else:
                density = n / np.array([ np.inf if (v is None or np.isnan(v)) else v for v in volume ])
                try:
                    density[density == 0] = np.min(density[0 < density])
                except ValueError:
                    raise ValueError('no data in bounded domains; null density everywhere')
                V_initial = np.log(np.max(density)) - np.log(density)
    else:
        #warn('`x0` is deprecated; please use `D0` and `V0` instead', DeprecationWarning)
        if x0.size != 2 * D_initial.size:
            raise ValueError('wrong size for x0')
        D_initial, V_initial = x0[:int(x0.size/2)], x0[int(x0.size/2):]
    if D0 is not None:
        if np.isscalar(D0):
            D_initial[...] = D0
        elif D0.size == D_initial.size:
            D_initial = D0
        else:
            raise ValueError('wrong size for D0')
    if V0 is not None:
        if np.isscalar(V0):
            # V0 may be given as an integer
            V_initial = np.full(D_initial.size, V0, dtype=D_initial.dtype)
        elif V0.size == D_initial.size:
            V_initial = V0
        else:
            raise ValueError('wrong size for V0')

    if diffusivity_prior is None:
        diffusivity_prior = diffusion_prior
    if diffusivity_spatial_prior is None:
        if diffusion_spatial_prior is None:
            diffusivity_spatial_prior = diffusivity_prior
        else:
            diffusivity_spatial_prior = diffusion_spatial_prior
    if potential_spatial_prior is None:
        potential_spatial_prior = potential_prior
    if not isinstance(time_prior, (tuple, list)):
        time_prior = (time_prior, time_prior)
    if diffusivity_time_prior is None:
        diffusivity_time_prior = diffusion_time_prior
    if diffusivity_time_prior is None and not (time_prior[0] is None or diffusivity_spatial_prior is None):
        diffusivity_time_prior = time_prior[0] * diffusivity_spatial_prior
    if potential_time_prior is None and not (time_prior[1] is None or potential_spatial_prior is None):
        potential_time_prior = time_prior[1] * potential_spatial_prior
    dv = LocalDV(D_initial, V_initial, diffusivity_spatial_prior, potential_spatial_prior,
        diffusivity_time_prior, potential_time_prior, min_diffusivity, prior_delay=prior_delay)
    if verbose:
        logger = dv.logger
    else:
        dv.logger = None
        logger = module_logger
    if posterior_max_count:
        warn('`posterior_max_count` is deprecated', RuntimeWarning)
    posterior_info = None # parallelization makes this inoperant

    # gradient options
    grad_kwargs = get_grad_kwargs(kwargs)
    # bounds
    if np.isscalar(V0) and V0 == 0:
        V_bounds = [(None, None)] * V_initial.size
    else:
        V_bounds = [(0., None)] * V_initial.size
    #if min_diffusivity is not None:
    #    assert np.all(min_diffusivity < D_initial)
    bounds = D_bounds + V_bounds

    # posterior function input arguments
    if jeffreys_prior is True:
        jeffreys_prior = 1.
    args = (dv, cells, localization_error, jeffreys_prior, dt_mean,
        index, reverse_index, grad_kwargs, posterior_info)

    m = len(index)
    if not stochastic:
        y0 = sum( local_dv_neg_posterior(j, dv.combined, *args[:-1]) for j in range(m) )
        if verbose:
            dv.logger.info('At X0\tactual posterior= {}'.format(y0))
        #args = args + (y0,)

    # keyword arguments to `minimize_sparse_bfgs`
    sbfgs_kwargs = dict(kwargs)

    # cell groups (a given cell + its neighbours)
    dv.regions = make_regions(cells, index, reverse_index)

    # covariates and subspaces
    if stochastic:
        component = m
        covariate = dv.region
        #covariate = lambda i: np.unique(np.concatenate([ dv.region(_i) for _i in dv.region(i) ]))
        descent_subspace = None
        if fulltime:
            space_cells = lookup_space_cells(cells)
            component = len(space_cells)
            def gradient_subspace(i):
                return space_cells[i]
        elif superlocal:
            gradient_subspace = dv.indices
        else:
            #def gradient_subspace(i):
            #    return np.r_[dv.diffusivity_indices(i), dv.potential_indices(dv.region(i))]
            def gradient_subspace(i):
                return dv.indices(dv.region(i))
        if 'independent_components' not in sbfgs_kwargs:
            sbfgs_kwargs['independent_components'] = True
        if 'memory' not in sbfgs_kwargs:
            sbfgs_kwargs['memory'] = None
    else:
        if superlocal:
            superlocal = False
            #raise ValueError('`stochastic=False` and `superlocal=True` are incompatible')
        component = lambda k: 0
        covariate = lambda i: range(m)
        descent_subspace = None
        gradient_subspace = None
        def col2rows(j):
            i = j % m
            return dv.region(i)
        if 'gradient_covariate' not in sbfgs_kwargs:
            sbfgs_kwargs['gradient_covariate'] = col2rows
        sbfgs_kwargs['worker_count'] = 0

    if os.name == 'nt':
        if sbfgs_kwargs.get('worker_count', None):
            logger.warning('multiprocessing may break on Windows')
        else:
            sbfgs_kwargs['worker_count'] = 0

    # other arguments
    if verbose:
        sbfgs_kwargs['verbose'] = verbose
        if 1 < verbose:
            args = args + (None, True)
    if max_iter:
        sbfgs_kwargs['max_iter'] = max_iter
    if bounds is not None:
        sbfgs_kwargs['bounds'] = bounds
    if 'eps' not in sbfgs_kwargs:
        sbfgs_kwargs['eps'] = 1. # beware: not the `tramway.inference.grad` option
    sbfgs_kwargs['newton'] = newton = sbfgs_kwargs.get('newton', True)
    if newton:
        default_step_max = 2.
        default_wolfe = (.5, None)
    else:
        default_step_max = .5
        default_wolfe = (.1, None)
    if True:#stochastic:
        sbfgs_kwargs['ls_wolfe'] = sbfgs_kwargs.get('ls_wolfe', default_wolfe)
        sbfgs_kwargs['ls_step_max'] = sbfgs_kwargs.get('ls_step_max', default_step_max)
    if 'ls_armijo_max' not in sbfgs_kwargs:
        sbfgs_kwargs['ls_armijo_max'] = 5
    #ls_step_max_decay = sbfgs_kwargs.get('ls_step_max_decay', None)
    #if ls_step_max_decay:
    #    sbfgs_kwargs['ls_step_max_decay'] /= float(m)
    if 'ftol' not in sbfgs_kwargs:
        sbfgs_kwargs['ftol'] = 1e-5
    if 'gtol' not in sbfgs_kwargs:
        sbfgs_kwargs['gtol'] = None
    if debug:
        debug_all = {'ncalls': 'ncalls',
                'f_history': 'f',
                'df_history': 'df',
                'projg_history': 'projg',
                'error': 'err',
                'diagnoses': 'diagnosis'}
        if debug is True:
            sbfgs_kwargs['returns'] = 'all'
        else:
            if isinstance(debug, str):
                debug = (debug,)
            sbfgs_kwargs['returns'] = { debug_all[attr] for attr in debug }
    else:
        sbfgs_kwargs['returns'] = set()

    assert not np.any(np.isnan(dv.combined))
    assert not np.any(np.isinf(dv.combined))

    # run the optimization routine
    result = minimize_sparse_bfgs(local_dv_neg_posterior, dv.combined, component, covariate,
            gradient_subspace, descent_subspace, args, **sbfgs_kwargs)
    #if not (result.success or verbose):
    #    warn('{}'.format(result.message), OptimizationWarning)

    # extract the different types of parameters
    dv.update(result.x)
    D, V = dv.D, dv.V
    if np.isscalar(V0) and V0 == 0 and not allow_negative_potential:
        Vmin = np.nanmin(V)
        if Vmin < 0:
            V -= Vmin
    DVF = pd.DataFrame(np.stack((D, V), axis=1), index=index,
        columns=['diffusivity', 'potential'])

    # derivate the forces
    index_, F = [], []
    for i in index:
        gradV = cells.grad(i, V, reverse_index, **grad_kwargs)
        if gradV is not None:
            index_.append(i)
            F.append(-gradV)
    if F:
        F = pd.DataFrame(np.stack(F, axis=0), index=index_,
            columns=[ 'force ' + col for col in cells.space_cols ])
    else:
        dv.logger.warning('not any cell is suitable for evaluating the local force')
        F = pd.DataFrame(np.zeros((0, len(cells.space_cols)), dtype=V.dtype),
            columns=[ 'force ' + col for col in cells.space_cols ])
    DVF = DVF.join(F)

    # add extra information if required
    if export_centers:
        xy = np.vstack([ cells[i].center for i in index ])
        DVF = DVF.join(pd.DataFrame(xy, index=index,
            columns=cells.space_cols))
        #DVF.to_csv('results.csv', sep='\t')

    info = {}
    # format the posteriors
    if posterior_info:
        cols = ['cell', 'fit', 'total']
        if len(posterior_info[0]) == 4:
            cols = ['iter'] + cols
        posterior_info = pd.DataFrame(np.array(posterior_info), columns=cols)
        info['posterior_info'] = posterior_info

    if return_struct:
        # cannot be rwa-stored
        info['result'] = result
    else:
        attrs = ['resolution', 'niter']
        if debug:
            if debug is True:
                attrs += [(val, key) for key, val in debug_all.items()]
            else:
                attrs += [(debug_all[attr], attr) for attr in debug]
        for src_attr in attrs:
            if isinstance(src_attr, str):
                dest_attr = src_attr
            else:
                src_attr, dest_attr = src_attr
            val = getattr(result, src_attr)
            if val is not None:
                info[dest_attr] = val
    return DVF, info

    return DVF, posterior_info

