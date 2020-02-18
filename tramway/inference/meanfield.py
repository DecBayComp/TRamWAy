# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

from .base import *
from .gradient import get_grad_kwargs
from warnings import warn
import numpy as np

_debug = True
def _debug_print(fname, *args):
    print('debug: in {}:\t'.format(fname), *args)

def nandot(a, b):
    return np.nansum(a * b)

def spatial_prior(a):
    return { k[8:]: a[k] for k in a if k.startswith('spatial_') } if isinstance(a, dict) else a
def time_prior(a):
    return { k[5:]: a[k] for k in a if k.startswith('time_') } if isinstance(a, dict) else a
def hyperparameter(a):
    return a.get('hyperparameter', None) if isinstance(a, dict) else a

def _get(_dict, _attr, _default):
    return _dict.get(_attr, _default) if isinstance(_dict, dict) else _default
# the following helpers encode the default values for the corresponding parameters
def nodes_free(a):
    return _get(a, 'nodes_free', False)
def neighbours_free(a):
    return _get(a, 'neighbours_free', True)
def cost_free(a):
    return _get(a, 'cost_free', False)

def unify_priors(global_prior=None, spatial_prior=None, time_prior=None):
    unified_prior = {}
    if global_prior is not None:
        if not isinstance(global_prior, dict):
            global_hyperparameter = global_prior
            unified_prior['spatial_hyperparameter'] = unified_prior['time_hyperparameter'] = \
                    global_hyperparameter
        else:
            unified_prior = dict(global_prior) # copy
    if spatial_prior is not None:
        if isinstance(spatial_prior, dict):
            for k in spatial_prior:
                unified_prior['spatial_'+k] = spatial_prior[k]
        else:
            unified_prior['spatial_hyperparameter'] = spatial_prior
    if time_prior is not None:
        if isinstance(time_prior, dict):
            for k in time_prior:
                unified_prior['time_'+k] = time_prior[k]
        else:
            unified_prior['time_hyperparameter'] = time_prior
    return unified_prior if unified_prior else None


class Meanfield(object):
    """
    The `Meanfield` class helps manage multiple features to be spatially
    and/or temporally regularized.

    It performs the common initial calculations such as mean displacements,
    and implements the meanfield regularization in multiple flavours,
    as detailed in :met:`regularize` and :met:`neg_log_prior`.

    Regularization flavours can be selected for each feature individually
    providing derived classes with a `prior` input argument described in
    :met:`parse_prior`.
    """
    def __init__(self, cells, dt=None, _inherit=False, **kwargs):
        """
        Derived classes should take a private `_inherit` input argument
        that should be ``False`` by default,
        and overwrite the `__init__` method first calling the parent class' `__init__`
        with ``_inherit=True``,
        declare additional features if any,
        and then run the following piece of code at the end of their `__init__` method::

            if not _inherit:
                self.__post_init__()
        """
        for arg in kwargs:
            warn('ignoring {} argument'.format(arg), RuntimeWarning)

        index, reverse_index, n, dt_mean, _, _, _, _ = \
            smooth_infer_init(cells, sigma2=0)#localization_error)
        if dt is None:
            dt = dt_mean
        elif np.isscalar(dt):
            dt = np.full_like(dt_mean, dt)

        index = np.array(index)
        ok = 1<n
        #print(ok.size, np.sum(ok))
        if not np.all(ok):
            reverse_index[index] -= np.cumsum(~ok)
            reverse_index[index[~ok]] = -1
            index, n, dt = index[ok], n[ok], dt[ok]

        dr, dr2 = [], []
        for i in index:
            dr_mean = np.mean(cells[i].dr, axis=0, keepdims=True)
            dr.append(dr_mean)
            dr2.append(np.sum(cells[i].dr * cells[i].dr))
        dr = np.vstack(dr)
        dr2 = np.array(dr2) / n

        self.cells = cells
        self.n = n
        self.dt = dt
        self.dr = dr
        self.dr2 = dr2
        self.index = index
        self.reverse_index = reverse_index

        self.unit_spatial_background = self.unit_temporal_background = None
        self.spatial_prior, self.time_prior = {}, {}
        self.static_nodes, self.static_neighbours = {}, {}
        self.static_cost = {}

        self._already_called = {}

        if not _inherit:
            raise NotImplementedError('Meanfield cannot be used directly as it does not declare any feature; use a derived class instead')
            self.__post_init__()

    # backward-compatibility code; maybe no longer used
    @property
    def _first_warning(self):
        return not self._already_called.get('warning', False)
    @_first_warning.setter
    def _first_warning(self, b):
        self._already_called['warning'] = not b

    @property
    def dtype(self):
        return self.dt.dtype

    def __post_init__(self):
        """
        This is to be run in __init__ at the end once ALL the features are declared.
        As a consequence, __init__ should take an _inherit input argument
        to determine whether the class is further derived and more features
        may be declared in a upper __init__ call.
        """
        any_spatial_prior = not all([ p is None for p in self.spatial_prior.values() ])
        any_time_prior = not all([ p is None for p in self.time_prior.values() ])
        if not any_spatial_prior and not any_time_prior:
            return

        # for all k, compute sum_l C_{k,l}
        ones = np.ones(self.n.size, dtype=self.dtype)
        if any_spatial_prior and self.unit_spatial_background is None:
            self.unit_spatial_background = self.spatial_penalties(ones)
        if any_time_prior and self.unit_temporal_background is None:
            self.unit_temporal_background = self.temporal_penalties(ones)

        self.A_additive_term, self.B_additive_term = {}, {}
        self.set_B_constants(self.unit_spatial_background, self.unit_temporal_background)

    def spatial_background(self, i, x):
        """
        for each k, computes:

            sum_l C_{k,l} x_l

        where C_{k,l} = _________1_________
                        N_k * (r_k - r_l)^2
        """
        dxs = []
        r0 = self.cells[i].center
        for j in self.cells.neighbours(i):
            dr = self.cells[j].center - r0
            dx = x[self.reverse_index[j]] / np.sum(dr * dr)
            if not np.isnan(dx):
                dxs.append(dx)
        return np.mean(dxs) if dxs else 0.

    def temporal_background(self, i, x):
        dxs = []
        t0 = self.cells[i].center_t
        for j in self.cells.time_neighbours(i):
            dt = self.cells[j].center_t - t0
            dx = x[self.reverse_index[j]] / (dt * dt)
            if not np.isnan(dx):
                dxs.append(dx)
        return np.mean(dxs) if dxs else 0.

    def regularized_cost(self, v, x, a, b):
        """ not used """
        ax = x - a
        ax[np.isnan(ax)] = 0
        b = np.array(b) # copy
        b[b==0] = np.inf
        cost = np.sum(ax * ax / (2 * b)) + self.neg_log_prior(v, x, a)
        return cost

    def neg_log_prior(self, v, psi, a_psi=None):
        """
        for any scalar feature psi at k,t:

            sum_l  C_{k,l} * (psi_k - a_psi_l)^2  +  sum_t'  C_{t,t'} * (psi_t - a_psi_t')^2

        each sum is computed as follows:

            psi_k *
                ( psi_k * (sum_l C_{k,l}) - 2 * (sum_l C_{k,l} * a_psi_l) )
                  +
                sum_l C_{k,l} * a_psi_l^2

        only `sum_l C_{k,l} * a_psi_l` and `sum_l C_{k,l} * a_psi_l^2` need to be evaluated
        for each new a_psi.

        If a_psi is not defined, a_psi and psi are the same variable.

        If static_cost is False for psi, a_psi is ignored anyway and psi is used instead.

        If psi and a_psi are matrices, their norm is used instead for the above expressions.
        """
        if _debug:
            key = ('neg_log_prior', v)
            if not self._already_called.get(key, False):
                self._already_called[key] = True
                _debug_print('neg_log_prior', v, self.static_cost[v])

        if psi.shape[1:]: # psi is a matrix
            psi = np.sqrt(np.sum(psi * psi, axis=1)) # use its norm instead
        if a_psi is None:
            a_psi = psi
        elif self.static_cost[v] is False:
            # ignore a_psi
            a_psi = psi
        elif a_psi.shape[1:]: # a_psi is a matrix
            a_psi = np.sqrt(np.sum(a_psi * a_psi, axis=1)) # use its norm instead

        cost = 0.
        if self.spatial_prior.get(v, None) is not None:
            if self.unit_spatial_background is None:
                self.unit_spatial_background = self.spatial_penalties(np.ones_like(psi))
            cost += nandot(self.spatial_prior[v],
                    psi * (psi * self.unit_spatial_background - 2 * self.spatial_penalties(a_psi)) + \
                            self.spatial_penalties(a_psi * a_psi))
        if self.time_prior.get(v, None) is not None:
            if self.unit_temporal_background is None:
                self.unit_temporal_background = self.temporal_penalties(np.ones_like(psi))
            cost += nandot(self.time_prior[v],
                    psi * (psi * self.unit_temporal_background - 2 * self.temporal_penalties(a_psi)) + \
                            self.temporal_penalties(a_psi * a_psi))
        return cost

    def set_B_constants(self, spatial_constants, time_constants):
        """
        In:

            A_psi = ( a_psi * b_psi + A_additive_term ) / B_psi
            B_psi = b_psi + B_additive_term

        B_additive_term does not depend on the value of psi and
        can be precomputed in the __init__ for all psi,
        hence the __post_init__ private method.
        """
        priors = self.spatial_prior
        for v in priors:
            B_additive_term = 0.
            if self.spatial_prior.get(v, None) is not None:
                B_additive_term = B_additive_term + \
                        2 * self.spatial_prior[v] * spatial_constants
            if self.time_prior.get(v, None) is not None:
                B_additive_term = B_additive_term + \
                        2 * self.time_prior[v] * time_constants
            self.B_additive_term[v] = B_additive_term

    def get_B_additive_term(self, feature):
        return self.B_additive_term[feature]

    def get_A_additive_term(self, feature, a=None, A=None):
        static = self.static_neighbours[feature]

        try:
            A_additive_term = self.A_additive_term[feature]
        except KeyError:
            pass
        else:
            if static:
                return A_addive_term
            else:
                warn('precomputed A_additive_term still available for feature \'{}\''.format(feature), RuntimeWarning)
                del self.A_additive_term[feature]

        if static or A is None:
            A = a

        if A is None:
            raise RuntimeError('initial {}{}{} is required'.format(
                    'a' if static else 'A',
                    '_' if 1<len(feature) else '',
                    feature))

        if A.shape[1:]:
            A_norm = np.sqrt(np.sum(A * A, axis=1))
            u = A / A_norm[:,np.newaxis]
            # penalise the norm instead
            A = A_norm
        else:
            u = None

        A_additive_term = 0.
        if self.spatial_prior[feature] is not None:
            A_additive_term = A_additive_term + \
                    2 * self.spatial_prior[feature] * self.spatial_penalties(A)
        if self.time_prior[feature] is not None:
            A_additive_term = A_additive_term + \
                    2 * self.time_prior[feature] * self.temporal_penalties(A)

        if u is not None:
            A_additive_term = A_additive_term[:,np.newaxis] * u

        if static:
            self.A_additive_term[feature] = A_additive_term

        if _debug:
            key = ('get_A_additive_term', feature)
            if not self._already_called.get(key, False):
                self._already_called[key] = True
                _debug_print('get_A_additive_term', feature, static)

        return A_additive_term

    def spatial_penalties(self, a):
        return np.array([ self.spatial_background(i, a) for i in self.index ])

    def temporal_penalties(self, a):
        return np.array([ self.temporal_background(i, a) for i in self.index ])

    def regularize(self, feature, a, b, A=None, B=None):
        """
        Computes A and B such that:

            A_k * B_k = a_k * b_k + 2 * sum_l C_{k,l} a_l
            B_k = b_k + 2 * sum_l C_{k,l}

        If the feature is not "static" and A^{(i-1)}, B^{(i-1)} are available,
        then A^{(i-1)}, B^{(i-1)} replace a, b.

        If the "nodes" only  are static, the above replacement applies only to a_l and not a_k, b_k.

        If the neighbours only are static, the above replacement applies only to a_k, b_k and not a_l.

        The "static" behaviour is initially set by :met:`parse_prior` and can be modified afterwards
        using methods like :met:`set_nodes_free` etc.

        If a and A are matrices, the norm of their rows is regularized instead.
        However b and B are vectors in any case.
        """
        A_additive_term = self.get_A_additive_term(feature, a, A)

        static = self.static_nodes[feature]
        if static:
            if B is None:
                B = b + self.B_additive_term[feature]
        else:
            if B is None:
                B = b
            B = B + self.B_additive_term[feature]

        if a.shape[1:]:
            A = ( a * b[:,np.newaxis] + A_additive_term ) / B[:,np.newaxis]
        else:
            A = ( a * b + A_additive_term ) / B

        if _debug:
            key = ('regularize', feature)
            if not self._already_called.get(key, False):
                self._already_called[key] = True
                _debug_print('regularize', feature, static)

        return A, B

    def add_feature(self, v):
        """ Registers a new feature to be included in the calculation of regularizing priors """
        self.spatial_prior[v] = self.time_prior[v] = None
        # get default values from the helper functions
        self.static_nodes[v] = not nodes_free(None)
        self.static_neighbours[v] = not neighbours_free(None)
        self.static_cost[v] = not cost_free(None)

    def parse_prior(self, feature, prior):
        """
        Allowed keys in the `prior` dict are:

        * spatial_hyperparameter (float): global trade-off hyperparameter for the spatial penalties

        * time_hyperparameter (float): global trade-off hyperparameter for the time penalties

        * nodes_free (bool): see :met:`regularize`

        * neighbours_free (bool): see :met:`regularize`

        * cost_free (bool): see :met:`neg_log_prior`

        """
        self.static_nodes[feature] = not nodes_free(prior)
        self.static_neighbours[feature] = not neighbours_free(prior)
        self.static_cost[feature] = not cost_free(prior)
        if _debug:
            _debug_print('parse_prior', feature, prior)
            _debug_print('parse_prior', feature,
                    self.static_nodes[feature],
                    self.static_neighbours[feature],
                    self.static_cost[feature])
        return spatial_prior(prior), time_prior(prior)

    def set_nodes_free(self, feature):
        self.static_nodes[feature] = False
    def set_nodes_static(self, feature):
        self.static_nodes[feature] = True
    def are_nodes_free(self, feature):
        return not self.static_nodes[feature]
    def set_neighbours_free(self, feature):
        try:
            del self.A_additive_term[feature]
        except KeyError:
            pass
        self.static_neighbours[feature] = False
    def set_neighbours_static(self, feature):
        try:
            # force A_additive_term to be precomputated again
            del self.A_additive_term[feature]
        except KeyError:
            pass
        self.static_neighbours[feature] = True
    def are_neighbours_free(self, feature):
        return not self.static_neighbours[feature]
    def static_landscape(self, feature):
        return self.static_nodes[feature] and self.static_neighbours[feature]
    def all_static(self):
        return all([ self.static_landscape(feature) for feature in self.static_nodes ])


############################
#  Single feature classes  #
############################

class Diffusivity(object):
    def __init__(self, cells, dt=None, diffusivity_prior=None,
            _inherit=False, **kwargs):

        Meanfield.__init__(self, cells, dt, _inherit=True, **kwargs)
        self.add_diffusivity_attributes(diffusivity_prior, _inherit)

    def add_diffusivity_attributes(self, diffusivity_prior, _inherit):

        self.add_feature('D')

        diffusivity_spatial_prior, diffusivity_time_prior = \
                self.parse_prior('D', diffusivity_prior)
        spatial_hyperparameter = hyperparameter(diffusivity_spatial_prior)
        time_hyperparameter = hyperparameter(diffusivity_time_prior)

        self.D_spatial_prior = spatial_hyperparameter if spatial_hyperparameter else None
        self.D_time_prior = time_hyperparameter if time_hyperparameter else None

        if not _inherit:
            self.__post_init__()

    @property
    def D_spatial_prior(self):
        return self.spatial_prior['D']
    @D_spatial_prior.setter
    def D_spatial_prior(self, D_spatial_prior):
        self.spatial_prior['D'] = D_spatial_prior

    @property
    def D_time_prior(self):
        return self.time_prior['D']
    @D_time_prior.setter
    def D_time_prior(self, D_time_prior):
        self.time_prior['D'] = D_time_prior

    @property
    def regularize_diffusivity(self):
        """ boolean property. Not to be confused with method `regularize_D` """
        return not (self.D_spatial_prior is None and self.D_time_prior is None)

    def regularize_D(self, aD, bD, AD=None, BD=None):
        return self.regularize('D', aD, bD, AD, BD)


class Friction(object):
    def __init__(self, cells, dt=None, friction_prior=None,
            _inherit=False, **kwargs):

        Meanfield.__init__(self, cells, dt, _inherit=True, **kwargs)
        self.add_friction_attributes(friction_prior, _inherit)

    def add_friction_attributes(self, friction_prior, _inherit):

        self.add_feature('psi')

        friction_spatial_prior, friction_time_prior = \
                self.parse_prior('psi', friction_prior)
        spatial_hyperparameter = hyperparameter(friction_spatial_prior)
        time_hyperparameter = hyperparameter(friction_time_prior)

        dt = self.dt
        self.psi_spatial_prior = spatial_hyperparameter / (2 * dt) \
                if spatial_hyperparameter else None
        self.psi_time_prior = time_hyperparameter / (2 * dt) \
                if time_hyperparameter else None

        if not _inherit:
            self.__post_init__()

    @property
    def psi_spatial_prior(self):
        return self.spatial_prior['psi']
    @psi_spatial_prior.setter
    def psi_spatial_prior(self, psi_spatial_prior):
        self.spatial_prior['psi'] = psi_spatial_prior

    @property
    def psi_time_prior(self):
        return self.time_prior['psi']
    @psi_time_prior.setter
    def psi_time_prior(self, psi_time_prior):
        self.time_prior['psi'] = psi_time_prior

    @property
    def regularize_friction(self):
        """ boolean property. Not to be confused with method `regularize_psi` """
        return not (self.psi_spatial_prior is None and self.psi_time_prior is None)

    def regularize_psi(self, a_psi, b_psi, A_psi=None, B_psi=None):
        return self.regularize('psi', a_psi, b_psi, A_psi, B_psi)


class Drift(Meanfield):
    def __init__(self, cells, dt=None, drift_prior=None,
            _inherit=False, **kwargs):

        Meanfield.__init__(self, cells, dt, _inherit=True, **kwargs)
        self.add_drift_attributes(drift_prior, _inherit)

    def add_drift_attributes(self, drift_prior, _inherit):

        self.add_feature('mu')

        drift_spatial_prior, drift_time_prior = \
                self.parse_prior('mu', drift_prior)
        spatial_hyperparameter = hyperparameter(drift_spatial_prior)
        time_hyperparameter = hyperparameter(drift_time_prior)

        dt = self.dt
        self.mu_spatial_prior = spatial_hyperparameter * dt \
                if spatial_hyperparameter else None
        self.mu_time_prior = time_hyperparameter * dt \
                if time_hyperparameter else None

        if not _inherit:
            self.__post_init__()

    @property
    def mu_spatial_prior(self):
        return self.spatial_prior['mu']
    @mu_spatial_prior.setter
    def mu_spatial_prior(self, mu_spatial_prior):
        self.spatial_prior['mu'] = mu_spatial_prior

    @property
    def mu_time_prior(self):
        return self.time_prior['mu']
    @mu_time_prior.setter
    def mu_time_prior(self, mu_time_prior):
        self.time_prior['mu'] = mu_time_prior

    @property
    def regularize_drift(self):
        """ boolean property. Not to be confused with method `regularize_mu` """
        return not (self.mu_spatial_prior is None and self.mu_time_prior is None)

    def regularize_mu(self, a_mu, b_mu, A_mu=None, B_mu=None):
        return self.regularize('mu', a_mu, b_mu, A_mu, B_mu)


class Potential(Meanfield):
    def __init__(self, cells, dt=None, potential_prior=None,
            _inherit=False, grad_selection_angle=.5, **kwargs):

        # identify and extract gradient-specific arguments;
        # this must be done before Meanfield.__init__
        grad_kwargs = get_grad_kwargs(kwargs, grad_selection_angle=grad_selection_angle)

        Meanfield.__init__(self, cells, dt, _inherit=True, **kwargs)

        self.add_potential_attributes(cells, potential_prior, _inherit,
                grad_kwargs=grad_kwargs)

    def add_potential_attributes(self, cells, potential_prior, _inherit,
            grad_kwargs=None, **kwargs):

        self.add_feature('V')

        ## precompute hyperparameter arrays
        potential_spatial_prior, potential_time_prior = \
                self.parse_prior('V', potential_prior)
        spatial_hyperparameter = hyperparameter(potential_spatial_prior)
        time_hyperparameter = hyperparameter(potential_time_prior)
        dt = self.dt
        self.V_spatial_prior = spatial_hyperparameter * dt \
                if spatial_hyperparameter else None
        self.V_time_prior = time_hyperparameter * dt \
                if time_hyperparameter else None

        n = self.n
        dr = self.dr
        dtype = self.dtype
        index, reverse_index = self.index, self.reverse_index

        ## spatial gradient routines for V
        if grad_kwargs is None:
            grad_kwargs = get_grad_kwargs(kwargs)
        self.gradient_sides = sides = grad_kwargs.get('side', '>')

        grad_kwargs['na'] = np.nan
        def local_grad(i, x, s):
            grad_kwargs['side'] = s
            return cells.grad(i, x, reverse_index, **grad_kwargs)
        def grad(x, s):
            return np.stack([ local_grad(i, x, s) for i in index ])
        _one = np.zeros(n.size, dtype=dtype) # internal
        def diff_grad(x, s):
            _x = np.array(x) # copy
            _dx = []
            for k in range(x.size):
                _x[k] = 0
                _dx.append(local_grad(index[k], _x, s))
                _x[k] = x[k]
            return np.vstack(_dx)

        self.grad, self.diff_grad = grad, diff_grad

        ## constants for the spatial gradient
        B, Bstar, B2 = {s: [] for s in sides}, {s: [] for s in sides}, {s: [] for s in sides}
        regions = []
        for k, i in enumerate(index):
            _one[k] = 1
            for s in sides:
                _unit_grad = local_grad(i, _one, s)
                _undefined = np.isnan(_unit_grad)
                _unit_grad[_undefined] = 0.
                _B = -_unit_grad
                _B2 = np.dot(_B, _B)
                _B[_undefined] = np.nan
                _B_over_B2 = _B if _B2 == 0 else _B / _B2
                B[s].append(_B)
                B2[s].append(_B2)
                Bstar[s].append(_B_over_B2)
            _one[k] = 0

        self.regions = regions
        self.Bstar = {s: np.vstack(Bstar[s]) for s in Bstar}
        self.B2 = {s: np.array(B2[s]) for s in B2}
        self.B = {s: np.vstack(B[s]) for s in B}

        ## make feature-specific attributes
        if not _inherit:
            self.__post_init__()

    @property
    def V_spatial_prior(self):
        return self.spatial_prior['V']
    @V_spatial_prior.setter
    def V_spatial_prior(self, V_spatial_prior):
        self.spatial_prior['V'] = V_spatial_prior

    @property
    def V_time_prior(self):
        return self.time_prior['V']
    @V_time_prior.setter
    def V_time_prior(self, V_time_prior):
        self.time_prior['V'] = V_time_prior

    @property
    def regularize_potential(self):
        """ boolean property. Not to be confused with method `regularize_V` """
        return not (self.V_spatial_prior is None and self.V_time_prior is None)

    def regularize_V(self, aV, bV, AV=None, BV=None):
        return self.regularize('V', aV, bV, AV, BV)


#########################
#  Two-feature classes  #
#########################

class DiffusivityDrift(Diffusivity, Drift):
    def __init__(self, cells, dt=None, diffusivity_prior=None, drift_prior=None,
            _inherit=False, **kwargs):

        Diffusivity.__init__(self, cells, dt, diffusivity_prior,
                _inherit=_inherit, **kwargs) # first init should not call __post_init__
        self.add_drift_attributes(drift_prior, _inherit)


class DiffusivityPotential(Diffusivity, Potential):
    def __init__(self, cells, dt=None, diffusivity_prior=None, potential_prior=None,
            _inherit=False, **kwargs):

        # it is by far safer to make potential-related attributes before diffusivity-
        Potential.__init__(self, cells, dt, potential_prior,
                _inherit=True, **kwargs) # first init should not call __post_init__
        self.add_diffusivity_attributes(diffusivity_prior, _inherit)


class FrictionDrift(Friction, Drift):
    def __init__(self, cells, dt=None, friction_prior=None, drift_prior=None,
            _inherit=False, **kwargs):

        Friction.__init__(self, cells, dt, friction_prior,
                _inherit=True, **kwargs) # first init should not call __post_init__
        self.add_drift_attributes(drift_prior, _inherit)


class FrictionPotential(Friction, Potential):
    def __init__(self, cells, dt=None, friction_prior=None, potential_prior=None,
            _inherit=False, **kwargs):

        Potential.__init__(self, cells, dt, potential_prior,
                _inherit=True, **kwargs) # first init should not call __post_init__
        self.add_friction_attributes(friction_prior, _inherit)

