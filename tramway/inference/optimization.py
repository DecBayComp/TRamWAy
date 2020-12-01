# -*- coding: utf-8 -*-

# Copyright © 2018-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

from __future__ import absolute_import

from math import *
import numpy as np
import scipy.optimize.linesearch as ls
import time
import scipy.sparse as sparse
from collections import namedtuple, defaultdict, deque
import traceback
from tramway.core import parallel
import logging


BFGSResult = namedtuple('BFGSResult', ('x', 'H', 'resolution', 'niter', 'f', 'df', 'projg', 'cumtime', 'err', 'diagnosis', 'ncalls'))


def wolfe_line_search(f, x, p, g, subspace=None, args_f=(), args_g=None, args=None, f0=None, g0=None,
        bounds=None, weight_regul=None, step_regul=None,
        eta_max=1., iter_max=10, step_max=None, c0=.5, c1=1e-4, c2=.9, c3=.9, c4=.1, c5=1e-10,
        armijo_max=None, return_resolution=False):
    """
    Wolfe line search along direction `p` possibly restricted to a subspace.

    If defined, argument `subspace` is passed to gradient function `g` as last positional argument,
    unless already present in `args_g`.

    `args` is an alias for `args_f`.
    If `args_g` is ``None``, then it falls back to `args_f`.

    .. code-block:: math

        f(x_k + eta_k p_k) <= f(x_k) + c1 eta_k p_k^T g(x_k) # Armijo rule
        c2 * p_k^T g(x_k) <= -p_k^T g(x_k + eta_k p_k) <= -c3 p_k^T g(x_k) # weak (c2=None) or strong (c2=c3) Wolfe condition
        c4 eta_k <= eta_{k+1} <= c0 eta_k
        eta_min = c5 eta_max

    """
    if args is not None:
        if args is args_f:
            pass
        elif args_f == ():
            args_f = args
        else:
            raise ValueError('both `args` and `args_f` are defined')
    if not isinstance(args_f, (tuple, list)):
        args_f = (args_f,)
    if args_g is None:
        args_g = args_f
    elif not isinstance(args_g, (tuple, list)):
        args_g = (args_g,)
    if f0 is None:
        f0 = f(x, *args_f)
    if subspace is not None and subspace not in args_g:
        args_g = list(args_g)
        args_g.append(subspace)
    if g0 is None:
        g0 = g(x, *args_g)
    if subspace is None:
        x0 = x
    else:
        x0 = x[subspace]
    x = np.array(x)
    #
    assert not x.shape[1:]
    assert not p.shape[1:]
    assert not g0.shape[1:]
    #
    slope = np.dot(p, g0)
    if 0 <= slope:
        if return_resolution:
            return None, 'not a descent direction'
        else:
            return None#raise ValueError('not a descent direction')
    if step_max:
        norm = np.max(np.abs(p))
        if step_max < norm * eta_max:
            eta_max *= step_max / norm
    f_prev = f0
    if weight_regul:
        f0 += weight_regul * np.dot(x0, x0)
    if bounds:
        lb, ub = bounds
        if subspace is not None:
            if lb is not None and len(lb) == x.size:
                lb = lb[subspace]
            if ub is not None and len(ub) == x.size:
                ub = ub[subspace]
    else:
        assert bounds is None
        bounds = lb = ub = None
        _p, _slope = p, slope
    if c5 is None:
        eta_min = -np.inf
    else:
        eta_min = eta_max * c5
    eta = eta_prev = eta_max
    #eta_hist, f_hist, norm_p_hist, df_hist, armijo_hist = [], [f0], [], [], []
    any_f_defined = False
    i = 0
    while True:
        #eta_hist.append(eta)
        if eta < eta_min:
            #print('eta < eta_min={}'.format(eta_min))
            res = 'eta < eta_min={}'.format(eta_min)
            break
        _x = x0 + eta * p
        if bounds:
            _x = _proj(_x, lb, ub)
            _p = _x - x0
        else:
            _slope = eta * slope
        #norm_p_hist.append(np.max(np.abs(_p)))
        if subspace is None:
            x = _x
        else:
            x[subspace] = _x
        try:
            fx = f(x, *args_f)
        except ValueError:
            fx = None
        #f_hist.append(fx)
        if fx is None:
            #assert np.all(0 < x) # DEBUG
            eta *= c0
        else:
            any_f_defined = True
            if weight_regul:
                fx += weight_regul * np.dot(_x, _x)
            if step_regul:
                fx += step_regul + np.dot(_p, _p)
            df = fx - f0
            #df_hist.append(df)
            if bounds:
                _slope = np.dot(_p, g0)
            #armijo_hist.append(c1 * _slope)
            if _slope < 0:
                armijo_threshold = c1 * _slope
                if armijo_max and armijo_threshold < -armijo_max:
                    armijo_threshold = -armijo_max
                armijo_rule = df <= armijo_threshold
                #print((f0, eta, fx, armijo_rule))
                if armijo_rule:
                    if c2 is None and c3 is None:
                        # do not test Wolfe condition; stop here
                        p = _x - x0
                        if return_resolution:
                            return p, 'Armijo criterion met'
                        else:
                            return p
                    try:
                        gx = g(x, *args_g)
                    except ValueError:
                        gx = None
                    if gx is not None:
                        new_slope = np.dot(_p, gx)
                        if c3 is not None:
                            curvature_condition = c3 * _slope <= new_slope # new_slope less negative than slope (can be positive!)
                        else:
                            curvature_condition = True
                        if curvature_condition and c2 is not None:
                            curvature_condition = new_slope <= -c2 * _slope
                        #print((_slope, new_slope, curvature_condition))
                        if curvature_condition:
                            p = _x - x0
                            if return_resolution:
                                return p, 'Wolfe criterion met'
                            else:
                                return p
            if eta == eta_max:
                eta1 = -slope / (2. * (df - slope))
            else:
                df_prev = f_prev - f0
                rhs1 = df - eta * slope
                rhs2 = df_prev - eta_prev * slope
                eta2 = eta_prev
                rhs1 /= eta * eta
                rhs2 /= eta2 * eta2
                a = (rhs1 - rhs2) / (eta - eta2)
                b = (eta * rhs2 - eta2 * rhs1) / (eta - eta2)
                if a == 0:
                    eta1 = -slope / (2. * b)
                else:
                    discr = b * b - 3. * a * slope
                    if discr < 0:
                        eta1 = eta * c0
                    elif b <= 0:
                        eta1 = (np.sqrt(discr) - b) / (3. * a)
                    else:
                        eta1 = -slope / (np.sqrt(discr) + b)
            eta_prev, f_prev = eta, fx
            eta = max(c4 * eta, min(eta1, c0 * eta)) # c3 * eta{k} <= eta{k+1} <= c0 * eta{k}
        i += 1
        if iter_max and i == iter_max:
            #print('iter_max reached')
            if any_f_defined:
                res = 'iter_max reached'
            else:
                res = "could not find the function's support"
            break
    #print((eta_hist, norm_p_hist, f_hist, df_hist, armijo_hist))
    if return_resolution:
        return None, res
    else:
        return None # line search failed


def _proj(x, lb, ub):
    if lb is not None:
        x = np.maximum(lb, x)
    if ub is not None:
        x = np.minimum(x, ub)
    return x


def define_pprint():
    component_strlen = [0]
    def format_component(_c, _f=True):
        if _c is None:
            return 'f' if _f else ''
        _c = str(_c)
        _strlen = len(_c)
        if component_strlen[0] <= _strlen:
            component_strlen[0] = _strlen
            _sp = ''
        else:
            _sp = ' ' * (component_strlen[0] - _strlen)
        return 'f({}{})'.format(_sp, _c)
    def msg0(_i, _c, _f, _dg):
        _i = str(_i)
        return 'At iterate {}{}\t{}= {}{:E} \tproj g = {:E}'.format(
            ' ' * max(0, 3 - len(_i)), _i,
            format_component(_c),
            ' ' if 0 <= _f else '', _f, _dg)
    def msg1(_i, _c, _f0, _f1, _dg=None):
        _i = str(_i)
        _df = _f1 - _f0
        return 'At iterate {}{}\t{}= {}{:E} \tdf= {}{:E}{}'.format(
            ' ' * max(0, 3 - len(_i)), _i,
            format_component(_c),
            ' ' if 0 <= _f1 else '', _f1,
            ' ' if 0 <= _df else '', _df,
            '' if _dg is None else ' \tproj g = {:E}'.format(_dg))
    def msg2(_i, _c, *_args):
        _c = format_component(_c)
        if len(_args) == 3 and isinstance(_args[1], str) and isinstance(_args[-1], (tuple, list)):
            _exc_type, _exc_msg, _exc_args = _args
            try:
                _exc_msg = _exc_msg.replace('%s', '{}').format(*_exc_args)
            except (SystemExit, KeyboardInterrupt):
                raise
            except:
                pass
            else:
                _args = (_exc_type, _exc_msg)
        if _c:
            _args = (_c,) + _args
        msg = ''.join(('At iterate {}{}\t', ':  '.join(['{}'] * len(_args))))
        _i = str(_i)
        return msg.format(' ' * max(0, 3 - len(_i)), _i, *_args)
    return msg0, msg1, msg2


class _defaultdict(object):
    """
    defaultdict that passes the key as first positional argument to the factory function.
    """
    __slots__ = ('_dict', 'arg', 'factory', 'args', 'kwargs')
    def __init__(self, factory, *args, **kwargs):
        """
        Arguments:

            factory (callable): factory function; takes a key as first positional argument.

            init_argument (map or callable): function or table that hashes the key before the latter
                is passed to `factory`; `init_argument` must be keyworded!

        Other positional and keyword arguments are passed to `factory`.
        """
        self._dict = {}
        self.factory = factory
        self.arg = kwargs.pop('init_argument', None)
        self.args = args
        self.kwargs = kwargs
    def __make__(self, arg):
        return self.factory(arg, *self.args, **self.kwargs)
    def __nonzero__(self):
        return bool(self._dict)
    def __getitem__(self, i):
        try:
            item = self._dict[i]
        except KeyError:
            if self.arg is None:
                arg = i
            elif callable(self.arg):
                arg = self.arg(i)
            else:
                arg = self.arg[i]
            item = self.__make__(arg)
            self._dict[i] = item
        return item
    def __setitem__(self, i, item):
        self._dict[i] = item
    def __len__(self):
        return len(self._dict)
    def __iter__(self):
        return iter(self._dict)


class SparseFunction(parallel.Workspace):
    """ Parameter singleton.

    Attributes:

        x (numpy.ndarray): working copy of the parameter vector.

        covariate (callable): returns the components that covary with the input component (all indices).

        gradient_subspace (callable): takes a component index and returns the indices of the related
            parameters.

        descent_subspace (callable): takes a component index and returns the indices of the related
            parameters.

        eps (float): initial scaling of the descent direction when no curvature information is available.

        fun (callable): local cost function (takes a component index before the parameter vector).

        _sum (callable): mixture function for local costs.

        args (sequence): positional input arguments to `fun`.

        regul (float): regularization coefficient on the parameters.

        bounds (tuple): (lower, upper) bounds as a couple of lists.

        h0 (float): gradient initial step.

        ncalls (int): number of calls to `fun`.

    See also :func:`minimize_sparse_bfgs`.
    """
    def __init__(self, x, covariate, gradient_subspace, descent_subspace,
            eps, fun, _sum, args, regul, bounds, h0):
        parallel.Workspace.__init__(self, x, *args)
        self.covariate = covariate
        self.gradient_subspace = gradient_subspace
        self.descent_subspace = descent_subspace
        self.eps = eps
        self._fun = fun
        self.sum = _sum
        self.args = args
        self.regul = regul
        self.bounds = bounds
        self.h0 = h0
        self.ncalls = 0

    @property
    def x(self):
        return self.data_array
    def update(self, component):
        #assert self.x is self._extensions[0].combined # stochastic_dv only
        parallel.Workspace.update(self, component)
        component.push()
        #self.x[component.descent_subspace] = component.x
    def fun(self, *args, **kwargs):
        self.ncalls += 1
        return self._fun(*args, **kwargs)

def extend_global(__global__, independent_components, memory, newton, gradient_covariate):
    """ Add attributes to the workspace.

    Arguments:

        __global__ (SparseFunction): parameter singleton (modified inplace).

        independent_components (bool): whether the curvature information is maintained for each
            component independently.

        memory (int): number of update pairs for inverse Hessian approximation.

        newton (bool): whether to operate as a quasi-Newton algorithm with Cauchy point estimation.

        gradient_covariate (callable): see also :func:`minimize_sparse_bfgs`.

    """
    # choose an implementation
    if not newton:
        __global__.inverse_hessian_block = GradientDescent
    elif independent_components:
        if memory:
            __global__.memory = memory
            __global__.inverse_hessian_block = LimitedMemoryInverseHessianBlock
        else:
            __global__.inverse_hessian_block = IndependentInverseHessianBlock
    else:
        # just ignore
        #if memory:
        #    raise NotImplementedError('`memory` requires `independent_components`')
        n = __global__.x.size
        __global__.H = sparse.lil_matrix((n, n))
        __global__.inverse_hessian_block = InverseHessianBlockView
    # component
    if gradient_covariate is not None:
        __global__.gradient_covariate = gradient_covariate


class LocalSubspace(parallel.abc.JobStep):
    """ Working associated gradient and descent subspaces.

    Abstract class; only for type testing ``isinstance(obj, LocalSubspace)``.

    Children classes should expose attributes or properties `__global__`, `i`, `n`, `covariate`,
    `gradient_subspace`, `gradient_subspace_size`, `descent_subspace`, `descent_subspace_size`
    and `subspace_map`,
    and methods `in_full_space`, `in_gradient_subspace` and `in_descent_subspace`.
    """
    pass
class LocalSubspaceSingleton(parallel.JobStep):
    """ For each different subspace, this class should be instanciated only once.

    From `__global__` requires `x`, `covariate`, `gradient_subspace` and `descent_subspace`.
    """
    __slots__ = ('_subspace_map', '_gradient_subspace_size', '_descent_subspace_size')
    def __init__(self, i=None, _global=None):
        parallel.JobStep.__init__(self, i, _global)
        self._subspace_map = None
        self._gradient_subspace_size = None
        self._descent_subspace_size = None
    @property
    def resources(self):
        return self.gradient_subspace
    @property
    def __global__(self):
        return self.get_workspace()
    @__global__.setter
    def __global__(self, ws):
        self.set_workspace(ws)
    @property
    def _size_error(self):
        return ValueError('vector size does not match any (sub)space')
    @property
    def n(self):
        return self.__global__.x.size
    @property
    def i(self):
        if self.resource_id is None:
            raise ValueError('component attribute `i` is not set')
        return self.resource_id
    @i.setter
    def i(self, i):
        if (self.resource_id is None) != (i != self.resource_id): # _i is None xor i != _i
            warn('`i` is supposed to be read-only', RuntimeWarning)
            self._id = i
            self._subspace_map = None
            self._gradient_subspace_size = None
            self._descent_subspace_size = None
    @property
    def covariate(self):
        return self.__global__.covariate(self.i)
    @property
    def gradient_subspace(self):
        return self.__global__.gradient_subspace(self.i)
    @property
    def gradient_subspace_size(self):
        if self._gradient_subspace_size is None:
            try:
                g = self.g
            except AttributeError:
                g = None
            if g is None:
                j = self.gradient_subspace
                if j is None: # full space
                    self._gradient_subspace_size = self.n
                else:
                    self._gradient_subspace_size = len(j)
            else:
                self._gradient_subspace_size = len(g)
        return self._gradient_subspace_size
    @property
    def descent_subspace(self):
        j = self.__global__.descent_subspace(self.i)
        if j is None: # falls back onto gradient subspace
            j = self.gradient_subspace
        return j
    @property
    def descent_subspace_size(self):
        if self._descent_subspace_size is None:
            j = self.descent_subspace
            if j is None: # full space
                self._descent_subspace_size = self.n
            else:
                self._descent_subspace_size = len(j)
        return self._descent_subspace_size
    def in_full_space(self, vec, copy=False):#, working_copy=x
        if vec is None:
            return None
        if vec.size == self.n:
            if copy:
                vec = np.array(vec)
            return vec
        if vec.size == self.gradient_subspace_size:
            j = self.gradient_subspace
        elif vec.size == self.descent_subspace_size:
            j = self.descent_subspace
        else:
            raise self._size_error
        working_copy = self.__global__.x
        if copy:
            working_copy = np.array(working_copy)
        working_copy[j] = vec
        return working_copy
    def in_gradient_subspace(self, vec, copy=False):
        if vec is None:
            return None
        if vec.size == self.n:
            j = self.gradient_subspace
            if j is not None:
                vec = vec[j]
            elif copy:
                vec = np.array(vec)
        elif vec.size == self.gradient_subspace_size:
            if copy:
                vec = np.array(vec)
        elif vec.size == self.descent_subspace_size:
            _vec = np.zeros(self.gradient_subspace_size, dtype=vec.dtype)
            _vec[self.subspace_map] = vec
            vec = _vec
        else:
            raise self._size_error
        return vec
    def in_descent_subspace(self, vec, copy=False):
        if vec is None:
            return None
        if vec.size == self.n:
            j = self.descent_subspace
            if j is not None:
                vec = vec[j]
            elif copy:
                vec = np.array(vec)
        elif vec.size == self.descent_subspace_size:
            if copy:
                vec = np.array(vec)
        elif vec.size == self.gradient_subspace_size:
            vec = vec[self.subspace_map]
        else:
            raise self._size_error
        return vec
    @property
    def subspace_map(self):
        if self._subspace_map is None:
            jg = self.gradient_subspace
            assert jg is not None
            jd = self.descent_subspace
            assert jd is not None
            self._subspace_map = [ (jg==j).nonzero()[0][0] for j in jd ]
        return self._subspace_map
class LocalSubspaceProxy(object):
    """ Working implementation of `LocalSubspace` that can be instanciated multiple times,
    reusing it.
    """
    __slots__ = ('__proxied__',)
    def __init__(self, i, *args, **kwargs):
        if isinstance(i, LocalSubspaceProxy):
            self.__proxied__ = i.__proxied__
        elif isinstance(i, LocalSubspaceSingleton):
            self.__proxied__ = i
        else:
            self.__proxied__ = LocalSubspaceSingleton(i, *args, **kwargs)
    @property
    def resource_id(self):
        return self.__proxied__.resource_id
    @property
    def resources(self):
        return self.__proxied__.resources
    def get_workspace(self):
        return self.__proxied__.get_workspace()
    def set_workspace(self, ws):
        self.__proxied__.set_workspace(ws)
    def unset_workspace(self):
        self.__proxied__.unset_workspace()
    @property
    def workspace_set(self):
        return self.__proxied__.workspace_set
    @property
    def __global__(self):
        return self.__proxied__.__global__
    @property
    def n(self):
        return self.__proxied__.n
    @property
    def i(self):
        return self.__proxied__.i
    @i.setter
    def i(self, i):
        raise AttributeError('the `i` property is read-only')
        self.__proxied__.i = i
    @property
    def covariate(self):
        return self.__proxied__.covariate
    @property
    def gradient_subspace(self):
        return self.__proxied__.gradient_subspace
    @property
    def gradient_subspace_size(self):
        return self.__proxied__.gradient_subspace_size
    @property
    def descent_subspace(self):
        return self.__proxied__.descent_subspace
    @property
    def descent_subspace_size(self):
        return self.__proxied__.descent_subspace_size
    @property
    def subspace_map(self):
        return self.__proxied__.subspace_map
    def in_full_space(self, *args, **kwargs):
        return self.__proxied__.in_full_space(*args, **kwargs)
    def in_gradient_subspace(self, *args, **kwargs):
        return self.__proxied__.in_gradient_subspace(*args, **kwargs)
    def in_descent_subspace(self, *args, **kwargs):
        return self.__proxied__.in_descent_subspace(*args, **kwargs)
    def _format(self, msg, *args):
        return ('in component {}: '+msg).format(self.i, *args)
LocalSubspace.register(LocalSubspaceSingleton)
LocalSubspace.register(LocalSubspaceProxy)
# inverse Hessian matrix
Pair = namedtuple('Pair', ('s', 'y', 'rho', 'gamma'))
class InverseHessianBlock(LocalSubspaceProxy):
    """ Abstract class for local inverse Hessian.

    From `__global__` requires `eps`.

    May also expose the internal representation of a block as attribute or property `block`.
    """
    __slots__ = ()
    @property
    def eps(self):
        return self.__global__.eps
    def dot(self, g):
        return self.block.dot(g)
    def update(self, s, y, proj):
        if proj is None:
            proj = np.dot(s, self.in_descent_subspace(y))
        H = self.block
        if sparse.issparse(H):
            H = H.toarray()
        Hy = self.dot(y) # or H.dot(y)
        yHy = np.dot(y, Hy)
        assert 0 <= yHy
        sp = s / proj
        # see also: https://github.com/scipy/scipy/blob/v1.3.0/scipy/optimize/_hessian_update_strategy.py#L294-L310
        self.block = H + (\
                np.outer((proj + yHy) * sp, sp) - np.outer(Hy, sp) - np.outer(sp, Hy))
    def drop(self):
        raise NotImplementedError('abstract method')
class GradientDescent(InverseHessianBlock):
    """Does not use Cauchy points."""
    __slots__ = ()
    def dot(self, g):
        return self.eps * g
    def update(self, *args):
        pass
    def drop(self):
        pass
class InverseHessianBlockView(InverseHessianBlock):
    __slots__ = ('fresh', 'slice')
    def __init__(self, component):
        InverseHessianBlock.__init__(self, component)
        self.fresh = True
        if self.gradient_subspace is None:
            self.slice = None
        else:
            self.slice = np.ix_(self.gradient_subspace, self.gradient_subspace)
    def drop(self):
        self.fresh = True
    @property
    def block(self):
        block = self.__global__.H
        if block is not None and self.slice is not None:
            block = block[self.slice]
        if self.fresh:
            if block is None:
                block = sparse.identity(self.gradient_subspace_size, format='lil')
            elif sparse.issparse(block):
                block = block + sparse.identity(self.gradient_subspace_size, format=block.getformat())
            else:
                block[np.diag_indices(block.shape[0])] += 1.
            block *= self.eps
        return block
    @block.setter
    def block(self, block):
        if self.slice is None:
            self.__global__.H = block
        else:
            self.__global__.H[self.slice] = block
        self.fresh = False
    def dot(self, g):
        p = InverseHessianBlock.dot(self, g)
        if p.shape[1:]:
            p = p.reshape(-1)
        return p
class IndependentInverseHessianBlock(InverseHessianBlock):
    __slots__ = ('block',)
    def __init__(self, component):
        InverseHessianBlock.__init__(self, component)
        self.drop()
    def drop(self):
        self.block = self.eps * np.identity(self.gradient_subspace_size)
class LimitedMemoryInverseHessianBlock(InverseHessianBlock):
    """From `__global__` requires `memory`."""
    __slots__ = ('block',)
    def __init__(self, component):
        InverseHessianBlock.__init__(self, component)
        self.drop()
    def drop(self):
        self.block = deque([], self.__global__.memory)
    def dot(self, g):
        if self.block:
            # `block` is nonempty
            p = np.array(g) # copy
            U = []
            for u in self.block: # for each past block from k-1 to k-m
                if self.gradient_subspace is None or len(u.s) == self.gradient_subspace_size:
                    alpha = u.rho * np.dot(u.s, p)
                else:
                    alpha = u.rho * np.dot(u.s, self.in_descent_subspace(p))
                p -= alpha * u.y
                U.append((u.s, u.y, u.rho, alpha))
            p *= -self.last.gamma # gamma_{k-1}
            for s, y, rho, alpha in U[::-1]: # from k-m to k-1
                beta = rho * np.dot(y, p)
                if len(s) == len(p):
                    p += (alpha - beta) * s
                else:
                    assert self.subspace_map is not None
                    p[self.subspace_map] += (alpha - beta) * s
            return -p
        else:
            return self.eps * g
    def update(self, s, y, proj):
        rho = 1. / proj
        gamma = proj / np.dot(y, y)
        self.block.appendleft(Pair(s, y, rho, gamma))
    @property
    def last(self):
        return self.block[0]
# component
class Component(LocalSubspaceProxy, parallel.UpdateVehicle):
    """
    From `__global__` requires `fun`, `_sum` and `args`.

    `__global__.fun` is the local cost function.
    `f` is the partially-evaluated cost function.
    """
    __slots__ = ('_x', '_f', '_g', '_H') + parallel.VehicleJobStep.__slots__
    def __init__(self, i, *args, **kwargs):
        LocalSubspaceProxy.__init__(self, i, *args, **kwargs)
        parallel.UpdateVehicle.__init__(self)
        self._x = None # gradient-active elements only
        self._f = None
        self._g = None # gradient-active elements only
        self._H = None
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, _x):
        assert _x is None
        self._x = self.in_gradient_subspace(_x)
        self.f = None
        self.g = None
        self.H = None
    def pull(self, _x):
        self._x = self.in_gradient_subspace(_x)
        #assert self._x is not _x # fails if not _stochastic
        self._f = self.__f__(_x)
        assert self._f is not None
        self._g = self.__g__(_x, update=True)
        assert self._g is not None
    def __f__(self, _x):
        #assert _x is x # check there is a single working copy
        return self.__global__.sum([ self.__global__.fun(j, _x, *self.__global__.args)
                    for j in self.covariate ])
    @property
    def f(self):
        if self._f is None:
            self._f = self.__f__(self.in_full_space(self.x))
        return self._f
    @f.setter
    def f(self, f):
        self._f = f
    def __g__(self, _x, subspace=None, covariate=None, update=False):
        #assert _x is x # check there is a single working copy
        if subspace is None:
            subspace = self.gradient_subspace
        if covariate is None:
            try:
                covariate = self.__global__.gradient_covariate
            except AttributeError:
                covariate = self.covariate
        _total_g, _partial_g = sparse_grad(self.__global__.fun, _x, covariate,
                subspace, self.__global__.args, self.__global__.sum, self.__global__.regul,
                self.__global__.bounds, self.__global__.h0)
        return _total_g
    @property
    def g(self):
        if self._g is None:
            self._g = self.__g__(self.in_full_space(self.x), update=True)
        return self._g
    @g.setter
    def g(self, g):
        self._g = self.in_gradient_subspace(g)
    @property
    def H(self):
        if self._H is None:
            self._H = self.__global__.inverse_hessian_block(self)
        return self._H
    @H.setter
    def H(self, H):
        self._H = H
    def commit(self, s):
        c = Component(self) # fails if `i` is not set
        if self.x is None:
            raise ValueError(self._format('parameter vector `x` not defined'))
        #if self.s is None:
        #    raise ValueError('parameter update `s` not defined')
        c._x = np.array(self.x)
        if c.descent_subspace is None:
            if len(s) != self.n:
                raise ValueError(self._format('wrong size for parameter update `s`; `s` has size {}; full space is {}-dimensional', len(s), self.n))
            c._x += s
        else:
            if len(s) != c.descent_subspace_size:
                raise ValueError(self._format('parameter update `s` is not in descent subspace; `s` has size {}; descent subspace has size {}', len(s), c.descent_subspace_size))
            c._x[c.subspace_map] += s
        return c
    def push(self, _x=None):
        if _x is None:
            _x = self.__global__.x
        __x = self.x
        if __x is None:
            raise RuntimeError(self._format('no parameters are defined'))
        if __x.size == _x.size:
            if __x is not _x:
                _x[:] = __x
        elif __x.size == self.gradient_subspace_size:
            _x[self.gradient_subspace] = __x
        else:
            raise self._size_error

parallel.abc.VehicleJobStep.register(Component)


def _fun_args(fun, x0, component, covariate, gradient_subspace, descent_subspace,
        args, bounds, _sum, gradient_sum, gradient_covariate):
    _all = None
    if not callable(fun):
        raise TypeError('fun is not callable')
    if not isinstance(x0, np.ndarray):
        raise TypeError('x0 is not a numpy.ndarray')
    if not callable(component):
        if isinstance(component, int):
            if component == 0:
                component = lambda k: 0
            elif 0 < component:
                m = component
                def mk_component(m):
                    components = np.arange(m)
                    def _component(k):
                        _i = k % m
                        if _i == 0:
                            np.random.shuffle(components)
                        return components[_i]
                    return _component
                component = mk_component(m)
            else:
                raise ValueError('wrong number of components')
        else:
            raise TypeError('component is not callable')
    if not callable(covariate):
        raise TypeError('covariate is not callable')
    if gradient_subspace is None:
        gradient_subspace = lambda i: _all
    elif not callable(gradient_subspace):
        raise TypeError('gradient_subspace is not callable')
    if descent_subspace is None:
        descent_subspace = lambda i: _all
    elif not callable(descent_subspace):
        raise TypeError('descent_subspace is not callable')
    if not callable(_sum):
        raise TypeError('_sum is not callable')
    if gradient_sum is None:
        gradient_sum = _sum
    elif not callable(gradient_sum):
        raise TypeError('gradient_sum is not callable')

    # bounds
    if bounds:
        lower_bound, upper_bound = np.full_like(x0, -np.inf), np.full_like(x0, np.inf)
        any_lower_bound = any_upper_bound = False
        for i, _bs in enumerate(bounds):
            _lb, _ub = _bs
            if _lb not in (None, -np.inf, np.nan):
                any_lower_bound = True
                lower_bound[i] = _lb
            if _ub not in (None, np.inf, np.nan):
                any_upper_bound = True
                upper_bound[i] = _ub
        if not any_lower_bound:
            lower_bound = None
        if not any_upper_bound:
            upper_bound = None
        if any_lower_bound or any_upper_bound:
            bounds = (lower_bound, upper_bound)
        else:
            bounds = None
        #if bounds and c < 1:
        #    warnings.warn('c < 1; bounds may be violated')

    return component, gradient_subspace, descent_subspace, bounds, gradient_sum


def _ls_args(step_scale, ls_step_max, ls_iter_max, ls_armijo_max, ls_wolfe, newton):
    if step_scale <= 0 or 1 < step_scale:
        raise ValueError('expected: 0 < step_scale <= 1')
    ls_kwargs = {}
    if ls_step_max:
        ls_kwargs['step_max'] = ls_step_max
    if ls_iter_max:
        ls_kwargs['iter_max'] = ls_iter_max
    if ls_armijo_max:
        ls_kwargs['armijo_max'] = ls_armijo_max
    if ls_wolfe:
        try:
            c2, c3 = ls_wolfe
        except TypeError:
            c2 = c3 = ls_wolfe
        ls_kwargs['c3'] = c3
    elif False:#newton: # TEST
        c2 = .9
    else:
        c2 = .1
    ls_kwargs['c2'] = c2

    return ls_kwargs


module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter('%(message)s\n'))
module_logger.addHandler(_console)


def minimize_sparse_bfgs1(fun, x0, component, covariate, gradient_subspace, descent_subspace,
        args=(), bounds=None, _sum=np.sum, gradient_sum=None, gradient_covariate=None,
        memory=10, eps=1e-6, ftol=1e-6, gtol=1e-10, low_df_rate=.9, low_dg_rate=.9, step_scale=1.,
        max_iter=None, regul=None, regul_decay=1e-5, ls_kwargs={}, ls_regul=None, ls_step_max=None,
        ls_step_max_decay=None, ls_iter_max=None,
        ls_armijo_max=None, ls_wolfe=None, ls_failure_rate=.9, fix_ls=None, fix_ls_trigger=5,
        gradient_initial_step=1e-8, Component=Component,
        independent_components=False, newton=True, verbose=False, diagnosis=None,
        returns=(), max_runtime=None, update_timeout=None, **kwargs):
    r"""
    Let the objective function :math:`f(x) = \sum_{i \in C} f_{i}(x) \forall x \in \Theta`
    be a linear function of sparse components :math:`f_{i}` such that
    :math:`\forall j \notin G_{i}, \forall x \in \Theta, {\partial f_{i}}{\partial x_{j}}(x) = 0`.

    Let the components also covary sparsely:
    :math:`\forall i \in C,
    \exists C_{i} \subset C | i \in C_{i},
    \exists D_{i} \subset G_{i},
    \forall j \in D_{i},
    \forall x \in \Theta,
    \frac{\\partial f}{\partial x_{j}}(x) =
    \sum_{i' \in C_{i}} \frac{\partial f_{i'}}{\partial x_{j}}(x)`.

    We may additionally need that
    :math:`\forall i \in C, D_{i} = \bigcap_{i' \in C_{i}} G_{i'}`,
    :math:`\bigcup_{i \in C} D_{i} = J` and :math:`D_{i} \cap D_{j} = \emptyset \forall i, j \in C^{2}`
    with :math:`J` the indices of parameter vector :math:`x = \lvert x_{j}\rvert_{j \in J}`
    (to be checked).

    At iteration :math:`k`, let choose component :math:`i`
    and minimize :math:`f` wrt parameters :math:`\{x_{j} | j \in D_{i}\}`.

    Compute gradient :math:`g_{i}(x) = \lvert\frac{\partial f}{\partial x_{j}}(x)\rvert_{j} =
    \sum_{i' \in C_{i}} \lvert\frac{\partial f_{i'}}{\partial x_{j}}(x) \rvert_{j}`
    with (again) :math:`g_{i}(x)\rvert_{j} = 0 \forall j \notin G_{i}`.

    Perform a Wolfe line search along descent direction restricted to subspace :math:`D_{i}`.
    Update :math:`x` and compute the gradient again.

    The inverse Hessian matrix must also be updated twice: before and after the update.

    Arguments:

        fun (callable): takes a component index (`int`) and the parameter vector (`numpy.ndarray`)
            and returns a scalar `float`.

        x0 (numpy.ndarray): initial parameter vector.

        component (callable): takes an iteration index (`int`) and returns a component index (`int`);
            all the components are expected to be drawn exactly once per epoch;
            if `component` is an `int`, will be interpreted as the number of components.

        covariate (callable): takes a component index `i` (`int`) and returns a sequence of
            indices of the components that covary with `i`, including `i`.

        gradient_subspace (callable): takes a component index (`int`) and returns a sequence of
            indices of the parameters spanning the gradient subspace;
            if `None`, the gradient 'subspace' is the full space.

        descent_subspace (callable): takes a component index (`int`) and returns a sequence of
            indices of the parameters spanning the descent subspace;
            if `None`, the descent subspace equals to the gradient subspace.

        args (tuple or list): sequence of positional arguments to `fun`.

        bounds (tuple or list): sequence of pairs of (lower, upper) bounds on the parameters.

        _sum (callable): takes a `list` of `float` values and returns a `float`.

        gradient_sum (callable): replacement for `_sum` in the calculation of the gradient.

        gradient_covariate (callable): takes a parameter index (`int`) and returns a sequence
            of the components affected by this parameter.

        memory (int): number of memorized pairs of `H` updates in quasi-Newton mode.

        eps (float): initial scaling of the descent direction.

        ftol (float): maximum decrease in the local objective.

        gtol (float): maximum decrease in the projected gradient.

        low_df_rate (float): epoch-wise rate of low decrease in local objective (below `ftol`);
            at the end of an epoch, if this rate has been reached, then the iteration stops.

        low_dg_rate (float): epoch-wise rate of low decrease in the projected gradient (below `gtol`);
            at the end of an epoch, if this rate has been reached, then the iteration stops.

        step_scale (float): reduction factor on the line-searched step.

        max_iter (int): maximum number of iterations.

        regul (float): regularization trade-off coefficient for the L2-norm of the parameter vector.

        regul_decay (float): decay parameter for `regul`.

        ls_kwargs (dict): keyword arguments to :func:`wolfe_line_search`.

        ls_regul (float): regularization trade-off coefficient for the L2-norm of the parameter vector
            update.

        ls_step_max (float): maximum L-infinite norm of the parameter vector update;
            if `ls_step_max_decay` is defined, `ls_step_max` can be an (initial, final) `tuple` instead.

        ls_step_max_decay (float): decay parameter for `ls_step_max`.

        ls_iter_max (int): maximum number of linesearch iterations.

        ls_armijo_max (float): maximum expected decrease in the local objective; throttles the
            Armijo threshold.

        ls_wolfe (tuple): (`c2`, `c3`) pair for :func:`wolfe_line_search`.

        ls_failure_rate (float): epoch-wise rate of linesearch failure;
            at the end of an epoch, if this rate has been reached, then the iteration stops.

        fix_ls (callable): takes a component index (`int`) and the parameter vector (`numpy.ndarray`)
            to be modified inplace in the case of recurrent linesearch failures for this component.

        fix_ls_trigger (int): minimum number of iterations with successive linesearch failures on a
            given component for `fix_ls` to be triggered.

        Component (type): component constructor.

        independent_components (bool): whether to represent the local inverse Hessian submatrix
            for a component independently of the other components.

        newton (bool): quasi-Newton (BFGS) mode; if `False`, use vanilla gradient descent instead.

        verbose (bool or logging.Logger): verbose mode or logger.

        diagnosis (callable): function of the iteration number and the current and candidate-new
            components.

        returns (sequence of str): any subset of {'f', 'df', 'projg', 'err', 'ncalls', 'diagnosis'}
            or 'all'.

        xref (numpy.ndarray): reference final parameter vector; if defined, at each iteration,
            the L2-norm of the difference between this and the current parameter vector is evaluated
            and returned as attribute `err`; note that this computation may add quite some overhead.

    Returns:

        BFGSResult: final parameter vector.

    See also :class:`SparseFunction` and :func:`wolfe_line_search`.
    """
    # logging
    logger = None
    if verbose:
        if isinstance(verbose, logging.Logger):
            logger = verbose
            verbose = True
        else:
            logger = module_logger
    # initial checks
    component, gradient_subspace, descent_subspace, bounds, gradient_sum = \
            _fun_args(fun, x0, component, covariate, gradient_subspace, descent_subspace,
                    args, bounds, _sum, gradient_sum, gradient_covariate)

    # component
    __global__ = SparseFunction(x0, covariate, gradient_subspace, descent_subspace,
            eps, fun, _sum, args, regul, bounds, gradient_initial_step)
    extend_global(__global__, independent_components, memory, newton, gradient_covariate)
    C = _defaultdict(Component, __global__)

    sched = SBFGSScheduler(__global__, C, component,
            max_iter=max_iter, ftol=ftol, gtol=gtol, low_df_rate=low_df_rate, low_dg_rate=low_dg_rate,
            newton=newton, step_scale=step_scale, regul_decay=regul_decay,
            ls_iter_max=ls_iter_max, ls_armijo_max=ls_armijo_max, ls_wolfe=ls_wolfe,
            ls_regul=ls_regul, ls_step_max=ls_step_max, ls_step_max_decay=ls_step_max_decay,
            ls_failure_rate=ls_failure_rate, fix_ls=fix_ls, fix_ls_trigger=fix_ls_trigger,
            verbose=verbose, logger=logger, diagnosis=diagnosis,
            returns={'f', 'df', 'projg', 'err', 'ncalls', 'diagnosis'} if returns == 'all' else returns,
            max_runtime=max_runtime, update_timeout=update_timeout,
            **kwargs)
    sched.logger = logger

    if verbose:
        compact_logs = False
        if logger is module_logger:
            if 1 < sched.worker_count:
                _console.setFormatter(logging.Formatter('%(message)s'))
                compact_logs = True
            else:
                _console.setFormatter(logging.Formatter('%(message)s\n'))
        logger.debug('number of workers: {}'.format(sched.worker_count))
        t0 = time.time()

    sched.run()

    try:
        resolution = sched.resolution
    except AttributeError:
        if sched.iter_max_reached():
            resolution = 'MAXIMUM ITERATION REACHED'
        else:
            resolution = 'INTERRUPTED'

    x = __global__.x
    k = sched.k_eff
    ncalls = sched.ncalls
    f_history  = sched.f_history
    df_history = sched.df_history
    dg_history = sched.dg_history
    err_history = sched.err_history
    diagnoses = sched.diagnoses

    if verbose:
        cumt = time.time() - t0
        logger.info('{}           * * *\n\n{}\n'.format('\n' if compact_logs else '', resolution))
        minute = floor(cumt / 60.)
        second = cumt - minute * 60.
        if minute:
            logger.info('Elapsed time = {:.0f}m{:.3f}s'.format(minute, second))
        else:
            logger.info('Elapsed time = {:.3f}s'.format(second))

    try:
        H = {i: C[i].H for i in C}
    except AttributeError:
        # first epoch has not completed
        #traceback.print_exc()
        H = None
    return BFGSResult(x, H, resolution, k,
            f_history  if f_history  else None,
            df_history if df_history else None,
            dg_history if dg_history else None,
            cumt if verbose else None,
            err_history if err_history else None,
            diagnoses  if diagnoses  else None,
            ncalls     if ncalls     else None)


class SBFGSScheduler(parallel.Scheduler):
    def __init__(self, __global__, C, component, worker_count=None,
            name=None, args=(), kwargs={}, daemon=None,
            max_iter=None, ftol=None, gtol=None, low_df_rate=None, low_dg_rate=None,
            ls_failure_rate=None, fix_ls=None, fix_ls_trigger=None, returns={},
            max_runtime=None, update_timeout=None, **_kwargs):
        __global__.gtol = gtol
        parallel.Scheduler.__init__(self, __global__, C, worker_count=worker_count, iter_max=max_iter,
                name=name, args=args, kwargs=kwargs, daemon=daemon, max_runtime=max_runtime,
                task_timeout=update_timeout, **_kwargs)
        self.component = component
        self.ftol = ftol
        self.gtol = gtol
        self.low_df_rate = low_df_rate
        self.low_dg_rate = low_dg_rate
        self.ls_failure_rate = ls_failure_rate
        self.ls_failure_count = self.low_df_count = self.low_dg_count = 0
        self.fix_ls = fix_ls
        self.fix_ls_trigger = fix_ls_trigger
        self.recurrent_ls_failure_count = {}
        self.ncalls = [] if 'ncalls' in returns else None
        self.f_history = [] if 'f' in returns else None
        self.df_history = [] if 'df' in returns else None
        self.dg_history = [] if 'projg' in returns else None
        self.err_history = [] if 'err' in returns else None
        self.diagnoses = [] if 'diagnosis' in returns else None
        self.paused = dict()

    @property
    def worker(self):
        return SBFGSWorker

    def pause(self, i, t):
        self.paused[i] = t
        return len(self.paused) < len(self.task)

    def draw(self, k):
        i = self.component(k)
        assert i is not None
        try:
            t = self.paused[i]
        except KeyError:
            pass
        else:
            if t == 1:
                del self.paused[i]
            else:
                self.paused[i] = t - 1
            #print('skipping {}'.format(i))
            i = None
        return i

    def stop(self, k, i, status):

        # check for convergence based on f
        ncomponents = len(self.task)
        #assert 0 < ncomponents
        # note: the first epoch is caracterized by ncomponents == k + 1;
        #       the second epoch begins when ncomponents == k;
        #       as a consequence, new_epoch is always True during the first epoch;
        #       the epoch-wise criteria however are ignored during the first epoch.
        new_epoch = (k + 1) % ncomponents == 0

        if ncomponents <= k: # first epoch ignores the following criterion
            if status.get('ls_failure', None):
                self.ls_failure_count += 1
            # relocate the blocks below in Worker.target
            if False:#status.get('ls_failure', None):
                # try fixing problematic components
                if self.fix_ls:
                    _count = self.recurrent_ls_failure_count[i] = \
                            self.recurrent_ls_failure_count.get(i, 0) + 1
                    if self.fix_ls_trigger <= _count:
                        if verbose:
                            self.logger.debug(msg2(k+1, i, 'TRYING TO FIX THE RECURRENT FAILURE'))
                        c.push(x)
                        self.fix_ls(i, x)
                        c.pull(x)
                        c.H.drop()
            elif False:#self.fix_ls:
                try:
                    del self.recurrent_ls_failure_count[i]
                except KeyError:
                    pass
            if new_epoch:
                stop = self.ls_failure_rate * ncomponents <= self.ls_failure_count
                if stop:
                    self.resolution = 'LINE SEARCH FAILED'
                    ncalls = status.get('ncalls', 0)
                    if ncalls:
                        if self.ncalls is not None:
                            self.ncalls.append(ncalls)
                        if self.err_history is not None:
                            err = status.get('err', None)
                            if err is None:
                                assert not self.err_history
                            else:
                                self.err_history.append(err)
                    return stop

        try:
            f = status['f']
            df = status['df']
            ncalls = status['ncalls']
        except KeyError:
            pass
        else:
            if self.ncalls is not None:
                self.ncalls.append(ncalls)
            if self.f_history is not None:
                self.f_history.append(f)
            if self.df_history is not None:
                self.df_history.append(df)
            if self.err_history is not None:
                err = status.get('err', None)
                if err is None:
                    assert not self.err_history
                else:
                    self.err_history.append(err)
            if self.diagnoses is not None:
                diag = status.get('diagnosis', None)
                if diag is not None:
                    self.diagnoses.append(diag)
            if self.ftol is not None and ncomponents <= k: # first epoch ignores this criterion
                if df < self.ftol:
                    self.low_df_count += 1
                    if df <= .1 * self.ftol:
                        if not self.pause(i, 1):
                            self.resolution = 'CONVERGENCE: DELTA F <= FTOL/10'
                            return True
                #if new_epoch:
                    stop = self.low_df_rate * ncomponents <= self.low_df_count
                    if stop:
                        self.resolution = 'CONVERGENCE: DELTA F < FTOL'
                        return stop

        # check for convergence based on g
        try:
            proj = status['dg']
        except KeyError:
            pass
        else:
            if self.dg_history is not None:
                self.dg_history.append((i, proj))
            if self.gtol is not None and ncomponents <= k: # first epoch ignores this criterion
                if proj < self.gtol:
                    self.low_dg_count += 1
                    if proj <= .1 * self.gtol:
                        if not self.pause(i, max(self.paused.get(i, 0), 1)):
                            self.resolution = 'CONVERGENCE: PROJ G <= GTOL/10'
                            return True
                #if new_epoch:
                    stop = self.low_dg_rate * ncomponents <= self.low_dg_count
                    if stop:
                        self.resolution = 'CONVERGENCE: PROJ G < GTOL'
                        return stop

        if new_epoch:
            self.ls_failure_count = 0
            self.low_df_count = self.low_dg_count = len(self.paused)

        return parallel.Scheduler.stop(self, k, i, status)


class SBFGSWorker(parallel.Worker):
    def target(self, newton=None, step_scale=None, regul_decay=None,
            ls_iter_max=None, ls_armijo_max=None, ls_wolfe=None, ls_regul=None,
            ls_step_max=None, ls_step_max_decay=None,
            verbose=None, logger=None, diagnosis=None, xref=None):
        __global__ = self.workspace
        x = __global__.x
        regul = __global__.regul
        bounds = __global__.bounds
        gtol = __global__.gtol

        if verbose:
            msg0, msg1, msg2 = define_pprint()
            cumt = 0.
            t0 = time.time()
        if regul:
            regul0 = regul
        # linesearch
        ls_kwargs = _ls_args(step_scale, ls_step_max, ls_iter_max, ls_armijo_max, ls_wolfe, newton)
        # linesearch
        if ls_regul:
            ls_regul0 = ls_regul
        if ls_step_max_decay:
            try:
                initial_ls_step_max, final_ls_step_max = ls_step_max
            except TypeError:
                initial_ls_step_max = ls_step_max
                final_ls_step_max = initial_ls_step_max * .1

        try:
            while True:
                k, c = self.get_task()
                i = c.i
                info = dict()
                __global__.ncalls = 0
                try:

                    #assert x is __global__.x
                    if regul_decay:
                        _decay = max(1. - float(k) * regul_decay, 1e-10)
                        if regul:
                            __global__.regul = regul = regul0 * _decay
                        if ls_regul:
                            ls_regul = ls_regul0 * _decay
                    if ls_step_max_decay:
                        ls_step_max = initial_ls_step_max * max(1. - float(k) * ls_step_max_decay, final_ls_step_max)


                    # check for changes in the corresponding parameters since last iteration on component `i`
                    new_component = c.x is None
                    if new_component:
                        c.pull(x)
                    else:
                        c.push(x)
                        c._f = c._g = None # reset `f` and `g`

                    # estimate the local gradient
                    g = c.g # g_{k}

                    # retrieve the local inverse Hessian
                    H = c.H # H_{k}

                    # check positive definiteness
                    ones = np.ones_like(g)
                    oHo = np.dot(H.dot(ones), ones)
                    if np.any(oHo <= 0):
                        if verbose:
                            logger.debug(msg2(k+1, i, 'H not positive definite (k)'))
                        H.drop()

                    _k = 0
                    while True:
                        # define the descent direction
                        p = -H.dot(g) # p_{k} or q_{k}
                        p = c.in_descent_subspace(p)
                        g0 = c.in_descent_subspace(g)

                        # sanity check
                        local_convergence = np.all(g0 == 0)
                        if local_convergence:
                            break
                        if newton and 0 <= np.dot(p, g0):
                            if verbose:
                                logger.debug(msg2(k+1, i, 'PROJ G <= 0 (k)'))
                            H.drop()
                            p = -H.dot(g)
                            p = c.in_descent_subspace(p)

                        # get the parameter update
                        s = wolfe_line_search(
                                c.__f__, x, p, c.__g__, c.descent_subspace,
                                f0=c.f, g0=g0, bounds=bounds,
                                weight_regul=regul, step_regul=ls_regul,
                                return_resolution=verbose,
                                **ls_kwargs) # s_{k}

                        if verbose:
                            s, res = s
                        break

                        # if the linesearch failed, make `g` sparser
                        if s is None:
                            #if verbose:
                            #    logger.debug(msg2(k+1, i, res))
                            if not _k:
                                g = np.array(g)
                                _active = np.argsort(np.abs(g))
                            g[_active[_k]] = 0.
                            _k += 1
                        else:
                            break

                    info['f'] = c.f
                    if local_convergence:
                        if verbose:
                            logger.info(msg2(k+1, i, 'LOCAL CONVERGENCE MET'))
                        info['df'] = 0
                        continue

                    # sanity checks
                    if s is None:
                        if verbose:
                            #logger.info(msg2(k+1, i, 'LINE SEARCH FAILED'))
                            logger.info(msg2(k+1, i, 'LINE SEARCH FAILED ({})'.format(res)))#.upper())))
                        info['ls_failure'] = True
                        # undo any change in the working copy of the parameter vector (default in finally block)
                        continue
                    if np.all(s == 0):
                        if verbose:
                            logger.info(msg2(k+1, i, 'NULL UPDATE'))
                        info['df'] = 0
                        # undo any change in the working copy of the parameter vector (default in finally block)
                        continue

                    s *= step_scale

                    # update the parameter vector
                    c1 = c.commit(s) # x_{k+1} = x_{k} + s_{k}

                    info['df'] = c.f - c1.f

                    if diagnosis:
                        info['diagnosis'] = diagnosis(k, c, c1)

                    if gtol is None and not newton:
                        if verbose:
                            logger.info(msg1(k+1, i, c.f, c1.f))
                        c = c1 # 'push' the parameter update `c1`
                        continue

                    # estimate the gradient at x_{k+1}
                    h = c1.g # g_{k+1}
                    # sanity checks
                    if h is None:
                        if verbose:
                            logger.debug(msg2(k+1, i, 'GRADIENT CALCULATION FAILED (k+1)'))
                        # drop c1 (default in finally block)
                        continue
                    if np.allclose(g, h):
                        if verbose:
                            logger.debug(msg2(k+1, i, 'NO CHANGE IN THE GRADIENT'))
                        c = c1 # 'push' the parameter update...
                        continue # ...but do not update H

                    #
                    y = h - g # y_{k} = g_{k+1} - g_{k}
                    #
                    proj = np.dot(s, c.in_descent_subspace(y)) # s_{k}^{T} . y_{k}
                    info['dg'] = proj
                    if proj <= 0:
                        if verbose:
                            logger.debug(msg2(k+1, i, 'PROJ G <= 0 (k+1)'))
                        # either drop c1 (default in finally block)...
                        # ... or drop the inverse Hessian
                        c1.H.drop()
                        c = c1 # push `c1`
                        continue
                    elif verbose:
                        logger.info(msg1(k+1, i, c.f, c1.f, proj))

                    # 'push' the parameter update together with H update
                    H1 = (s, y, proj)
                    c1.H.update(*H1)
                    c = c1 # push `c1`

                finally:
                    c.push(x)
                    info['ncalls'] = __global__.ncalls
                    if xref is not None:
                        err = x - xref
                        err = np.dot(err, err)
                        info['err'] = err
                    self.push_update(c, info)

        except (SystemExit, KeyboardInterrupt):
            pass




def minimize_sparse_bfgs0(fun, x0, component, covariate, gradient_subspace, descent_subspace,
        args=(), bounds=None, _sum=np.sum, gradient_sum=None, gradient_covariate=None,
        memory=10, eps=1e-6, ftol=1e-6, gtol=1e-10, low_df_rate=.9, low_dg_rate=.9, step_scale=1.,
        max_iter=None, regul=None, regul_decay=1e-5, ls_regul=None, ls_step_max=None,
        ls_step_max_decay=None, ls_iter_max=None,
        ls_armijo_max=None, ls_wolfe=None, ls_failure_rate=.9, fix_ls=None, fix_ls_trigger=5,
        gradient_initial_step=1e-8,
        independent_components=True, newton=True, verbose=False):
    r"""
    Let the objective function :math:`f(x) = \sum_{i \in C} f_{i}(x) \forall x \in \Theta`
    be a linear function of sparse components :math:`f_{i}` such that
    :math:`\forall j \notin G_{i}, \forall x \in \Theta, {\partial f_{i}}{\partial x_{j}}(x) = 0`.

    Let the components also covary sparsely:
    :math:`\forall i \in C,
    \exists C_{i} \subset C | i \in C_{i},
    \exists D_{i} \subset G_{i},
    \forall j \in D_{i},
    \forall x \in \Theta,
    \frac{\\partial f}{\partial x_{j}}(x) =
    \sum_{i' \in C_{i}} \frac{\partial f_{i'}}{\partial x_{j}}(x)`.

    We may additionally need that
    :math:`\forall i \in C, D_{i} = \bigcap_{i' \in C_{i}} G_{i'}`,
    :math:`\bigcup_{i \in C} D_{i} = J` and :math:`D_{i} \cap D_{j} = \emptyset \\forall i, j \in C^{2}`
    with :math:`J` the indices of parameter vector :math:`x = \lvert x_{j}\rvert_{j \in J}`
    (to be checked).

    At iteration :math:`k`, let choose component :math:`i`
    and minimize :math:`f` wrt parameters :math:`\{x_{j} | j \in D_{i}\}`.

    Compute gradient :math:`g_{i}(x) = \lvert\frac{\partial f}{\partial x_{j}}(x)\rvert_{j} =
    \sum_{i' \in C_{i}} \lvert\frac{\partial f_{i'}}{\partial x_{j}}(x) \rvert_{j}`
    with (again) :math:`g_{i}(x)\rvert_{j} = 0 \forall j \notin G_{i}`.

    Perform a Wolfe line search along descent direction restricted to subspace :math:`D_{i}`.
    Update :math:`x` and compute the gradient again.

    The inverse Hessian matrix must also be updated twice: before and after the update.

    Arguments:

        fun (callable): takes a component index (`int`) and the parameter vector (`numpy.ndarray`)
            and returns a scalar `float`.

        x0 (numpy.ndarray): initial parameter vector.

        component (callable): takes an iteration index (`int`) and returns a component index (`int`);
            all the components are expected to be drawn exactly once per epoch;
            if `component` is an `int`, will be interpreted as the number of components.

        covariate (callable): takes a component index `i` (`int`) and returns a sequence of
            indices of the components that covary with `i`, including `i`.

        gradient_subspace (callable): takes a component index (`int`) and returns a sequence of
            indices of the parameters spanning the gradient subspace;
            if `None`, the gradient 'subspace' is the full space.

        descent_subspace (callable): takes a component index (`int`) and returns a sequence of
            indices of the parameters spanning the descent subspace;
            if `None`, the descent subspace equals to the gradient subspace.

        args (tuple or list): sequence of positional arguments to `fun`.

        bounds (tuple or list): sequence of pairs of (lower, upper) bounds on the parameters.

        _sum (callable): takes a `list` of `float` values and returns a `float`.

        gradient_sum (callable): replacement for `_sum` in the calculation of the gradient.

        gradient_covariate (callable): takes a parameter index (`int`) and returns a sequence
            of the components affected by this parameter.

        memory (int): number of memorized pairs of `H` updates in quasi-Newton mode.

        eps (float): initial scaling of the descent direction.

        ftol (float): maximum decrease in the local objective.

        gtol (float): maximum decrease in the projected gradient.

        low_df_rate (float): epoch-wise rate of low decrease in local objective (below `ftol`);
            at the end of an epoch, if this rate has been reached, then the iteration stops.

        low_dg_rate (float): epoch-wise rate of low decrease in the projected gradient (below `gtol`);
            at the end of an epoch, if this rate has been reached, then the iteration stops.

        step_scale (float): reduction factor on the line-searched step.

        max_iter (int): maximum number of iterations.

        regul (float): regularization trade-off coefficient for the L2-norm of the parameter vector.

        regul_decay (float): decay parameter for `regul`.

        ls_regul (float): regularization trade-off coefficient for the L2-norm of the parameter vector
            update.

        ls_step_max (float): maximum L-infinite norm of the parameter vector update;
            if `ls_step_max_decay` is defined, `ls_step_max` can be an (initial, final) `tuple` instead.

        ls_step_max_decay (float): decay parameter for `ls_step_max`.

        ls_iter_max (int): maximum number of linesearch iterations.

        ls_armijo_max (float): maximum expected decrease in the local objective; throttles the
            Armijo threshold.

        ls_wolfe (tuple): (`c2`, `c3`) pair for :func:`wolfe_line_search`.

        ls_failure_rate (float): epoch-wise rate of linesearch failure;
            at the end of an epoch, if this rate has been reached, then the iteration stops.

        fix_ls (callable): takes a component index (`int`) and the parameter vector (`numpy.ndarray`)
            to be modified inplace in the case of recurrent linesearch failures for this component.

        fix_ls_trigger (int): minimum number of iterations with successive linesearch failures on a
            given component for `fix_ls` to be triggered.

        independent_components (bool): whether to represent the local inverse Hessian submatrix
            for a component independently of the other components.

        newton (bool): quasi-Newton (BFGS) mode; if `False`, use vanilla gradient descent instead.

        verbose (bool or logging.Logger): verbose mode or logger.

    Returns:

        BFGSResult: final parameter vector.

    See also :class:`SparseFunction` and :func:`wolfe_line_search`.
    """
    # logging
    if verbose:
        if isinstance(verbose, logging.Logger):
            logger = verbose
            verbose = True
        else:
            logger = module_logger
    if verbose:
        msg0, msg1, msg2 = define_pprint()
        cumt = 0.
        t0 = time.time()
    # initial checks
    component, gradient_subspace, descent_subspace, bounds, gradient_sum = \
            _fun_args(fun, x0, component, covariate, gradient_subspace, descent_subspace,
                    args, bounds, _sum, gradient_sum, gradient_covariate)
    # working copy of the parameter vector
    x = x0
    n = x0.size

    # component
    __global__ = SparseFunction(x, covariate, gradient_subspace, descent_subspace,
            eps, fun, _sum, args, regul, bounds, gradient_initial_step)
    Component = define_component(__global__, independent_components, memory, newton, gradient_covariate)
    C = _defaultdict(Component, __global__)
    def mk_push(C):
        def _push(c):
            i = c.i
            C[i] = c
            c.push() # update the full parameter vector
        return _push
    push = mk_push(C)

    # linesearch
    ls_kwargs = _ls_args(step_scale, ls_step_max, ls_iter_max, ls_armijo_max, ls_wolfe, newton)
    if regul:
        regul0 = regul
    if ls_regul:
        ls_regul0 = ls_regul
    if ls_step_max_decay:
        try:
            initial_ls_step_max, final_ls_step_max = ls_step_max
        except TypeError:
            initial_ls_step_max = ls_step_max
            final_ls_step_max = initial_ls_step_max * .1

    # initial values
    ls_failure_count = low_df_count = low_dg_count = 0
    if fix_ls:
        recurrent_ls_failure_count = {}
    f_history = []
    df_history = []
    dg_history = []
    i_prev = None # previous component index
    k = 0 # iteration index

    while True:
        try:

            assert x is __global__.x
            #print(x[[0, 196, 197, 210]])
            if max_iter and max_iter <= k:
                resolution = 'MAXIMUM ITERATION REACHED'
                break
            if regul_decay:
                _decay = max(1. - float(k) * regul_decay, 1e-10)
                if regul:
                    __global__.regul = regul = regul0 * _decay
                if ls_regul:
                    ls_regul = ls_regul0 * _decay
            if ls_step_max_decay:
                ls_step_max = initial_ls_step_max * max(1. - float(k) * ls_step_max_decay, final_ls_step_max)

            # choose the target component (as a component index)
            i = component(k)
            # retrieve the corresponding component object
            c = C[i]

            # check for changes in the corresponding parameters since last iteration on component `i`
            if i != i_prev:
                new_component = c.x is None
                if not new_component:
                    # copy part of the component for initial H update
                    x_prev, g_prev = c.x, c.g
                # update with current parameters
                c.pull(x)
                if not new_component:
                    # update H with inter-iteration changes
                    if np.allclose(x_prev, c.x):
                        pass
                    elif newton:
                        s_ii = c.x - x_prev
                        y_ii = c.g - g_prev
                        proj_ii = np.dot(s_ii, y_ii)
                        if proj_ii <= 0:
                            if verbose:
                                logger.debug(msg2(k, i, 'PROJ G <= 0 (k-1)'))
                            c.H.drop()
                        else:
                            c.H.update(s_ii, y_ii, proj_ii)

            # estimate the local gradient
            g = c.g # g_{k}

            # retrieve the local inverse Hessian
            H = c.H # H_{k}

            _k = 0
            while True:
                # define the descent direction
                p = -H.dot(g) # p_{k} or q_{k}
                p = c.in_descent_subspace(p)
                g0 = c.in_descent_subspace(g)

                # sanity check
                local_convergence = np.all(g0 == 0)
                if local_convergence:
                    break
                if newton and 0 <= np.dot(p, g0):
                    if verbose:
                        logger.debug(msg2(k, i, 'PROJ G <= 0 (k)'))
                    H.drop()
                    p = -H.dot(g)
                    p = c.in_descent_subspace(p)

                # get the parameter update
                s = wolfe_line_search(
                        c.__f__, x, p, c.__g__, c.descent_subspace,
                        f0=c.f, g0=g0, bounds=bounds,
                        weight_regul=regul, step_regul=ls_regul,
                        #return_resolution=verbose,
                        **ls_kwargs) # s_{k}

                # if the linesearch failed, make `g` sparser
                if s is None:
                    if not _k:
                        g = np.array(g)
                        _active = np.argsort(np.abs(g))
                    g[_active[_k]] = 0.
                    _k += 1
                else:
                    break
            if local_convergence:
                if verbose:
                    logger.info(msg2(k, i, 'LOCAL CONVERGENCE MET'))
                continue

            # sanity checks
            ncomponents = len(C)
            #assert 0 < ncomponents
            # note: the first epoch is caracterized by ncomponents == k + 1;
            #       the second epoch begins when ncomponents == k;
            #       as a consequence, new_epoch is always True during the first epoch;
            #       the epoch-wise criteria however are ignored during the first epoch.
            new_epoch = (k + 1) % ncomponents == 0
            if s is None:
                if verbose:
                    logger.info(msg2(k, i, 'LINE SEARCH FAILED'))
                    #s, _res = s
                    #if s is None:
                    #    print(msg2(k, i, 'LINE SEARCH FAILED ({})'.format(_res.upper())))
            if ncomponents <= k: # first epoch ignores the following criterion
                if s is None:
                    ls_failure_count += 1
                    # try fixing problematic components
                    if fix_ls:
                        _count = recurrent_ls_failure_count[i] = recurrent_ls_failure_count.get(i, 0) + 1
                        if fix_ls_trigger <= _count:
                            if verbose:
                                logger.debug(msg2(k, i, 'TRYING TO FIX THE RECURRENT FAILURE'))
                            c.push(x)
                            fix_ls(i, x)
                            c.pull(x)
                            c.H.drop()
                elif fix_ls:
                    try:
                        del recurrent_ls_failure_count[i]
                    except KeyError:
                        pass
                if new_epoch:
                    if ls_failure_rate * ncomponents <= ls_failure_count:
                        resolution = 'LINE SEARCH FAILED'
                        break
            if s is None:
                # undo any change in the working copy of the parameter vector (default in finally block)
                continue
            if np.all(s == 0):
                if verbose:
                    logger.info(msg2(k, i, 'NULL UPDATE'))
                # undo any change in the working copy of the parameter vector (default in finally block)
                continue

            s *= step_scale

            # update the parameter vector
            c1 = c.commit(s) # x_{k+1} = x_{k} + s_{k}

            # check for convergence based on f
            if ftol is not None and ncomponents <= k: # first epoch ignores this criterion
                df = c.f - c1.f
                f_history.append(c.f)
                df_history.append(df)
                if df < ftol:
                    low_df_count += 1
                if new_epoch:
                    if low_df_rate * ncomponents <= low_df_count:
                        resolution = 'CONVERGENCE: DELTA F < FTOL'
                        c = c1 # 'push' the parameter update `c1`
                        break

            if gtol is None and not newton:
                if verbose:
                    logger.info(msg1(k, i, c.f, c1.f))
                c = c1 # 'push' the parameter update `c1`
                continue

            # estimate the gradient at x_{k+1}
            h = c1.g # g_{k+1}
            # sanity checks
            if h is None:
                if verbose:
                    logger.debug(msg2(k, i, 'GRADIENT CALCULATION FAILED (k+1)'))
                # drop c1 (default in finally block)
                continue
            if np.allclose(g, h):
                if verbose:
                    logger.debug(msg2(k, i, 'NO CHANGE IN THE GRADIENT'))
                c = c1 # 'push' the parameter update...
                continue # ...but do not update H

            #
            y = h - g # y_{k} = g_{k+1} - g_{k}
            #
            proj = np.dot(s, c.in_descent_subspace(y)) # s_{k}^{T} . y_{k}
            if proj <= 0:
                if verbose:
                    logger.debug(msg2(k, i, 'PROJ G <= 0 (k+1)'))
                # either drop c1 (default in finally block)...
                # ... or drop the inverse Hessian
                c1.H.drop()
                c = c1 # push `c1`
                continue
            elif verbose:
                logger.info(msg1(k, i, c.f, c1.f, proj))

            # check for convergence based on g
            if gtol is not None and ncomponents <= k: # first epoch ignores this criterion
                dg_history.append(proj)
                if proj < gtol:
                    low_dg_count += 1
                if new_epoch:
                    if low_dg_rate * ncomponents <= low_dg_count:
                        resolution = 'CONVERGENCE: PROJ G < GTOL'
                        c = c1 # push `c1`
                        break

            # 'push' the parameter update together with H update
            H1 = (s, y, proj)
            c1.H.update(*H1)
            c = c1 # push `c1`

        except KeyboardInterrupt:
            resolution = 'INTERRUPTED'
            break

        finally:
            if new_epoch:
                ls_failure_count = low_df_count = low_dg_count = 0
            # refresh the working copy of the parameter vector
            push(c)
            # loop
            i_prev = i
            k += 1

    if verbose:
        cumt += time.time() - t0
        logger.info('           * * *\n\n{}\n'.format(resolution))
        minute = floor(cumt / 60.)
        second = cumt - minute * 60.
        if minute:
            logger.info('Elapsed time = {:d}m{:.3f}s\n'.format(minute, second))
        else:
            logger.info('Elapsed time = {:.3f}s\n'.format(second))
    H = {i: C[i].H for i in C}
    return BFGSResult(x, H, resolution, k,
            f_history if f_history else None,
            df_history if df_history else None,
            dg_history if dg_history else None,
            cumt if verbose else None, None, None, None)


def sparse_grad(fun, x, active_i, active_j, args=(), _sum=np.sum, regul=None, bounds=None, h0=1e-8):
    """
    Compute the derivative of a function.
    """
    SAFE, CON = 2., 1.4
    CON2 = CON * CON
    H = h0 * ((1./CON) ** np.arange(10))
    if active_j is None:
        active_j = range(x.size)
    if bounds is None:
        lower = upper = None
    else:
        lower, upper = bounds
    if not regul:
        penalty = 0.
    total_grad, partial_grad = [], {}
    any_ok, failures = False, []
    a = np.zeros((H.size, H.size), dtype=float)
    for j in active_j:
        if callable(active_i):
            I = active_i(j)
        else:
            I = active_i
        total_grad_j, err = None, np.inf
        xj = x[j] # keep copy
        try:
            for u, h in enumerate(H):
                try:
                    #
                    x[j] = xj + h
                    if upper is not None:
                        x[j] = min(x[j], upper[j])
                    f_a = np.array([ fun(i, x, *args) for i in I ])
                    #
                    x[j] = xj - h
                    if lower is not None:
                        x[j] = max(lower[j], x[j])
                    f_b = np.array([ fun(i, x, *args) for i in I ])
                    #
                    partial_grad_j = (f_a - f_b) / (2. * h)
                    if regul:
                        penalty = regul * 2. * xj
                    a[u,0] = a_up = _sum(partial_grad_j) + penalty
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    #raise
                    #traceback.print_exc()
                    break
                if u == 0:
                    continue
                fac = CON2
                for v in range(1, a.shape[1]):
                    a_pp = a[u-1,v-1] # pp stands for [previous, previous]
                    a[u,v] = a_uv = (a_up * fac - a_pp) / (fac - 1.)
                    fac *= CON2
                    err_uv = max(abs(a_uv - a_up), abs(a_uv - a_pp))
                    if err_uv <= err:
                        err, total_grad_j = err_uv, a_uv
                    a_up = a_uv
                if SAFE * err <= abs(a[u,u] - a[u-1,u-1]):
                    break
            if total_grad_j is None:
                failures.append(j)
                total_grad_j = 0.
            else:
                any_ok = True
                partial_grad[j] = (I, np.array(partial_grad_j))
            total_grad.append(total_grad_j)
        finally:
            x[j] = xj # restore
    if any_ok:
        if failures:
            module_logger.warning('sparse_grad failed at column(s): {}'.format(', '.join([ str(i) for i in failures ])))
        total_grad = np.array(total_grad)
        return total_grad, partial_grad
    else:
        module_logger.warning('sparse_grad failed at all the columns')
        return None, None

minimize_sparse_bfgs = minimize_sparse_bfgs1

__all__ = [ 'BFGSResult', 'minimize_sparse_bfgs', 'minimize_sparse_bfgs0', 'minimize_sparse_bfgs1', 'SparseFunction', 'wolfe_line_search', 'sparse_grad' ]

