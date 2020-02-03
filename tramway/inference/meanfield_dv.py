# -*- coding: utf-8 -*-

# Copyright © 2019-2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
from .gradient import get_grad_kwargs, setup_with_grad_arguments
from .meanfield_friction_drift import Meanfield, dot
from math import *
import numpy as np
import pandas as pd
from collections import OrderedDict
from warnings import warn


setup = {'name': ('meanfield.dv', 'meanfield diffusivity,potential'),
        'provides': ('dv', 'diffusivity,potential'),
        'arguments': OrderedDict((
            #('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
            ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
            ('potential_prior',     ('-v', dict(type=float, help='prior on the potential energy'))),
            ('diffusivity_time_prior',   ('--time-d', dict(type=float, help='prior on the temporal variations of the diffusivity'))),
            ('potential_time_prior',     ('--time-v', dict(type=float, help='prior on the temporal variations of the potential'))),
            ('verbose',         ()))),
        'default_grad':     'onesided_gradient',
        'default_rgrad':    'delta0'}
setup_with_grad_arguments(setup)


def _nan(x):
    """modifies in-place and returns `x` itself."""
    x[np.isnan(x)] = 0.
    return x

class MeanfieldPotential(Meanfield):
    def __init__(self, cells, dt=None,
            diffusivity_spatial_prior=None, diffusivity_time_prior=None,
            friction_spatial_prior=None, friction_time_prior=None,
            potential_spatial_prior=None, potential_time_prior=None,
            _inherit=False, **kwargs):
        Meanfield.__init__(self, cells, dt,
                diffusivity_spatial_prior, diffusivity_time_prior,
                friction_spatial_prior, friction_time_prior,
                _inherit=True)
        self.add_feature('V')

        dt = self.dt
        self.V_spatial_prior = potential_spatial_prior * dt \
                if potential_spatial_prior else None
        self.V_time_prior = potential_time_prior * dt \
                if potential_time_prior else None

        if kwargs.get('grad_selection_angle', None) is None:
            kwargs['grad_selection_angle'] = .5
        grad_kwargs = get_grad_kwargs(kwargs)
        #sides = grad_kwargs.get('side', '<>')
        self.gradient_sides = sides = grad_kwargs.get('side', '>')

        n = self.n
        dr = self.dr
        dtype = self.dtype
        index, reverse_index = self.index, self.reverse_index

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

        B, Bstar, B2 = {s: [] for s in sides}, {s: [] for s in sides}, {s: [] for s in sides}
        C_neighbours, C = [], []
        Ct_neighbours, Ct = [], []
        regions = []

        for k, i in enumerate(index):
            # spatial gradients
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
            # spatial smoothing
            _neighbours = cells.neighbours(i)
            if _neighbours.size:
                _dr = np.stack([ cells[j].center for j in _neighbours ], axis=0) - cells[i].center[np.newaxis,:]
                _C = 1. / np.sum(_dr*_dr,axis=1) / len(_dr)
                _neighbours = reverse_index[_neighbours]
            else:
                _C = _neighbours = None
            C.append(_C)
            C_neighbours.append(_neighbours)
            regions.append(np.r_[k,_neighbours])
            # temporal smoothing
            _neighbours = cells.time_neighbours(i)
            if _neighbours.size:
                _dt = np.array([ cells[j].center_t for j in _neighbours ]) - cells[i].center_t
                _Ct = 1. / (_dt*_dt) / len(_dt)
                _neighbours = reverse_index[_neighbours]
            else:
                _Ct = _neighbours = None
            Ct.append(_Ct)
            Ct_neighbours.append(_neighbours)

        self.regions = regions
        self.Bstar = {s: np.vstack(Bstar[s]) for s in Bstar}
        self.B2 = {s: np.array(B2[s]) for s in B2}
        self.B = {s: np.vstack(B[s]) for s in B}

        self.C, self.Ct = list(zip(C_neighbours, C)), list(zip(Ct_neighbours, Ct))

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
        return not (self.V_spatial_prior is None and self.V_time_prior is None)

    def set_V_regularizer(self, aV):
        self.set_regularizer('V', aV)

    def regularize_V(self, aV, bV, AV=None, BV=None, oneshot=True):
        return self.regularize('V', aV, bV, AV, BV, oneshot)

    def spatial_penalties(self, a):
        return self.background(a, self.C)
    def temporal_penalties(self, a):
        return self.background(a, self.Ct)
    def background(self, a, C, sign=None):
        try:
            f_ = self.f_
        except AttributeError:
            f_ = lambda x: x
        return np.array([
                0. if neighbours is None else np.dot(f_(a[neighbours], sign), _C)
                for neighbours, _C in C
            ])


def merge_a(_as, sign=None, eps=None):
    _as = np.stack(_as, axis=-1)
    _undefined = np.isnan(_as)
    _as[_undefined] = 0.
    #if sign:
    #    _undefined |= (_as < eps) | (1./eps < _as)
    #else:
    #    _undefined |= (_as < -1./eps) | (1./eps < _as)
    #_as[_undefined] = 0.
    _n = np.sum(~_undefined, axis=1)
    _undefined = _n == 0
    _n[_undefined] = 1. # any value other than 0
    _a = np.sum(_as, axis=1) / _n
    _a[_undefined] = np.nan
    return _a
def merge_b(_bs, eps=None):
    _bs = np.stack(_bs, axis=-1)
    _undefined = np.isnan(_bs)# | np.isinf(_bs)
    #_bs[_undefined] = 0.
    #_undefined |= (_bs < eps) | (1./eps < _bs)
    _bs[_undefined] = np.inf
    _sum_inv = np.sum(1. / _bs, axis=1)
    _n = np.sum(~_undefined, axis=1)
    _defined = 0 < _n
    _sum_inv, _n = _sum_inv[_defined], _n[_defined]
    _b = np.full(_bs.shape[0], np.nan, dtype=_bs.dtype)
    _b[_defined] = (_n * _n) / _sum_inv
    return _b


class DiffusivityPotential(MeanfieldPotential):
    def __init__(self, cells, dt=None,
            diffusivity_spatial_prior=None, diffusivity_time_prior=None,
            potential_spatial_prior=None, potential_time_prior=None,
            _inherit=False, **kwargs):
        for friction_arg in ('friction_prior', 'friction_spatial_prior', 'friction_time_prior'):
            try:
                kwargs.pop(friction_arg)
            except KeyError:
                pass
            else:
                warn('ignoring {} argument'.format(friction_arg), RuntimeWarning)
        MeanfieldPotential.__init__(self, cells, dt,
                diffusivity_spatial_prior=diffusivity_spatial_prior,
                diffusivity_time_prior=diffusivity_time_prior,
                potential_spatial_prior=potential_spatial_prior,
                potential_time_prior=potential_time_prior,
                _inherit=_inherit, **kwargs)
        dt = self.dt
        index = self.index
        #while True:
        #    ok, all_ok = np.ones(index.size, dtype=bool), True
        #    for i in index:
        #        cells.grad(i, dt, reverse_index, **grad_kwargs)
        #        k, _neighbours_lt, _, _neighbours_gt, _ = cells[i].cache['onesided_gradient']
        #        if sides == '<':
        #            if _neighbours_lt is None:
        #                ok[k] = all_ok = False
        #        elif sides == '>':
        #            if _neighbours_gt is None:
        #                ok[k] = all_ok = False
        #        elif _neighbours_lt is None and _neighbours_gt is None:
        #            ok[k] = all_ok = False
        #    #print(np.sum(ok))
        #    if all_ok:
        #        break
        #    else:
        #        reverse_index[index] -= np.cumsum(~ok)
        #        reverse_index[index[~ok]] = -1
        #        index, n, dt = index[ok], n[ok], dt[ok]
        #    cells.clear_caches()
        if index.size == 0:
            raise ValueError('no valid cell')


def infer_meanfield_DV(cells, diffusivity_spatial_prior=None, potential_spatial_prior=None,
        diffusivity_time_prior=None, potential_time_prior=None,
        diffusivity_prior=None, potential_prior=None,
        diffusion_prior=None, diffusion_spatial_prior=None, diffusion_time_prior=None,
        dt=None, tol=1e-6, eps=1e-3, verbose=False, aD_formula='exact', **kwargs):
    """
    """
    #localization_error = cells.get_localization_error(kwargs, 0.03, True)

    if diffusivity_prior is None:
        diffusivity_prior = diffusion_prior
    if diffusivity_spatial_prior is None:
        if diffusion_spatial_prior is None:
            diffusivity_spatial_prior = diffusivity_prior
        else:
            diffusivity_spatial_prior = diffusion_spatial_prior
    if potential_spatial_prior is None:
        potential_spatial_prior = potential_prior
    if diffusivity_time_prior is None:
        diffusivity_time_prior = diffusion_time_prior

    mf = DiffusivityPotential(cells, dt,
            diffusivity_spatial_prior, diffusivity_time_prior,
            potential_spatial_prior, potential_time_prior,
            **kwargs)

    n = mf.n
    dt = mf.dt
    mean_dr = mf.dr
    mean_dr2 = mf.dr2
    sum_dr2 = n * mf.dr2
    index = mf.index

    B, Bstar, B2 = mf.B, mf.Bstar, mf.B2

    try:
        if False:#compatibility:
            raise Exception # skip to the except block
        volume = [ cells[i].volume for i in index ]
    except:
        V_initial = -np.log(n / np.max(n))
    else:
        density = n / np.array([ np.inf if v is None else v for v in volume ])
        density[density == 0] = np.min(density[0 < density])
        V_initial = np.log(np.max(density)) - np.log(density)

    def f_(_x, sign=None, invalid=0., inplace=True):
        _valid = ~(np.isnan(_x) | np.isinf(_x))
        #if sign:
        #    _valid[_valid] = (eps < _x[_valid]) & (_x[_valid] < 1./eps)
        #else:
        #    _valid[_valid] = (-1./eps < _x[_valid]) & (_x[_valid] < 1./eps)
        if verbose and not np.all(_valid):
            if _x.shape[1:]:
                _is,_ = (~_valid).nonzero()
            else:
                _is, = (~_valid).nonzero()
            warn('invalid value(s) encountered at indices: {}'.format(_is), RuntimeWarning)
        __x = _x if inplace else np.array(_x) # copy
        __x[~_valid] = invalid
        return __x
    mf.f_ = f_

    # constants
    approx_aD_constant_factor = 1. / (4. * dt)
    bD_constant_factor = n / (2. * dt)
    exact_aD_constant_factor_1 = mean_dr2 / (2. * dt)
    exact_aD_constant_factor_2 = exact_aD_constant_factor_1 / (2. * dt)
    bV_constant_factor = {s: B2[s] * bD_constant_factor for s in B2}

    if aD_formula.startswith('exact'):
        exact_aD_formula = True
        approximate_aD_formula = aD_formula.endswith('+')
    elif aD_formula.startswith('approx'):
        approximate_aD_formula = True
        exact_aD_formula = aD_formula.endswith('+')
    else:
        raise ValueError("aD_formula='{}' not supported".format(aD_formula))

    # initial values
    chi2_over_n = mean_dr2 - np.sum(mean_dr * mean_dr, axis=1)
    D = approx_aD_constant_factor * chi2_over_n#mean_dr2
    #V = np.zeros_like(D)
    V = V_initial

    fixed_neighbours = False
    slow_neighbours = True
    first_step = True

    resolution = None
    try:
        aVs, bVs, aDs, bDs = [], [], [], []
        grad_aVs, grad_aV2s = [], []
        for s in mf.gradient_sides:

            # 1. adjust D and V jointly with no regularization
            aD, aV = D, V

            i = 0
            neg_L = neg_2L = np.inf
            while True:
                # depends on the implementation
                neg_L_prev = neg_L
                neg_2L_prev = neg_2L

                # new aV
                if i==0 or not fixed_neighbours:
                    gradV_plus_VB = _nan(mf.diff_grad(aV, s))
                aV = np.sum(  Bstar[s] * (gradV_plus_VB + mean_dr / aD[:,np.newaxis])  ,axis=1)
                if slow_neighbours:
                    grad_aV = gradV_plus_VB - aV[:,np.newaxis] * B[s]
                else:
                    grad_aV = mf.grad(aV, s)
                grad_aV2 = np.nansum(grad_aV * grad_aV, axis=1)

                # new aD
                if approximate_aD_formula:
                    aD2 = aD * aD
                    chi2_over_n = mean_dr2 + 2 * aD * np.sum(mean_dr * _nan(grad_aV), axis=1) + aD2 * grad_aV2
                    approx_aD = approx_aD_constant_factor * chi2_over_n
                if exact_aD_formula:
                    sqrt_discr = np.sqrt( 1. + exact_aD_constant_factor_2 * grad_aV2 )
                    exact_aD = exact_aD_constant_factor_1 / (1. + sqrt_discr)
                aD = approx_aD if aD_formula.startswith('approx') else exact_aD

                # new bV and bD
                bV = aD * bV_constant_factor[s]
                inv_aD = f_( 1./ aD )
                inv_aD2 = inv_aD * inv_aD
                bD = n * inv_aD2 + bD_constant_factor * grad_aV2 * inv_aD

                if verbose:
                    if aD_formula.endswith('+'):
                        print('[{}|{}] max aD difference: {}'.format(i, s, np.max(np.abs(exact_aD-approx_aD))))
                    print('[{}|{}] max values for: ||gradV|| {:.2f} aV:bV {:.2f}:{:.2f} aD:bD {:.2f}:{:.2f}'.format(i, s, np.sqrt(np.nanmax(grad_aV2)), np.nanmax(np.abs(aV)), np.nanmax(bV), np.nanmax(aD), np.nanmax(bD)))
                if not first_step:
                    break

                if False: # -L(q)
                    inv_aD2_bD = inv_aD2 / bD
                    inv_aD3_bD = inv_aD2_bD * inv_aD

                    B2_over_bV = B2[s] / bV
                    B2_over_bV_star = np.array([ np.sum(B2_over_bV[_region]) for _region in mf.regions ])
                    log_bV = f_( np.log(bV) )
                    log_bV_star = np.sum([ np.sum(log_bV[_region]) for _region in mf.regions ])
                    neg_2L = dot(n, \
                                f_( 2 * np.log(aD) ) - inv_aD2_bD + \
                                1./ (2 * dt) * ( \
                                    (inv_aD + inv_aD3_bD) * mean_dr2 + \
                                    2 * np.sum(grad_aV * mean_dr, axis=1) + \
                                    aD * (grad_aV2 + B2_over_bV_star) \
                                    )) + \
                            np.sum(f_( np.log(bD) )) + log_bV_star

                    if verbose:
                        print('[{}|{}] approx -L(q): {}'.format(i, s, neg_2L))
                    # stopping criterion
                    if neg_2L_prev - neg_2L < tol:
                        break
                else: # -logP
                    neg_L = dot(n, \
                                f_(np.log( aD * dt )) + \
                                1./(4* dt) * ( \
                                    f_( mean_dr2 / aD ) + \
                                    2* np.sum(mean_dr * grad_aV, axis=1) + \
                                    grad_aV2 * aD \
                                ))

                    if verbose:
                        print('[{}|{}] approx -logP: {}'.format(i, s, neg_L))
                    # stopping criterion
                    if np.abs(neg_L_prev - neg_L) == 0:
                        # important note: the effective potential values are scaled to their proper range;
                        #                 this happens from a corner to the opposite corner and generates a propagating frontier
                        #                 between two plateaus;
                        #                 when this diagonal frontier includes many bins, the cost function begins to increase;
                        #                 this should not stop the algorithm to update more effective potential parameters;
                        #                 only full convergence is a reliable stopping criterion here
                        break
                i += 1

            # 2. regularize
            if mf.regularize_diffusion or mf.regularize_potential:
                if mf.regularize_diffusion:
                    AD, BD = mf.regularize_D(aD, bD)
                else:
                    AD, BD = aD, bD
                if mf.regularize_potential:
                    AV, BV = aV, bV

                    # aD or AD?
                    dr_over_D = mean_dr / AD[:,np.newaxis]

                    new_epoch = True
                    i = 0
                    neg_L = np.inf
                    while True:
                        neg_L_prev = neg_L

                        if i==0 or not fixed_neighbours:
                            gradV_plus_VB = _nan(mf.diff_grad(AV, s))
                            aV = np.sum(  Bstar[s] * (gradV_plus_VB + dr_over_D)  ,axis=1)
                            AV = BV = None

                        AV, BV = mf.regularize_V(aV, bV, AV, BV)

                        #neg_L = mf.regularized_cost('V', AV, aV, bV) # meanfield cost

                        grad_aV = gradV_plus_VB - AV[:,np.newaxis] * B[s] # constant neighbours
                        #grad_aV = mf.diff_grad(AV, s) - aV[:,np.newaxis] * B[s] # neighbours only vary
                        #grad_aV = mf.grad(AV, s) # all vary
                        grad_aV2 = np.nansum(grad_aV * grad_aV, axis=1)

                        # include only the terms that depend on aV and bV (or AV and BV)
                        if False:
                            if i==0: # BV actually does not vary
                                B2_over_bV = _nan( B2[s] / BV )
                                B2_over_bV_star = np.array([ np.sum(B2_over_bV[_region]) for _region in mf.regions ])
                                log_bV = f_( np.log(BV) )
                                log_bV_star = np.sum([ np.sum(log_bV[_region]) for _region in mf.regions ]) / 2
                            neg_L = dot(n / (4 * dt), \
                                        2 * np.sum(mean_dr * grad_aV, axis=1) + \
                                        (1./AD + 1./(AD**3 * BD)) * (grad_aV2 + B2_over_bV_star)) + \
                                    log_bV_star + \
                                    mf.neg_log_prior('V', aV, AV)

                            if verbose:
                                print('[{}] approx -L(q): {}'.format(i, neg_L))
                        else:
                            neg_L = dot(n / (2* dt), \
                                        np.sum(mean_dr * grad_aV, axis=1) + grad_aV2 * AD / 2 ) + \
                                    mf.neg_log_prior('V', AV)
                            if verbose:
                                print('[{}] approx -logP: {}'.format(i, neg_L))

                        # stopping criterion
                        if np.isnan(neg_L) or np.isinf(neg_L):
                            break
                        elif np.abs(neg_L_prev - neg_L) < tol:
                            if new_epoch:
                                if neg_L_prev - neg_L < -tol:
                                    # one step back
                                    AV, BV = aV, bV
                                if verbose:
                                    print('convergence')
                                break
                            else:
                                if verbose:
                                    print('new epoch')
                                new_epoch = True
                                aV, bV = AV, BV
                        #else:
                        #    new_epoch = False

                        i += 1
                else:
                    AV, BV = aV, bV


                aD, bD, aV, bV = AD, BD, AV, BV

            aVs.append(aV)
            bVs.append(bV)
            aDs.append(aD)
            bDs.append(bD)

            grad_aVs.append(grad_aV)
            grad_aV2s.append(grad_aV2)

        # merge
        aV = merge_a(aVs)
        bV = merge_b(bVs)
        aD = merge_a(aDs, '>')
        bD = merge_b(bDs)

        neg_L, _ctr = 0., 0
        for _D,_V,_gradV,_gradV2 in zip(aDs, aVs, grad_aVs, grad_aV2s):
            neg_L += np.dot(n, \
                        f_(np.log( 4*pi* _D * dt )) + \
                        1./(4* dt) * ( \
                            mean_dr2 / _D + \
                            2* np.sum(mean_dr * _gradV, axis=1) + \
                            _gradV2 * _D \
                        ))
            if mf.regularize_diffusion or mf.regularize_potential:
                neg_L += mf.neg_log_prior('V', _V)
            _ctr += 1
        neg_L /= _ctr

    except KeyboardInterrupt:
        resolution = 'INTERRUPTED'
        if verbose:
            print('interrupted')
            print('D={}'.format(aD))
            print('V={}'.format(aV))
            try:
                print('L={}'.format(-neg_L))
            except:
                print('L={}'.format(-neg_2L))
        pass


    D = aD
    V = aV / dt

    if np.any(V<0):
        V -= np.nanmin(V)

    DV = pd.DataFrame(np.stack((D, V), axis=1), index=index, \
        columns=['diffusivity', 'potential'])

    # derivate the forces
    #index_, F = [], []
    #for i in index:
    #    gradV = mf.local_grad(i, V, mf.gradient_sides[0])#cells.grad(i, V, reverse_index, **grad_kwargs)
    #    if gradV is not None:
    #        index_.append(i)
    #        F.append(-gradV)
    #if F:
    #    F = pd.DataFrame(np.stack(F, axis=0), index=index_, \
    #        columns=[ 'force ' + col for col in cells.space_cols ])
    #else:
    #    warn('not any cell is suitable for evaluating the local force', RuntimeWarning)
    #    F = pd.DataFrame(np.zeros((0, len(cells.space_cols)), dtype=V.dtype), \
    #        columns=[ 'force ' + col for col in cells.space_cols ])
    #DVF = DV.join(F)

    info = dict(resolution=resolution, log_likelyhood=-neg_L)

    return DV, info


__all__ = ['setup', 'MeanfieldPotential', 'DiffusivityPotential', 'infer_meanfield_DV']

