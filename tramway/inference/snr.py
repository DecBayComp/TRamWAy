# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core.namedcolumns import *
from .base import *
from .gradient import *
import numpy as np
import pandas as pd
from collections import OrderedDict
import itertools
import warnings

_extensions = [ 'n', 'V_prior', 'V', 'zeta_total', 'zeta_spurious' ]


setup = {'name': ('snr', 'd.snr'),
    'arguments': OrderedDict((
        ('localization_error',  ('-e', dict(type=float, help='localization precision (see also sigma; default is 0.03)'))),
        ('diffusivity_prior',   ('-d', dict(type=float, help='prior on the diffusivity'))),
        ('jeffreys_prior',      ('-j', dict(action='store_true', help="Jeffreys' prior"))),
        ('min_diffusivity',     dict(type=float, help='minimum diffusivity value allowed')),
        ('max_iter',        dict(type=int, help='maximum number of iterations')))),
    'cell_sampling': 'group'}
setup_with_grad_arguments(setup)


def infer_snr(cells, **kwargs):
    '''
    Add signal-to-noise ratio related variables useful for Bayes factor calculation.

    If variable *diffusivity* is not already available, it is inferred using D mode,
    with or without spatial smoothing.
    '''
    for i in cells:
        any_cell = cells[i]
        break
    if hasattr(any_cell, 'diffusivity'):
        try:
            variables = any_cell.__slots__
        except AttributeError:
            warnings.warn('attribute `__slots__` missing in cells', RuntimeWarning)
            variables = ('diffusivity',)
        else:
            if not isinstance(variables, (tuple, list)):
                variables = (variables,)
            if 'diffusivity' not in variables:
                warnings.warn('`__slots__` does not refer to diffusivity in cells', RuntimeWarning)
                variables = list(variables)
                variables.append('diffusivity')
        index = []
        _maps = {}
        _ix = {}
        for v in variables:
            _maps[v] = []
            _ix[v] = []
        for i in cells:
            cell = cells[i]
            # sanity check similar to :func:`inference.base.smooth_infer_init`
            try:
                adjacent = cells.adjacency.indices[cells.adjacency.indptr[i]:cells.adjacency.indptr[i+1]]
                adjacent = [ c for c in adjacent if cells[c] ]
                if not adjacent:
                    continue
            except ValueError:
                continue
            #
            index.append(i)
            for v in variables:
                try:
                    _val = getattr(cell, v)
                except AttributeError:
                    _val = None
                if _val is None:
                    continue
                _ix[v].append(i)
                _maps[v].append(_val)
        index = np.array(index)
        _cols = []
        for v in variables:
            if not _maps[v]:
                continue
        _cols = list(itertools.chain(_cols))
        maps = None
        for v in variables:
            if not _maps[v]:
                continue
            _data = np.array(_maps[v])
            if not _data.shape[1:]:
                _data = _data[:,np.newaxis]
            _cols = expandcoord(v, _data.shape[1])
            _map = pd.DataFrame(_data, columns=_cols, index=_ix[v])
            if maps is None:
                maps = _map
            else:
                maps = maps.join(_map)
    else:
        import tramway.helper as helper
        if kwargs.get('diffusivity_prior', None):
            maps = helper.infer(cells, 'smooth.d', **kwargs)
        else:
            maps = helper.infer(cells, 'd', **kwargs)
    maps = add_snr_extensions(cells, maps, get_grad_kwargs(**kwargs))
    return maps


def add_snr_extensions(cells, maps=None, grad_kwargs={}, _zeta_spurious=True):
    '''
    Infer signal-to-noise ratio related variables useful for Bayes factor calculation.

    Diffusion is read from `maps` (if available) or `cells`.
    Variable/attribute `diffusivity` is expected.

    Arguments:

        cells (tramway.inference.base.Distributed): distributed cells.

        maps (pandas.DataFrame or Maps): existing maps.

        grad_kwargs (dict): keyword arguments to :met:`cells.grad`.

    Returns:

        pandas.DataFrame or Maps: `maps` aumented with the SNR variables.
    '''
    if maps is None:
        index = np.array(list(cells.keys()))
        if _zeta_spurious:
            for i in cells:
                any_cell = cells[i]
                break
            if not hasattr(any_cell, 'diffusivity'):
                raise AttributeError('missing attribute `diffusivity`; please infer diffusion first')
            D = []
    else:
        if isinstance(maps, Maps):
            _maps = maps.maps
        elif isinstance(maps, pd.DataFrame):
            _maps = maps
        else:
            raise TypeError('unsupported type for `maps`: {}'.format(type(maps)))
        index = _maps.index.values
        if _zeta_spurious:
            D = _maps['diffusivity'].values
    # compute mean displacement m and variances V and V_prior (defined at cells `index`)
    sum_pts  = lambda a: np.sum(a, axis=0, keepdims=True)
    sum_dims = lambda a: np.sum(a, axis=1, keepdims=True)
    m, n, dr, dr2 = [], [], [], []
    dts = []
    for i in index:
        cell = cells[i]
        n.append(len(cell))
        m.append(np.mean(cell.dr, axis=0, keepdims=True))
        dr.append(sum_pts(cell.dr))
        dr2.append(sum_pts(cell.dr * cell.dr))
        dts.append(cell.dt)
        if maps is None and _zeta_spurious:
            D.append(cell.diffusivity)
    n = np.array(n)
    nnz = 1 < n
    m   = np.concatenate(m, axis=0)
    n   = n[:,np.newaxis]
    dr  = np.concatenate(dr, axis=0)
    dr2 = np.concatenate(dr2, axis=0)
    V   = sum_dims(dr2 - dr * dr / n) / n #(n - 1)
    n_prior   = np.sum(n)    - n
    dr_prior  = sum_pts(dr)  - dr
    dr2_prior = sum_pts(dr2) - dr2
    V_prior   = sum_dims(dr2_prior - dr_prior * dr_prior / n_prior) / n_prior #(n_prior - 1)
    # compute zeta_total (defined at cells `index`) and zeta_spurious (defined at cells `g_index`)
    sd = np.sqrt(V)
    zeta_total = np.zeros_like(m)
    zeta_total[nnz] = m[nnz] / sd[nnz]
    if _zeta_spurious:
        if maps is None:
            D = np.array(D)
        dts = np.concatenate(dts)
        dt = np.median(dts)
        if not np.all(np.isclose(dts, dt)):
            raise ValueError('dts are not all equal')
        reverse_index = np.full(cells.adjacency.shape[0], -1, dtype=int)
        reverse_index[index] = np.arange(len(index))
        # compute diffusivity gradient g (defined at cells `g_index`)
        g_index, g = [], []
        g_defined = np.zeros(len(index), dtype=bool)
        for j, i in enumerate(index):
            gradD = cells.grad(i, D, reverse_index, **grad_kwargs)
            if gradD is not None and nnz[j]:
                # the nnz condition does not prevent the gradient to be defined
                # but such cases are excluded anyway in the calculation of zeta_spurious
                g_defined[j] = True
                g_index.append(i)
                g.append(gradD[np.newaxis,:])
        g = np.concatenate(g, axis=0)
        # zeta_spurious
        zeta_spurious = g * dt / sd[g_defined]
    # format the output
    __maps = pd.DataFrame(
        np.concatenate((n, V_prior), axis=1),#, dr, dr2
        index=index,
        columns=['n'] + \
            ['V_prior'],# + \
            #['dr '+col for col in cells.space_cols] + \
            #['dr2 '+col for col in cells.space_cols],
        )
    if maps is None:
        _maps = __maps
    else:
        columns = ['n', 'V', 'V_prior'] \
                + ['zeta_total '+col for col in cells.space_cols]
        if _zeta_spurious:
            columns += ['zeta_spurious '+col for col in cells.space_cols]
        else:
            columns.append('sd')
        _maps = _maps.drop(columns=columns, errors='ignore').join(__maps)
    _maps = _maps.join(pd.DataFrame(
        np.concatenate((V, zeta_total), axis=1)[nnz],
        index=index[nnz],
        columns=['V'] + \
            ['zeta_total '+col for col in cells.space_cols],
        ))
    if _zeta_spurious:
        _maps = _maps.join(pd.DataFrame(
            zeta_spurious,
            index=g_index,
            columns=['zeta_spurious '+col for col in cells.space_cols],
            ))
    else:
        _maps = _maps.join(pd.DataFrame(
            sd[nnz],
            index=index[nnz],
            columns=['sd'],
            ))
    if isinstance(maps, Maps):
        maps.maps = _maps
    else:
        maps = _maps
    return maps


__all__ = [ 'add_snr_extensions', 'infer_snr', 'setup' ]

