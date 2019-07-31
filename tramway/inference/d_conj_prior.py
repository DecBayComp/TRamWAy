# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd

from tramway.inference.base import Maps
from tramway.inference.bayes_factors.get_D_posterior import *
from tramway.inference.gradient import setup_with_grad_arguments, get_grad_kwargs
from tramway.inference.snr import add_snr_extensions


setup = {
    'name': 'd.conj_prior',
    'provides': 'snr',
    'infer': 'infer_d_conj_prior',
    'arguments': dict(
        localization_error=(
            '-e', dict(type=float, help='localization error (same units as the variance)')),
        alpha=dict(type=float, default=.95, help='confidence level')),
}
setup_with_grad_arguments(setup)

def infer_d_conj_prior(cells, alpha=.95, return_zeta_spurious=True, trust=False, **kwargs):
    """
    Infer diffusivity MAP and confidence interval using a conjugate prior [Serov et al. 2019]

    Arguments:

        cells (tramway.inference.base.Distributed): distributed cells.

        alpha (float): confidence level for CI estimation.

        return_zeta_spurious (bool): add variable *zeta_spurious* to the returned dataframe.

        trust (bool): if ``False``, silently catch any exception raised in `get_D_confidence_interval`.

    Returns:

        pandas.DataFrame: maps including diffusivity MAPs and confidence intervals, and SNR extensions.

    Other valid input arguments are localization- and gradient-related arguments.

    See also :func:`~tramway.inference.bayes_factors.get_D_posterior.get_D_confidence_interval`,
    :meth:`~tramway.inference.base.Local.get_localization_error`
    and :func:`~tramway.inference.gradient.get_grad_kwargs`.
    """
    dim = cells.dim
    dt = cells.any_cell().dt[0]
    sigma2 = cells.get_localization_error(kwargs)
    if sigma2 is None:
        raise ValueError('undefined localization precision; please define `sigma` or `sigma2`')
    maps = add_snr_extensions(cells, _zeta_spurious=False)
    n, zeta_t, V, V_pi = maps['n'], Maps(maps)['zeta_total'], maps['V'], maps['V_prior']
    index, D_map, D_ci = [], [], []
    for i in cells:
        try:
            n_i = n.loc[i]
            zeta_i = zeta_t.loc[i]
            V_i = V.loc[i]
            V_minus_i = V_pi.loc[i]
        except KeyError:
            continue
        try:
            _map, _ci = get_D_confidence_interval(
                alpha, n_i, zeta_i, V_i, V_minus_i, dt, sigma2, dim)
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            if trust:
                raise
            continue
        index.append(i)
        D_map.append(_map)
        D_ci.append(_ci[np.newaxis, :])
    D_map = np.array(D_map)
    D_ci = np.vstack(D_ci)
    if return_zeta_spurious:
        reverse_index = np.full(cells.adjacency.shape[0], -1, dtype=int)
        reverse_index[index] = np.arange(len(index))
        grad_kwargs = get_grad_kwargs(**kwargs)
        g_index, g = [], []
        sd = maps['sd']
        g_defined = np.zeros(len(sd), dtype=bool)
        for j, i in enumerate(sd.index):
            gradD = cells.grad(i, D_map, reverse_index, **grad_kwargs)
            if gradD is not None:
                # the nnz condition does not prevent the gradient to be defined
                # but such cases are excluded anyway in the calculation of zeta_spurious
                g_defined[j] = True
                g_index.append(i)
                g.append(gradD[np.newaxis, :])
        g = np.concatenate(g, axis=0)
        # zeta_spurious
        dt = cells[i].dt[0]  # `add_snr_extensions` checked that all dts are equal
        sd = sd.values[:, np.newaxis]
        zeta_spurious = g * dt / sd[g_defined]
        maps = maps.join(pd.DataFrame(
            zeta_spurious,
            index=g_index,
            columns=['zeta_spurious ' + col for col in cells.space_cols],
        ))
    maps = maps.drop(columns=['sd']).join(pd.DataFrame(
        np.hstack((D_map[:, np.newaxis], D_ci)),
        index=index,
        columns=['diffusivity', 'ci low', 'ci high']))
    return maps

def infer_d_map_and_ci(*args, **kwargs):
    """
    Backward-compatibility alias for :func:`infer_d_conj_prior`.
    """
    return infer_d_conj_prior(*args, **kwargs)

__all__ = ['infer_d_conj_prior', 'infer_d_map_and_ci', 'setup']
