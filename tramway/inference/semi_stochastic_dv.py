# -*- coding: utf-8 -*-

# Copyright © 2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from . import stochastic_dv as sdv
import numpy as np
import logging


setup = dict(sdv.setup) # copy

setup.update({'name': 'semi.stochastic.dv',
    'infer': 'infer_DV'})


module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter('%(message)s'))
module_logger.addHandler(_console)


def infer_DV(cells, *args, **kwargs):
    """
    See also :func:`~tramway.inference.stochastic_dv.infer_stochastic_DV`.
    """

    ## stochastic start

    n = len(cells)

    kwargs['stochastic'] = True

    max_iter = kwargs.pop('max_iter', None)
    kwargs['max_iter'] = kwargs.pop('init_max_iter', 300) * n

    #try:
    #    ftol = kwargs['ftol']
    #except KeyError:
    #    ftol = None
    #else:
    #    kwargs['ftol'] = ftol / n

    verbose = kwargs.get('verbose', False)
    if verbose:
        module_logger.debug('======== stochastic optimization ========\n')

    DV0, info0 = sdv.infer_stochastic_DV(cells, *args, **kwargs)

    D0 = DV0['diffusivity'].values
    V0 = DV0['potential'].values

    if info0['resolution'] == 'interrupted':
        return DV0, info0

    ## non-stochastic fine-tuning

    kwargs['stochastic'] = False
    kwargs['D0'] = D0
    kwargs['V0'] = V0

    try:
        ls_step_max = kwargs['ls_step_max']
    except KeyError:
        pass
    else:
        if isinstance(ls_step_max, np.ndarray):
            kwargs['ls_step_max'] = ls_step_max.max()

    if max_iter is None:
        kwargs.pop('max_iter')
    else:
        kwargs['max_iter'] = max_iter

    #if ftol is not None:
    #    kwargs['ftol'] = ftol
    try:
        ftol = kwargs['ftol']
    except KeyError:
        pass
    else:
        kwargs['ftol'] = ftol * n

    if verbose:
        module_logger.debug('\n======= non-stochastic fine-tuning =======\n')
    try:

        DV, info = sdv.infer_stochastic_DV(cells, *args, **kwargs)

    except KeyboardInterrupt:
        DV, info = DV0, info0
    else:
        info['init'] = info0

    return DV, info

