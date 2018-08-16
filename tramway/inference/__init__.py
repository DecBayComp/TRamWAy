# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import os.path
from .base import *
from tramway.core.plugin import Plugins


plugins = Plugins(os.path.dirname(__file__), __package__,
                {'infer': 'infer.*'}, require='setup')

# add `worker_count` argument to every mode with `cell_sampling` option set
# and `dilation` and `max_cell_count` if `cell_sampling` is not 'individual'
_wc_args = dict(type=int, help='number of parallel processes to spawn')
_mcc_args = dict(type=int, default=999999999999999999, help='max number of cells per group')
_dil_args = dict(type=int, help='cell group overlap as a number of "layers" of cells')
def _post_load(plugins):
        for _mode in plugins:
                _setup, _module = plugins[_mode]
                if 'cell_sampling' in _setup:
                        if 'arguments' in _setup:
                                if 'worker_count' in _setup['arguments']:
                                        continue
                                _flags = [ _a[0] for _a in _setup['arguments'].values()
                                                if isinstance(_a, (tuple, list)) and _a ]
                        else:
                                _setup['arguments'] = {}
                                _flags = []
                        if '-w' in _flags:
                                _setup['arguments']['worker_count'] = _wc_args
                        else:
                                _setup['arguments']['worker_count'] = ('-w', _wc_args)
                        if _setup['cell_sampling'] != 'individual':
                                if '-C' in _flags:
                                        _setup['arguments']['max_cell_count'] = _mcc_args
                                else:
                                        _setup['arguments']['max_cell_count'] = ('-C', _mcc_args)
                                _setup['arguments']['dilation'] = _dil_args

plugins.post_load = _post_load

