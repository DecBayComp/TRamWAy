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
from tramway.core.plugin import *


plugins = list_plugins(os.path.dirname(__file__), __package__,
		{'infer': 'infer.*'}, require='setup')

# add `worker_count` argument to every mode with `cell_sampling` option set
_args = dict(type=int, help='number of parallel processes to spawn')
for _mode in plugins:
	_setup, _module = plugins[_mode]
	if 'cell_sampling' in _setup:
		if 'arguments' in _setup:
			if 'worker_count' in _setup['arguments']:
				continue
			_flags = [ _a[0] for _a in _setup['arguments'].values()
					if isinstance(_a, (tuple, list)) ]
		else:
			_setup['arguments'] = {}
			_flags = []
		if '-w' in _flags:
			_setup['arguments']['worker_count'] = _args
		else:
			_setup['arguments']['worker_count'] = ('-w', _args)

