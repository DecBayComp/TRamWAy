# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import importlib
import copy
import os
import re
try:
	fullmatch = re.fullmatch
except AttributeError: # Py2
	fullmatch = re.match
from warnings import warn


def list_plugins(dirname, package, patterns={}):
	pattern = re.compile(r'[a-zA-Z0-9].*[.]py')
	candidate_modules = [ os.path.splitext(fn)[0] \
		for fn in os.listdir(dirname) \
		if fullmatch(pattern, fn) is not None ]
	modules = {}
	for name in candidate_modules:
		path = '{}.{}'.format(package, name)
		module = importlib.import_module(path)
		if hasattr(module, 'setup'):
			setup = module.setup
			try:
				name = setup['name']
			except KeyError:
				setup['name'] = name
		else:
			setup = dict(name=name)
		namespace = list(module.__dict__.keys())
		for key in patterns:
			matches = [ var for var in namespace if fullmatch(patterns[key], var) is not None ]
			if matches:
				if matches[1:]:
					warn("multiple matches in module '{}' for key '{}'".format(path, key), RuntimeWarning)
				setup[key] = matches[0]
		modules[name] = (setup, module)
	return modules

