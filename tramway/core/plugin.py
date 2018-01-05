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


def list_plugins(dirname, package, lookup={}, force=False):
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
		missing = conflicting = None
		for key in lookup:
			if key in setup:
				continue
			ref = lookup[key]
			if isinstance(ref, type):
				matches = [ var for var in namespace
					if isinstance(getattr(module, var), ref) ]
			else:
				matches = [ var for var in namespace
					if fullmatch(ref, var) is not None ]
			if matches:
				if matches[1:]:
					conflicting = key
					if not force:
						break
				setup[key] = matches[0]
			else:
				missing = key
				if not force:
					break
		if conflicting:
			warn("multiple matches in module '{}' for key '{}'".format(path, conflicting), ImportWarning)
			if not force:
				continue
		if missing:
			warn("no match in module '{}' for key '{}'".format(path, missing), ImportWarning)
			if not force:
				continue
		if isinstance(name, (frozenset, set, tuple, list)):
			names = name
			for name in names:
				modules[name] = (setup, module)
		else:
			modules[name] = (setup, module)
	return modules

