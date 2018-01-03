# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import os
import itertools
import traceback
from tramway.core.analyses import *
from tramway.io import HDF5Store


try:
	input = raw_input # Py2
except NameError:
	pass


def list_rwa(path):
	if not isinstance(path, (tuple, list)):
		path = (path,)
	ext = '.rwa'
	paths = []
	for p in path:
		if os.path.isdir(p):
			paths.append([ os.path.join(p, f) for f in os.listdir(p) if f.endswith(ext) ])
		else:
			if p.endswith(ext):
				ps = [p]
				paths.append(ps)
	paths = tuple(itertools.chain(*paths))
	return paths


def find_analysis(path, labels=None):
	if isinstance(path, (tuple, list, set, frozenset)):
		paths = path
		matches = {}
		for path in paths:
			try:
				matches[path] = find_analysis(path, labels)
			except KeyError:
				pass
		return matches
	else:
		analyses = load_rwa(path)
		if labels:
			analyses = extract_analysis(analyses, labels)
		return analyses


def load_rwa(path):
	#if not os.path.isfile(path):
	#	raise OSError(2, "missing file '{}'".format(path))
	try:
		hdf = HDF5Store(path, 'r')
		try:
			analyses = hdf.peek('analyses')
		finally:
			hdf.close()
	except EnvironmentError:
		print(traceback.format_exc())
		raise OSError('HDF5 libraries may not be installed')
	return coerce_labels(analyses)


def save_rwa(path, analyses, verbose=False, force=False):
	if not force and os.path.isfile(path):
		answer = input("overwrite file '{}': [N/y] ".format(path))
		if not (answer and answer[0].lower() == 'y'):
			return
	try:
		store = HDF5Store(path, 'w', int(verbose) - 1 if verbose else False)
		if verbose:
			print('writing file: {}'.format(path))
		store.poke('analyses', analyses)
		store.close()
	except Exception as e:
		try:
			os.unlink(path)
		except OSError as e1: # Py3 has FileNotFoundError
			if e1.errno == 2:
				pass
			elif verbose:
				print(traceback.format_exc(e1))
		else:
			if verbose:
				print('deleting file: {}'.format(path))
		if isinstance(e, EnvironmentError):
			print(traceback.format_exc(e))
			raise ImportError('HDF5 libraries may not be installed')
		else:
			raise e
	if verbose:
		print('written analysis tree:'.format(output_file))
		print(format_analyses(analyses, global_prefix='\t'))


def format_analyses(analyses, prefix='\t', node=type, global_prefix=''):
	if not isinstance(analyses, Analyses) and os.path.isfile(analyses):
		analyses = find_analysis(analyses)
	def _format(data, label=None, comment=None, depth=0):
		s = [global_prefix + prefix * depth]
		t = []
		if label is None:
			assert comment is None
			if node:
				s.append(str(node(data)))
			else:
				return None
		else:
			try:
				label + 0 # check numeric types
			except TypeError:
				s.append("'{}'")
			else:
				s.append('[{}]')
			t.append(label)
			if node:
				s.append(' {}')
				t.append(node(data))
			if comment:
				assert isinstance(comment, str)
				s.append(':\t"{}"')
				t.append(comment)
		return ''.join(s).format(*t)
	def _flatten(_node):
		if _node is None:
			return []
		elif isinstance(_node, str):
			return [ _node ]
		try:
			_node, _children = _node
		except TypeError:
			return []
		else:
			assert isinstance(_node, str)
			return itertools.chain([_node], *[_flatten(c) for c in _children])
	return '\n'.join(_flatten(map_analyses(_format, analyses, label=True, comment=True, depth=True)))

