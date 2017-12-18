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
		if not os.path.isfile(path):
			raise OSError("missing file '{}'".format(path))
		hdf = HDF5Store(path, 'r')
		try:
			analyses = hdf.peek('analyses')
		except EnvironmentError:
			print(traceback.format_exc())
			raise OSError('HDF5 libraries may not be installed')
		else:
			if labels:
				analyses = select_analysis(analyses, labels)
		finally:
			try:
				hdf.close()
			except:
				pass
		return analyses

def map_analyses(fun, analyses, label=False, comment=False, depth=False):
	with_label, with_comment, with_depth = label, comment, depth
	def _fun(x, **kwargs):
		y = fun(x, **kwargs)
		if isinstance(y, tuple):
			raise ValueError('type conflict: function returned a tuple')
		return y
	def _map(analyses, label=None, comment=None, depth=0):
		kwargs = {}
		if with_label:
			kwargs['label'] = label
		if with_comment:
			kwargs['comment'] = comment
		if with_depth:
			kwargs['depth'] = depth
		node = _fun(analyses.data, **kwargs)
		if analyses.instances:
			depth += 1
			tree = []
			for label in analyses.instances:
				child = analyses.instances[label]
				comment = analyses.comments[label]
				if isinstance(child, Analyses):
					tree.append(_map(child, label, comment, depth))
				else:
					if with_label:
						kwargs['label'] = label
					if with_comment:
						kwargs['comment'] = comment
					if with_depth:
						kwargs['depth'] = depth
					tree.append(_fun(child, **kwargs))
			return (node, tuple(tree))
		else:
			return node
	return _map(analyses)

def format_analyses(analyses, prefix='\t', node=type, global_prefix=''):
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

