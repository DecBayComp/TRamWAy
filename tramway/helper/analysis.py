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
from tramway.core.lazy import lightcopy
from tramway.core.analyses import *
from tramway.core.hdf5 import HDF5Store, lazytype, lazyvalue
from tramway.tessellation.base import Tessellation, CellStats
from rwa.lazy import LazyPeek, PermissivePeek


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
	try:
		hdf = HDF5Store(path, 'r')
		#hdf._default_lazy = PermissivePeek
		hdf.lazy = True
		try:
			analyses = lazyvalue(hdf.peek('analyses'))
		finally:
			hdf.close()
	except EnvironmentError as e:
		if e.args[1:]:
			print(traceback.format_exc())
			raise OSError('HDF5 libraries may not be installed')
		else:
			raise
	return coerce_labels(analyses)



known_lossy = [Tessellation, CellStats]


def save_rwa(path, analyses, verbose=False, force=False, compress=True, lossy=None):
	if compress or lossy:
		import warnings
		warnings.warn('the `compression` and `lossy` are currently ignored', DeprecationWarning)
		try:
			pass#analyses = lightcopy(analyses)
		except (KeyboardInterrupt, SystemExit):
			raise
		except:
			if verbose:
				print('loss-less compression failed with the following error:')
				print(traceback.format_exc())
		if False:#lossy:
			if isinstance(lossy, (type, tuple, list, frozenset, set)):
				lossy = set(lossy) + known_lossy
			else:
				lossy = known_lossy
			def lossy_compress(data):
				t = lazytype(data)
				if any(issubclass(t, _t) for _t in lossy):
					if isinstance(data, LazyPeek):
						data = data.deep()
					data.freeze()
			try:
				map_analyses(lossy_compress, analyses)
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				if verbose:
					print('lossy compression failed with the following error:')
					print(traceback.format_exc())
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
	except EnvironmentError:
		print(traceback.format_exc())
		raise ImportError('HDF5 libraries may not be installed')
	if verbose:
		print('written analysis tree:')
		print(format_analyses(analyses, global_prefix='\t'))


def format_analyses(analyses, prefix='\t', node=None, global_prefix=''):
	if not isinstance(analyses, Analyses) and os.path.isfile(analyses):
		analyses = find_analysis(analyses)
	if node is None:
		try:	node = lazytype
		except:	node = type
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

