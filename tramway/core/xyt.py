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
import numpy as np
import pandas as pd
import warnings
import itertools
from .exceptions import *


def _translocations(df, sort=True): # very slow; may soon be deprecated
	def diff(df):
		df = df.sort_values('t')
		t = df['t'][1:]
		df = df.diff()
		df['t'] = t
		return df
	return df.groupby(['n'], sort=False).apply(diff)

def translocations(df, sort=False):
	'''each trajectories should be represented by consecutive rows sorted by time.'''
	if sort:
		return _translocations(df) # not exactly equivalent
		#raise NotImplementedError
	i = 'n'
	xyz = ['x', 'y']
	if 'z' in df.columns:
		xyz.append('z')
	ixyz = xyz + [i]
	jump = df[ixyz].diff()
	#df[xyz] = jump[xyz]
	#df = df[jump[i] != 0]
	#return df
	jump = jump[jump[i] == 0][xyz]
	return jump#np.sqrt(np.sum(jump * jump, axis=1))


def load_xyt(path, columns=['n', 'x', 'y', 't'], concat=True, return_paths=False, verbose=False):
	if 'n' not in columns:
		raise ValueError("trajectory index should be denoted 'n'")
	if not isinstance(path, list):
		path = [path]
	paths = []
	for p in path:
		if os.path.isdir(p):
			paths.append([ os.path.join(p, f) for f in os.listdir(p) ])
		else:
			paths.append([p])
	index_max = 0
	df = []
	paths = list(itertools.chain(*paths))
	for f in paths:
		try:
			if verbose:
				print('loading file: {}'.format(f))
			dff = pd.read_table(f, names=columns)
		except OSError:
			warnings.warn(f, FileNotFoundWarning)
		else:
			sample = dff[dff['n']==dff['n'].iloc[-1]]
			sample_dt = sample['t'].diff()[1:]
			if not all(0 < sample_dt):
				if any(0 == sample_dt):
					print(sample)
					raise ValueError("some indices refer to multiple simultaneous trajectories in table: '{}'".format(f))
				else:
					warnings.warn(EfficiencyWarning("table '{}' is not properly ordered".format(f)))
				# faster sort
				data = np.asarray(dff)
				dff = pd.DataFrame(data=data[np.lexsort((dff['t'], dff['n']))],
					columns=dff.columns)
				#sorted_dff = []
				#for n in dff['n'].unique():
				#	sorted_dff.append(dff[dff['n'] == n].sort_values(by='t'))
				#dff = pd.concat(sorted_dff)
				#dff.index = np.arange(dff.shape[0]) # optional
			if dff['n'].min() < index_max:
				dff['n'] += index_max
				index_max = dff['n'].max()
			df.append(dff)
	if concat:
		df = pd.concat(df)
	if return_paths:
		return (df, paths)
	else:
		return df

