# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


#import numpy as np
#import pandas as pd

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


