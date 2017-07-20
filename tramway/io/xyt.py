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
import scipy.sparse as sparse
import h5py
import itertools

class IOWarning(Warning):
	pass
class FileNotFoundWarning(IOWarning):
	pass

def load_xyt(path, columns=['n', 'x', 'y', 't'], concat=True, return_paths=False, verbose=False):
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
			if dff['n'].min() < index_max:
				dff['n'] += index_max
				index_max = dff['n'].max()
			df.append(dff)
		except:
			warnings.warn(f, FileNotFoundWarning)
	if concat:
		df = pd.concat(df)
	if return_paths:
		return (df, paths)
	else:
		return df

