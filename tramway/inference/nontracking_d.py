
# plug-in for D inference with nontracking as a model

from .nontracking import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import OrderedDict


setup = {'name': 'nontracking.d',
	'provides': 'd',
	'cell_sampling': 'individual'}


def nontracking_d_neg_posterior(diffusivity, x0, x1):
	return -nontracking_posterior(diffusivity, x0, x1)


def infer(cells, **kwargs):
	inferred = {}
	for i in cells:
		cell = cells[i]
		assert cell # cells are not empty
		# split the frames
		ts = np.unique(cell.t)
		try:
			t0, t1 = ts
		except ValueError:
			if 2 < len(ts):
				raise ValueError('more than two frames per cell')
			raise NotImplementedError('all the visible molecules appeared or disappeared')
			continue
		x0 = cell.xy[cell.t == t0] # locations in the first frame
		x1 = cell.xy[cell.t == t1] # locations in the second frame

		# initialize the local diffusivity parameter
		raise NotImplementedError('initial_diffusivity')

		# maximize `nontracking_posterior`
		result = minimize(nontracking_d_neg_posterior, initial_diffusivity,
			args=(x0, x1), **kwargs)
		inferred[i] = result.x[0]

	# return a `DataFrame` with cell indices as indices and variable names as columns
	return pd.DataFrame(data={'diffusivity': pd.Series(data=inferred)})

