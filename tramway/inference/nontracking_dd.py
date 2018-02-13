
# plugin for DD inference with nontracking as a model

from .nontracking import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from collections import OrderedDict


setup = {'name': 'nontracking.dd',
	'provides': 'dd',
	'cell_sampling': 'group',
	'arguments': OrderedDict((
		('diffusivity_prior',
			('-d', dict(type=float, default=0.01, help='prior on the diffusivity'))),
		)),
	}


def nontracking_dd_neg_posterior(diffusivity_array, cells, locations, diffusivity_prior):
	# most of the nontracking code should reside in the `inference.nontracking` subpackage
	neg_posterior = 0.
	for i in locations:

		# apply non-tracking posterior estimation at cell `i`
		x0, x1 = locations[i]
		diffusivity = diffusivity_array[i]
		neg_posterior -= nontracking_posterior(diffusivity, x0, x1)

		# regularize
		if diffusivity_prior:
			weights = cells.grad_sum(i)
			grad_diffusivity = cells.grad(i, diffusivity_array)
			neg_posterior += diffusivity_prior * \
				np.dot(grad_diffusivity * grad_diffusivity, weights)

	return neg_posterior


def infer(cells, diffusivity_prior=None, **kwargs):
	# split the frames
	locations = {}
	for i in cells:
		cell = cells[i]
		assert cell # cells are not empty
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
		locations[i] = (x0, x1)

	# initialize the diffusivity array
	raise NotImplementedError('initial_diffusivity')

	# minimize `nontracking_dd_neg_posterior`
	result = minimize(nontracking_dd_neg_posterior, initial_diffusivity,
		args=(cells, locations, diffusivity_prior), **kwargs)

	# return a `DataFrame` with cell indices as indices and variable names as columns
	index = list(cells.keys())
	return pd.DataFrame(data=result.x[:,np.newaxis], index=index, columns=['diffusivity'])


