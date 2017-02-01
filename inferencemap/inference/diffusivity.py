
from math import pi
import numpy as np
import pandas as pd

from .base import Cell, Distributed
#import scipy.optimize as optimize
from scipy.optimize import minimize_scalar


def d_neg_posterior(diffusivity, cell, square_localization_error=0.0, jeffreys_prior=False):
	if diffusivity < 0:
		print('negative diffusivity: {}'.format(diffusivity))
		return np.zeros_like(diffusivity)
	noise_dt = square_localization_error
	if cell.cache is None:
		cell.cache = np.sum(cell.dxy * cell.dxy, axis=1)
	diffusivity_dt = 4.0 * (diffusivity * cell.dt + noise_dt)
	d_neg_posterior_dt = np.sum(np.log(pi * diffusivity_dt))
	d_neg_posterior_dxy = np.sum(cell.cache / diffusivity_dt)
	if jeffreys_prior:
		d_neg_posterior_dt += diffusivity * np.mean(cell.dt) + noise_dt
	return d_neg_posterior_dt + d_neg_posterior_dxy


def inferD(cell, localization_error=0.0, jeffreys_prior=False, **kwargs):
	if isinstance(cell, Distributed):
		infered_map = {i: inferD(c, localization_error, jeffreys_prior, **kwargs) \
			for i, c in cell.cells.items()}
		cell.infered = pd.DataFrame(data={'D': pd.Series(data=infered_map)})
		return cell
	else:
		dInitial = np.mean(cell.dxy * cell.dxy) / (2.0 * np.mean(cell.dt))
		result = minimize_scalar(d_neg_posterior, method='bounded', bounds=(0,1), \
			args=(cell, localization_error * localization_error, jeffreys_prior), \
			**kwargs)
		return result.x

