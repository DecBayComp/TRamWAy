# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import ChainArray
from .base import *
from warnings import warn
from math import pi, log
import numpy as np
import pandas as pd
import numpy.ma as ma
from scipy.optimize import minimize
from collections import OrderedDict


setup = {'arguments': OrderedDict((
		('localization_error',	('-e', dict(type=float, default=0.01, help='localization error'))),
		('diffusivity_prior',	('-d', dict(type=float, default=0.01, help='prior on the diffusivity'))),
		('potential_prior',	('-v', dict(type=float, default=0.01, help='prior on the potential'))),
		('jeffreys_prior',	('-j', dict(action='store_true', help="Jeffreys' prior"))),
		('min_diffusivity',	dict(type=float, default=0, help='minimum diffusivity value allowed')))),
		'cell_sampling': 'group'}


class DV(ChainArray):
	__slots__ = ('diffusivity_prior', 'potential_prior', 'minimum_diffusivity')

	def __init__(self, diffusivity, potential, diffusivity_prior=None, potential_prior=None, \
		minimum_diffusivity=None, positive_diffusivity=None):
		# positive_diffusivity is for backward compatibility
		ChainArray.__init__(self, 'D', diffusivity, 'V', potential)
		self.diffusivity_prior = diffusivity_prior
		self.potential_prior = potential_prior
		self.minimum_diffusivity = minimum_diffusivity
		if minimum_diffusivity is None and positive_diffusivity is True:
			self.minimum_diffusivity = 0

	@property
	def D(self):
		return self['D']

	@property
	def V(self):
		return self['V']

	@D.setter
	def D(self, diffusivity):
		self['D'] = diffusivity

	@V.setter
	def V(self, potential):
		self['V'] = potential



def dv_neg_posterior(x, dv, cells, sq_loc_err, jeffreys_prior=False):
	"""
	Adapted from InferenceMAP's *dvPosterior* procedure::

		for (int a = 0; a < NUMBER_OF_ZONES; a++) {
			ZONES[a].gradVx = dvGradVx(DV,a);
			ZONES[a].gradVy = dvGradVy(DV,a);
			ZONES[a].gradDx = dvGradDx(DV,a);
			ZONES[a].gradDy = dvGradDy(DV,a);
			ZONES[a].priorActive = true;
		}


		for (int z = 0; z < NUMBER_OF_ZONES; z++) {
			const double gradVx = ZONES[z].gradVx;
			const double gradVy = ZONES[z].gradVy;
			const double gradDx = ZONES[z].gradDx;
			const double gradDy = ZONES[z].gradDy;

			const double D = DV[2*z];

			for (int j = 0; j < ZONES[z].translocations; j++) {
				const double dt = ZONES[z].dt[j];
				const double dx = ZONES[z].dx[j];
				const double dy = ZONES[z].dy[j];
				const double  Dnoise = LOCALIZATION_ERROR*LOCALIZATION_ERROR/dt;

				result += - log(4.0*PI*(D + Dnoise)*dt) - ((dx-D*gradVx*dt)*(dx-D*gradVx*dt) + (dy-D*gradVy*dt)*(dy-D*gradVy*dt))/(4.0*(D+Dnoise)*dt);
			}

			if (ZONES[z].priorActive == true) {
				result -= V_PRIOR*(gradVx*gradVx*ZONES[z].areaX + gradVy*gradVy*ZONES[z].areaY);
				result -= D_PRIOR*(gradDx*gradDx*ZONES[z].areaX + gradDy*gradDy*ZONES[z].areaY);
				if (JEFFREYS_PRIOR == 1) {
					result += 2.0*log(D*1.00) - 2.0*log(D*ZONES[z].dtMean + LOCALIZATION_ERROR*LOCALIZATION_ERROR);
			}
		}

	"""
	# extract `D` and `V`
	dv.update(x)
	D = dv.D
	V = dv.V
	# for all cell
	result = 0.0
	for i in cells:
		cell = cells[i]
		n = len(cell) # number of translocations
		# spatial gradient of the local potential energy
		gradV = cells.grad(i, V)
		if gradV is None:
			continue
		# various posterior terms
		Ddt = D[i] * cell.dt
		Dtot = 4.0 * (Ddt + sq_loc_err)
		dxdy_minus_D_gradV_dt = cell.dxy - np.outer(Ddt, gradV)
		result += n * log(pi) + np.sum( \
				np.log(Dtot) + \
				np.sum(dxdy_minus_D_gradV_dt * dxdy_minus_D_gradV_dt, axis=1) / \
				Dtot)
		# priors
		if dv.potential_prior:
			area = cells.grad_sum(i)
			result += dv.potential_prior * np.dot(gradV * gradV, area)
		if dv.diffusivity_prior:
			area = cells.grad_sum(i) # `area` is cached, therefore `grad_sum` can be called several times at no extra cost
			# spatial gradient of the local diffusivity
			gradD = cells.grad(i, D)
			assert gradD is not None
			result += dv.diffusivity_prior * np.dot(gradD * gradD, area)
		if jeffreys_prior:
			result += 2.0 * (log(D[i] * np.mean(cell.dt) + sq_loc_err) - log(D[i]))
	return result


def inferDV(cells, localization_error=0.0, diffusivity_prior=None, potential_prior=None, \
		jeffreys_prior=False, min_diffusivity=0, **kwargs):
	# ensure that cells are non-empty and translocations are properly oriented in time
	for c in cells.values():
		if not bool(c):
			raise ValueError('empty cells')
		if not np.all(0 < c.dt):
			warn('translocation dts are not all positive', RuntimeWarning)
			c.dxy[c.dt < 0] *= -1.
			c.dt[ c.dt < 0] *= -1.
	# initial values
	initialD = np.array([ np.mean(c.dxy * c.dxy) / 2. * np.mean(c.dt) for c in cells.values() ])
	n = np.array([ float(len(c)) for c in cells.values() ])
	initialV = -np.log(n / np.sum(n))
	dv = DV(initialD, initialV, diffusivity_prior, potential_prior, \
		minimum_diffusivity=min_diffusivity)
	# parametrize the optimization algorithm
	if min_diffusivity is None:
		#kwargs['method'] = 'BFGS'
		pass
	else:
		kwargs['bounds'] = [(min_diffusivity, None)] * dv.combined.size
		#kwargs['method'] = 'L-BFGS-B'
	# run the optimization routine
	sq_loc_err = localization_error * localization_error
	result = minimize(dv_neg_posterior, dv.combined, \
		args=(dv, cells, sq_loc_err, jeffreys_prior), \
		**kwargs)
	# collect the result
	dv.update(result.x)
	D, V = dv.D, dv.V
	# format the output
	index = np.array(list(cells.keys()))
	if index.size:
		inferred = pd.DataFrame(np.stack((D, V), axis=1), index=index, \
			columns=['diffusivity', 'potential'])
	else:
		inferred = None
	return inferred

