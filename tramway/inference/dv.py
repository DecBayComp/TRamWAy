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
from scipy.optimize import minimize
from collections import OrderedDict


setup = {'arguments': OrderedDict((
		('localization_error',	('-e', dict(type=float, default=0.03, help='localization error'))),
		('diffusivity_prior',	('-d', dict(type=float, default=0., help='prior on the diffusivity'))),
		('potential_prior',	('-v', dict(type=float, default=2., help='prior on the potential'))),
		('jeffreys_prior',	('-j', dict(action='store_true', help="Jeffreys' prior"))),
		('min_diffusivity',	dict(type=float, default=0., help='minimum diffusivity value allowed')),
		('diffusive_drift',	dict(action='store_true', help='make the diffusive drift available')))),
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



def dv_neg_posterior(x, dv, cells, squared_localization_error, jeffreys_prior, dt_mean):
	"""
	Adapted from InferenceMAP's *dvPosterior* procedure modified::

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
	#
	if dv.minimum_diffusivity is not None:
		observed_min = np.min(D)
		if observed_min < dv.minimum_diffusivity and \
				not np.isclose(observed_min, dv.minimum_diffusivity):
			warn(DiffusivityWarning(observed_min, dv.minimum_diffusivity))
	noise_dt = squared_localization_error
	# for all cell
	result = 0.
	for j, i in enumerate(cells):
		cell = cells[i]
		n = len(cell) # number of translocations
		# spatial gradient of the local potential energy
		gradV = cells.grad(i, V)
		if gradV is None:
			continue
		# various posterior terms
		D_dt = D[j] * cell.dt
		denominator = 4. * (D_dt + noise_dt)
		dxy_minus_drift = cell.dxy + np.outer(D_dt, gradV)
		# non-directional squared displacement
		ndsd = np.sum(dxy_minus_drift * dxy_minus_drift, axis=1)
		result += n * log(pi) + np.sum(np.log(denominator)) + np.sum(ndsd / denominator)
		# priors
		if dv.potential_prior:
			result += dv.potential_prior * cells.grad_sum(i, gradV * gradV)
		if dv.diffusivity_prior:
			# spatial gradient of the local diffusivity
			gradD = cells.grad(i, D)
			if gradD is not None:
				# `grad_sum` memoizes and can be called several times at no extra cost
				result += dv.diffusivity_prior * cells.grad_sum(i, gradD * gradD)
	if jeffreys_prior:
		result += 2. * np.sum(np.log(D * dt_mean + noise_dt) - np.log(D))
	return result


def inferDV(cells, localization_error=0.03, diffusivity_prior=None, potential_prior=2., \
		jeffreys_prior=False, min_diffusivity=0, diffusive_drift=False, **kwargs):
	# initial values and sanity checks
	index, n, dt_mean, initial = [], [], [], []
	for i in cells:
		cell = cells[i]
		if not bool(cell):
			raise ValueError('empty cells')
		# ensure that translocations are properly oriented in time
		if not np.all(0 < cell.dt):
			warn('translocation dts are not all positive', RuntimeWarning)
			cell.dxy[cell.dt < 0] *= -1.
			cell.dt[ cell.dt < 0] *= -1.
		# initialize the local diffusivity parameter
		dt_mean_i = np.mean(cell.dt)
		initial_i = np.mean(cell.dxy * cell.dxy) / (2.0 * dt_mean_i)
		#
		index.append(i)
		n.append(float(len(cell)))
		dt_mean.append(dt_mean_i)
		initial.append(initial_i)
	any_cell = cell
	index, n, dt_mean, initialD = np.array(index), np.array(n), np.array(dt_mean), np.array(initial)
	initialV = -np.log(n / np.sum(n))
	dv = DV(initialD, initialV, diffusivity_prior, potential_prior, min_diffusivity)
	# parametrize the optimization algorithm
	if min_diffusivity is not None:
		kwargs['bounds'] = [(min_diffusivity, None)] * initialD.size + \
			[(None, None)] * initialV.size
		#kwargs['method'] = 'L-BFGS-B'
	# run the optimization routine
	squared_localization_error = localization_error * localization_error
	result = minimize(dv_neg_posterior, dv.combined, \
		args=(dv, cells, squared_localization_error, jeffreys_prior, dt_mean), \
		**kwargs)
	# collect the result
	dv.update(result.x)
	D, V = dv.D, dv.V
	index = np.array(list(cells.keys()))
	DVF = pd.DataFrame(np.stack((D, V), axis=1), index=index, \
		columns=[ 'diffusivity', 'potential'])
	# derivate the forces
	index_, F = [], []
	for i in index:
		gradV = cells.grad(i, V)
		if gradV is not None:
			index_.append(i)
			F.append(-gradV)
	F = pd.DataFrame(np.stack(F, axis=0), index=index_, \
		columns=[ 'force ' + col for col in any_cell.space_cols ])
	DVF = DVF.join(F)
	# derivate the diffusive drifts
	if diffusive_drift:
		index_, drift = [], []
		for i, D_i in zip(index, D):
			gradD_i = cells.grad(i, D)
			if gradD_i is not None:
				index_.append(i)
				drift.append( D_i * gradD_i )
		drift = pd.DataFrame(np.stack(drift, axis=0), index=index_, \
			columns=[ 'drift ' + col for col in any_cell.space_cols ])
		DVF = DVF.join(drift)
	return DVF

