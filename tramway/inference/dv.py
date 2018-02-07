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
	dv.update(x)
	D = dv.D
	V = dv.V
	result = 0.0
	for i in cells.cells:
		cell = cells.cells[i]
		n = cell.tcount
		assert 0 < n
		#if n == 0:
		#	continue
		# gradient of potential
		gradV = cells.grad(i, V)
		if gradV is None:
			continue
		# posterior
		Ddt = D[i] * cell.dt
		Dtot = 4.0 * (Ddt + sq_loc_err)
		residual = cell.dxy - np.outer(Ddt, gradV)
		result += n * log(pi) + np.sum(np.log(Dtot) + np.sum(residual * residual, axis=1) / Dtot)
		if dv.potential_prior:
			area = cells.grad_sum(i)
			result += dv.potential_prior * np.dot(gradV * gradV, area)
		if dv.diffusivity_prior:
			area = cells.grad_sum(i) # area is cached, therefore grad_sum can be called several times at no extra cost
			# gradient of diffusivity
			gradD = cells.grad(i, D)
			assert gradD is not None
			result += dv.diffusivity_prior * np.dot(gradD * gradD, area)
		if jeffreys_prior:
			result += 2.0 * (log(D[i] * np.mean(cell.dt) + sq_loc_err) - log(D[i]))
	return result


def inferDV(cell, localization_error=0.0, diffusivity_prior=None, potential_prior=None, jeffreys_prior=False, \
	min_diffusivity=0, **kwargs):
	sq_loc_err = localization_error * localization_error
	# initial values
	initial = []
	for i in cell.cells:
		c = cell.cells[i]
		if 0 < c.tcount:
			if not np.all(0 < c.dt):
				warn('translocation dts are non-positive', RuntimeWarning)
				c.dt = abs(c.dt)
			initial.append((i, \
				np.mean(c.dxy * c.dxy) / (2.0 * np.mean(c.dt)), \
				-log(float(c.tcount) / float(cell.tcount)), \
				False))
		else:
			initial.append((i, 0, 0, True))
	index, initial_diffusivity, initial_potential, mask = zip(*initial)
	mask = list(mask)
	index = ma.array(index, mask=mask)
	initial_diffusivity = ma.array(initial_diffusivity, mask=mask)
	initial_potential = ma.array(initial_potential, mask=mask)
	dv = DV(initial_diffusivity, initial_potential, diffusivity_prior, potential_prior, \
		minimum_diffusivity=min_diffusivity)
	# initialize the caches
	for c in cell.cells:
		cell.cells[c].cache = dict(vanders=None, area=None)
	if min_diffusivity is None:
		#kwargs['method'] = 'BFGS'
		pass
	else:
		kwargs['bounds'] = [(min_diffusivity, None)] * dv.combined.size
		#kwargs['method'] = 'L-BFGS-B'
	# run the optimization routine
	result = minimize(dv_neg_posterior, dv.combined, \
		args=(dv, cell, sq_loc_err, jeffreys_prior), \
		**kwargs)
	dv.update(result.x)
	# collect the result
	#inferred['D'] = dv.D
	#inferred['V'] = dv.V
	D, V = dv.D, dv.V
	if isinstance(D, ma.MaskedArray): D = D.compressed()
	if isinstance(V, ma.MaskedArray): V = V.compressed()
	if isinstance(index, ma.MaskedArray): index = index.compressed()
	if index.size:
		inferred = pd.DataFrame(np.stack((D, V), axis=1), index=index, \
			columns=['diffusivity', 'potential'])
	else:
		inferred = None
	return inferred

