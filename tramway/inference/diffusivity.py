# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import pi, log
import numpy as np
import pandas as pd
import numpy.ma as ma
from warnings import warn
import itertools

from .base import Cell, Distributed, ChainArray
from scipy.optimize import minimize_scalar, minimize



class DiffusivityWarning(RuntimeWarning):
	def __init__(self, diffusivity, lower_bound):
		self.diffusivity = diffusivity
		self.lower_bound = lower_bound

	def __repr__(self):
		return 'DiffusivityWarning({}, {})'.format(self.diffusivity, self.lower_bound)

	def __str__(self):
		return 'diffusivity too low: {} < {}'.format(self.diffusivity, self.lower_bound)



def d_neg_posterior(diffusivity, cell, square_localization_error=0.0, jeffreys_prior=False, \
	min_diffusivity=0):
	if diffusivity < min_diffusivity and not np.isclose(diffusivity, min_diffusivity):
		warn(DiffusivityWarning(diffusivity, min_diffusivity))
	noise_dt = square_localization_error
	if cell.cache is None:
		cell.cache = np.sum(cell.dxy * cell.dxy, axis=1) # dx**2 + dy**2 + ..
	n = cell.cache.size # number of translocations
	diffusivity_dt = 4.0 * (diffusivity * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
	if np.any(np.isclose(diffusivity_dt, 0)):
		raise RuntimeError('near-0 diffusivity; increase `localization_error`')
	d_neg_posterior_dt = n * log(pi) + np.sum(np.log(diffusivity_dt)) # sum(log(4*pi*Dtot*dt))
	d_neg_posterior_dxy = np.sum(cell.cache / diffusivity_dt) # sum((dx**2+dy**2+..)/(4*Dtot*dt))
	if jeffreys_prior:
		d_neg_posterior_dt += diffusivity * np.mean(cell.dt) + noise_dt
	return d_neg_posterior_dt + d_neg_posterior_dxy


def inferD(cell, localization_error=0.0, jeffreys_prior=False, min_diffusivity=0, **kwargs):
	sq_loc_err = localization_error * localization_error
	if isinstance(cell, Distributed):
		inferred = {i: inferD(c, localization_error, jeffreys_prior, \
				min_diffusivity, **kwargs) \
			for i, c in cell.cells.items() if 0 < c.tcount}
		inferred = pd.DataFrame(data={'diffusivity': pd.Series(data=inferred)})
		return inferred
	else:
		#assert not np.isclose(np.mean(cell.dt), 0)
		if not np.all(0 < cell.dt):
			warn('translocation dts are non-positive', RuntimeWarning)
			cell.dt = abs(cell.dt)
		initialD = np.mean(cell.dxy * cell.dxy) / (2.0 * np.mean(cell.dt))
		cell.cache = None # initialize empty cache
		if min_diffusivity is not None:
			kwargs['bounds'] = [(min_diffusivity,None)]
		result = minimize(d_neg_posterior, initialD, \
			args=(cell, sq_loc_err, jeffreys_prior, min_diffusivity), \
			**kwargs)
		return result.x[0]




def df_neg_posterior(x, df, cell, square_localization_error=0.0, jeffreys_prior=False, \
	min_diffusivity=0):
	df.update(x)
	diffusivity = df['D']
	force = df['F']
	if diffusivity < min_diffusivity:
		warn(DiffusivityWarning(diffusivity, min_diffusivity))
	noise_dt = square_localization_error
	n = cell.dt.size # number of translocations
	diffusivity_dt = diffusivity * cell.dt
	diffusivity_term = 4.0 * (diffusivity_dt + noise_dt) # 4*(D+Dnoise)*dt
	residual = cell.dxy - np.outer(diffusivity_dt, force)
	force_term = np.sum(residual * residual, axis=1)
	neg_posterior_d = n * log(pi) + np.sum(np.log(diffusivity_term)) # sum(log(4*pi*Dtot*dt))
	neg_posterior_f = np.sum(force_term / diffusivity_term)
	if jeffreys_prior:
		neg_posterior_d += 2.0 * (log(diffusivity * np.mean(cell.dt) + noise_dt) - log(diffusivity))
	return neg_posterior_d + neg_posterior_f


def inferDF(cell, localization_error=0.0, jeffreys_prior=False, min_diffusivity=0, **kwargs):
	if isinstance(cell, Distributed):
		sq_loc_err = localization_error * localization_error
		index, inferred = zip(*[ (i,
				inferDF(c, sq_loc_err, jeffreys_prior, min_diffusivity, **kwargs))
			for i, c in cell.cells.items() if 0 < c.tcount ])
		inferred = pd.DataFrame(data=np.stack(inferred, axis=0), \
			index=np.array(index), \
			columns=[ 'diffusivity' ] + \
				[ 'force '+col for col in next(iter(cell.cells.values())).space_cols ])
		return inferred
	else:
		if not np.all(0 < cell.dt):
			warn('translocation dts are non-positive', RuntimeWarning)
			cell.dt = abs(cell.dt)
		initialD = np.mean(cell.dxy * cell.dxy) / (2.0 * np.mean(cell.dt))
		initialF = np.zeros(cell.dim, dtype=initialD.dtype)
		df = ChainArray('D', initialD, 'F', initialF)
		if min_diffusivity is not None:
			if 'bounds' in kwargs:
				print(kwargs['bounds'])
			kwargs['bounds'] = [(min_diffusivity, None)] + [(None, None)] * initialF.size
		#cell.cache = None # no cache needed
		result = minimize(df_neg_posterior, df.combined, \
			args=(df, cell, localization_error, jeffreys_prior, min_diffusivity), \
			**kwargs)
		# note that localization_error here is actually the square localization error
		#df.update(result.x)
		#return (df['D'], df['F'])
		return result.x # needless to split df.combined into D and F




def dd_neg_posterior(diffusivity, cells, square_localization_error, priorD, jeffreys_prior, dt_mean, \
	index_map=None, min_diffusivity=0):
	observed_min = np.min(diffusivity)
	if observed_min < min_diffusivity and not np.isclose(observed_min, min_diffusivity):
		warn(DiffusivityWarning(observed_min, min_diffusivity))
	noise_dt = square_localization_error
	j = 0
	result = 0.0
	for i in cells.cells:
		cell = cells.cells[i]
		n = cell.tcount
		if n == 0:
			continue
		# posterior calculations
		if cell.cache['dxy2'] is None:
			cell.cache['dxy2'] = np.sum(cell.dxy * cell.dxy, axis=1) # dx**2 + dy**2 + ..
		diffusivity_dt = 4.0 * (diffusivity[j] * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
		result += n * log(pi) + np.sum(np.log(diffusivity_dt)) # sum(log(4*pi*Dtot*dt))
		result += np.sum(cell.cache['dxy2'] / diffusivity_dt) # sum((dx**2+dy**2+..)/(4*Dtot*dt))
		# prior
		if priorD:
			area = cells.grad_sum(i, index_map)
			# gradient of diffusivity
			gradD = cells.grad(i, diffusivity, index_map)
			if gradD is None:
				#raise RuntimeError('missing gradient')
				continue
			result += priorD * np.dot(gradD * gradD, area)
		j += 1
	if jeffreys_prior:
		result += 2.0 * np.sum( log(diffusivity * dt_mean + square_localization_error) - \
					log(diffusivity) )
	return result

def inferDD(cells, localization_error=0.0, priorD=None, jeffreys_prior=None, \
	min_diffusivity=0, **kwargs):
	sq_loc_err = localization_error * localization_error
	initial = []
	j = 0
	for i in cells.cells:
		cell = cells.cells[i]
		if 0 < cell.tcount:
			if not np.all(0 < cell.dt):
				warn('translocation dts are non-positive', RuntimeWarning)
				cell.dt = abs(cell.dt)
			cell.cache = dict(dxy2=None, area=None) # initialize cached constants
			dt_mean_i = np.mean(cell.dt)
			initial.append((i, j, \
				dt_mean_i, \
				np.mean(cell.dxy * cell.dxy) / (2.0 * dt_mean_i)))
			j += 1
	i, j, dt_mean, initialD = (np.array(xs) for xs in zip(*initial))
	index = np.full(cells.adjacency.shape[0], -1, dtype=int)
	index[i] = j
	if min_diffusivity is not None:
		kwargs['bounds'] = [(min_diffusivity,None)] * initialD.size
	result = minimize(dd_neg_posterior, initialD,
		args=(cells, sq_loc_err, priorD, jeffreys_prior, dt_mean, index, min_diffusivity),
		**kwargs)
	return pd.DataFrame(data=result.x[:,np.newaxis], index=i, columns=['diffusivity'])




class DV(ChainArray):
	__slots__ = ChainArray.__slots__ + ['priorD', 'priorV', 'minimumD']

	def __init__(self, diffusivity, potential, priorD=None, priorV=None, minimumD=None, positiveD=None):
		# positiveD is for backward compatibility
		ChainArray.__init__(self, 'D', diffusivity, 'V', potential)
		self.priorD = priorD
		self.priorV = priorV
		self.minimumD = minimumD
		if minimumD is None and positiveD is True:
			self.minimumD = 0

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
		if n == 0:
			continue
		# gradient of potential
		gradV = cells.grad(i, V)
		if gradV is None:
			continue
		# posterior
		Ddt = D[i] * cell.dt
		Dtot = 4.0 * (Ddt + sq_loc_err)
		residual = cell.dxy - np.outer(Ddt, gradV)
		result += n * log(pi) + np.sum(np.log(Dtot) + np.sum(residual * residual, axis=1) / Dtot)
		if dv.priorV:
			area = cells.grad_sum(i)
			result += dv.priorV * np.dot(gradV * gradV, area)
		if dv.priorD:
			area = cells.grad_sum(i) # area is cached, therefore grad_sum can be called several times at no extra cost
			# gradient of diffusivity
			gradD = cells.grad(i, D)
			assert gradD is not None
			result += dv.priorD * np.dot(gradD * gradD, area)
		if jeffreys_prior:
			result += 2.0 * (log(D[i] * np.mean(cell.dt) + sq_loc_err) - log(D[i]))
	return result


def inferDV(cell, localization_error=0.0, priorD=None, priorV=None, jeffreys_prior=False, \
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
	index, initialD, initialV, mask = zip(*initial)
	mask = list(mask)
	index = ma.array(index, mask=mask)
	initialD = ma.array(initialD, mask=mask)
	initialV = ma.array(initialV, mask=mask)
	dv = DV(initialD, initialV, priorD, priorV, minimumD=min_diffusivity)
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



