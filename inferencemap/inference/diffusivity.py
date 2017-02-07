
from math import pi, log
import numpy as np
import pandas as pd

from .base import Cell, Distributed, ArrayChain
from scipy.optimize import minimize_scalar, minimize


def d_neg_posterior(diffusivity, cell, square_localization_error=0.0, jeffreys_prior=False):
	if diffusivity < 0:
		print('negative diffusivity: {}'.format(diffusivity))
		return np.zeros_like(diffusivity)
	noise_dt = square_localization_error
	if cell.cache is None:
		cell.cache = np.sum(cell.dxy * cell.dxy, axis=1) # dx**2 + dy**2 + ..
	n = cell.cache.size # number of translocations
	diffusivity_dt = 4.0 * (diffusivity * cell.dt + noise_dt) # 4*(D+Dnoise)*dt
	d_neg_posterior_dt = n * log(pi) + np.sum(np.log(diffusivity_dt)) # sum(log(4*pi*Dtot*dt))
	d_neg_posterior_dxy = np.sum(cell.cache / diffusivity_dt) # sum((dx**2+dy**2+..)/(4*Dtot*dt))
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
		cell.cache = None
		result = minimize_scalar(d_neg_posterior, method='bounded', bounds=(0,1), \
			args=(cell, localization_error * localization_error, jeffreys_prior), \
			**kwargs)
		cell.cache = None # clear the cache
		return result.x



class DV(ArrayChain):
	def __init__(self, diffusivity, potential, priorD=None, priorV=None):
		ArrayChain.__init__(self, D=diffusivity, V=potential)
		self.both = np.empty(self.shape)
		self.D = diffusivity
		self.V = potential
		self.priorD = priorD
		self.priorV = priorV

	@property
	def D(self):
		return self.get(self.both, 'D')

	@property
	def V(self):
		return self.get(self.both, 'V')

	@D.setter
	def D(self, diffusivity):
		self.set(self.both, 'D', diffusivity)

	@V.setter
	def V(self, potential):
		self.set(self.both, 'V', potential)

	def update(self, x):
		self.both = x


def dv_neg_posterior(x, dv, cells, sq_loc_err, jeffreys_prior=False):
	dv.update(x)
	D = dv.D
	V = dv.V
	ncells = D.size #cells.adjacency.shape[0]
	result = 0.0
	for i in cells.cells:
		cell = cells.cells[i]
		n = cell.translocations.shape[0]
		# gradient of potential
		if cell.cache['vanders'] is None:
			cell.cache['vanders'] = [ np.vander(col, 3)[...,:2] for col in cell.area.T ]
		adj = cells.adjacency[i].indices
		dV = V[adj] - V[i]
		gradV = np.array([ np.linalg.lstsq(vander, dV)[0][1] \
			for vander in cell.cache['vanders'] ])
		# posterior
		Ddt = D[i] * cell.dt
		Dtot = 4.0 * (Ddt + sq_loc_err)
		DgradV = cell.dxy - np.outer(Ddt, gradV)
		result += n * log(pi) + np.sum(np.log(Dtot) + np.sum(DgradV * DgradV, axis=1) / Dtot)
		if cell.cache['area'] is None:
			# we want prod_i(area_i) = area_tot
			# just a random approximation:
			cell.cache['area'] = np.sqrt(np.mean(cell.area * cell.area, axis=0))
		if dv.priorV:
			result += dv.priorV * np.dot(gradV * gradV, cell.cache['area'])
		if dv.priorD:
			# gradient of diffusivity
			dD = D[adj] - D[i]
			gradD = np.array([ np.linalg.lstsq(vander, dD)[0][1] \
				for vander in cell.cache['vanders'] ])
			result += dv.priorD * np.dot(gradD * gradD, cell.cache['area'])
		if jeffreys_prior:
			result += 2.0 * (log(D[i] * np.mean(cell.dt) + sq_loc_err) - log(D[i]))
	return result


def inferDV(cell, localization_error=0.0, priorD=None, priorV=None, jeffreys_prior=False, **kwargs):
	ncells = cell.adjacency.shape[0]
	# initial values
	initialD = np.zeros(ncells)
	initialV = np.zeros(ncells)
	for i in cell.cells:
		c = cell.cells[i]
		initialD[i] = np.mean(c.dxy * c.dxy) / (2.0 * np.mean(c.dt))
		initialV[i] = -log(float(c.translocations.shape[0]) / float(cell.tcount))
	dv = DV(initialD, initialV, priorD, priorV)
	# initialize the caches
	for c in cell.cells:
		cell.cells[c].cache = dict(vanders=None, area=None)
	# run the optimization routine
	result = minimize(dv_neg_posterior, dv.both, method='BFGS', \
		args=(dv, cell, localization_error * localization_error, jeffreys_prior), \
		**kwargs)
	# collect the result
	cell.infered['D'] = dv.D
	cell.infered['V'] = dv.V
	return cell # useless

