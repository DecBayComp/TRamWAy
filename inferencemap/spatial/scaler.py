
import numpy as np
import pandas as pd


class Scaler(object):
	""":class:`Scaler` scales data points, point differences (vectors) or distances.
	It initializes itself with the first provided sample, and then scales equally the next samples.
	It manages a constraint in the calculation of the scaling parameters, forcing a common factors
	over a subset of dimensions. Attribute :attr:`euclidian` controls the selection of this subset.
        Distances are scaled and unscaled only in this subspace, if it is defined.
	A default `Scaler()` instance does not scale, neither raises errors.

	Beware that when possible data are scaled in place."""
	__slots__ = ['init', 'center', 'factor', 'function', 'euclidian']

	def __init__(self, scale=None, euclidian=None):
		self.init   = True
		self.center = None
		self.factor = None
		self.function = scale
		if euclidian and not \
			(isinstance(euclidian, list) and euclidian[1:]):
			raise TypeError('`euclidian` should be a multi-element list')
		self.euclidian = euclidian

	@property
	def ready(self): return not (self.function and self.init)

	def scalePoint(self, points, inplace=True):
		if self.function:
			if self.init:
				self.center, self.factor = self.function(points)
				if self.euclidian:
					if isinstance(points, pd.DataFrame):
						xyz = points[self.euclidian].values
					else:
						xyz = points[:,self.euclidian]
					_, self.factor[self.euclidian] = self.function(xyz.flatten())
				self.init = False
			if not inplace:
				points = points.copy(deep=False)
			if self.center is not None:
				points -= self.center
			if self.factor is not None:
				points /= self.factor
		return points

	def unscalePoint(self, points, inplace=True):
		if self.function:
			if self.init: raise AttributeError('scaler has not been initialized')
			if not inplace:
				points = points.copy(deep=False)
			if self.factor is not None:
				points *= self.factor
			if self.center is not None:
				points += self.center
		return points

	def scaleVector(self, vect, inplace=True):
		if self.function:
			if self.init: raise AttributeError('scaler has not been initialized')
			if not inplace:
				vect = vect.copy(deep=False)
			if self.factor is not None:
				vect /= self.factor
		return vect

	def scaleDistance(self, dist, inplace=True):
		if self.function:
			if self.init: raise AttributeError('scaler has not been initialized')
			if self.factor is not None:
				if self.euclidian:
					if not inplace:
						dist = dist.copy(deep=False)
					dist /= self.factor[self.euclidian[0]]
				else:
					raise AttributeError('distance cannot be scaled because no euclidian variables have been designated')
		return dist


def _whiten(x):
	'''Scaling function for :class:`Scaler`. Performs `(x - mean(x)) / std(x)`. Consider using
	:func:`whiten` instead.'''
	scaling_center = x.mean(axis=0)
	scaling_factor = x.std(axis=0)
	return (scaling_center, scaling_factor)

def whiten(): # should be a function so that each new instance is a distinct one
	return Scaler(_whiten)


def _unitrange(x):
	'''Scaling function for :class:`Scaler`. Performs `(x - min(x)) / (max(x) - min(x))`. Consider 
	using :func:`unitrange` instead.'''
	scaling_center = x.min(axis=0)
	scaling_factor = x.max(axis=0) - scaling_center
	return (scaling_center, scaling_factor)

def unitrange():
	return Scaler(_unitrange)

