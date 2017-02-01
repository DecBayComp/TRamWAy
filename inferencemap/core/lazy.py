
import sys
from copy import copy, deepcopy

class Lazy(object):
	__slots__ = ['_lazy']

	__lazy__ = []

	def __init__(self):
		self._lazy = {name: True for name in self.__lazy__}

	def __tolazy__(self, name):
		return name[1:]

	def __fromlazy__(self, name):
		return '_{}'.format(name)

	def __setlazy__(self, name, value):
		self._lazy[name] = value is None
		setattr(self, self.__fromlazy__(name), value)

	def __lazysetter__(self, value):
		self.__setlazy__(sys._getframe(1).f_code.co_name, value)

	def unload(self):
		try:
			names = self.__slots__
		except:
			names = self.__dict__
		standard_attrs = []
		# set lazy attributes to None (unset them so that memory is freed)
		for name in names:
			if self._lazy.get(self.__tolazy__(name), False):
				try:
					setattr(self, name, None)
				except AttributeError:
					pass
			else:
				standard_attrs.append(name)
		# search for Lazy object attributes so that they too can be unloaded
		for name in standard_attrs: # standard or overwritten lazy
			try:
				attr = getattr(self, name)
			except AttributeError:
				pass
			if isinstance(attr, Lazy):
				attr.unload()

def lightcopy(x):
	if isinstance(x, Lazy):
		y = deepcopy(x)
		y.unload()
		return y
	else:
		return copy(x)
		#raise TypeError('type {} does not implement Lazy'.format(type(x)))

