
import sys
from copy import copy, deepcopy
from warnings import warn


def _ro_property_msg(property_name, related_attribute):
	if related_attributed:
		return '`{}` is a read-only property that reflects `{}`''s state'.format(property_name, related_attribute)
	else:
		return '`{}` is a read-only property'.format(property_name)

class PermissionError(AttributeError):
	def __init__(self, property_name, related_attribute):
		RuntimeError.__init__(self, property_name, related_attribute)

	def __str__(self):
		return _ro_property_msg(*self.args)

def ro_property_assert(obj, supplied_value, related_attribute=None, property_name=None, depth=0):
	if property_name is None:
		property_name = sys._getframe(depth + 1).f_code.co_name
	if supplied_value == getattr(obj, property_name):
		warn(_ro_property_msg(property_name, related_attribute), RuntimeWarning)
	else:
		raise PermissionError(property_name, related_attribute)



class Lazy(object):
	__slots__ = ['_lazy']

	__lazy__  = []

	def __init__(self):
		self._lazy = {name: True for name in self.__lazy__}
		for name in self.__lazy__:
			setattr(self, self.__fromlazy__(name), None)

	def __tolazy__(self, name):
		return name[1:]

	def __fromlazy__(self, name):
		return '_{}'.format(name)

	def __setlazy__(self, name, value):
		self._lazy[name] = value is None
		setattr(self, self.__fromlazy__(name), value)

	def __lazysetter__(self, value, depth=0):
		self.__setlazy__(sys._getframe(depth + 1).f_code.co_name, value)

	def __lazyassert__(self, value, related_attribute=None, name=None, depth=0):
		ro_property_assert(self, value, related_attribute, name, depth + 1)

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

