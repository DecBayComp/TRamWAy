# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import sys
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
    """Lazy store.

    Lazily computes and stores attributes through properties, so that the stored attributes can be
    (explicitly) deleted anytime to save memory.

    The :attr:`__lazy__` static attribute is a list of the properties that implement such a
    mechanism.

    Per default each lazy property ``name`` manages a private ``_name`` attribute.
    This naming convention can be overwritten by heriting `Lazy` and overloading
    :meth:`__tolazy__` and :meth:`__fromlazy__` methods.

    An unset lazy attribute/property always has value ``None``.

    A getter will typically look like this:

    .. code-block:: python

        @property
        def name(self):
            if self._name is None:
                self._name = # add some logics
            return self.__lazyreturn__(self._name)

    A fully functional setter will typically look like this:

    .. code-block:: python

        @name.setter
        def name(self, value):
            self.__lazysetter__(value)

    A read-only lazy property will usually look like this:

    .. code-block:: python

        @name.setter
        def name(self, value):
            self.__lazyassert__(value)

    `__lazyassert__` can unset ``_name`` (set it to ``None``) but any other value is treated as
    illegal. `__lazyassert__` compares ``value`` with ``self.name`` and raises a warning if the
    values equal to each other, or throws an exception otherwise.

    """
    __slots__ = ('_lazy',)

    __lazy__  = ()

    def __init__(self):
        self._lazy = {name: True for name in self.__lazy__}
        for name in self.__lazy__:
            setattr(self, self.__fromlazy__(name), None)

    def __returnlazy__(self, name, value):
        return value

    def __lazyreturn__(self, value, depth=0):
        #caller = sys._getframe(depth + 1).f_code.co_name
        #return self.__returnlazy__(caller, value)
        return value

    def __tolazy__(self, name):
        """Returns the property name that corresponds to an attribute name."""
        return name[1:]

    def __fromlazy__(self, name):
        """Returns the attribute name that corresponds to a property name."""
        return '_{}'.format(name)

    def __setlazy__(self, name, value):
        """Sets property `name` to `value`."""
        self._lazy[name] = value is None
        setattr(self, self.__fromlazy__(name), value)

    def __lazysetter__(self, value, depth=0):
        """Sets the property which name is the name of the caller."""
        caller = sys._getframe(depth + 1).f_code.co_name
        self.__setlazy__(caller, value)

    def __assertlazy__(self, name, value, related_attribute=None):
        if value is None: # None has a special meaning for lazy attributes/properties
            self.__setlazy__(name, value)
        else:
            ro_property_assert(self, value, related_attribute, name)

    def __lazyassert__(self, value, related_attribute=None, name=None, depth=0):
        if value is None: # None has a special meaning for lazy attributes/properties
            if name is None:
                self.__lazysetter__(value, depth + 1)
            else:
                self.__setlazy__(name, value)
        else:
            ro_property_assert(self, value, related_attribute, name, depth + 1)

    def unload(self, visited=None):
        """
        Recursively clear the lazy attributes.

        Beware: only direct Lazy object attributes are unloaded,
            not Lazy objects stored in non-lazy attributes!

        *Deprecated*
        """
        warn('`unload` has been removed and will raise an exception in future versions', DeprecationWarning)
        if visited is None:
            visited = set()
        elif id(self) in visited:
            # already unloaded
            return
        visited.add(id(self))
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
        # search for Lazy object attributes so that they can be unloaded
        for name in standard_attrs: # standard or overwritten lazy
            try:
                attr = getattr(self, name)
            except AttributeError:
                pass
            if isinstance(attr, Lazy):
                attr.unload(visited)



def lightcopy(x):
    """
    Return a copy and call `unload` if available.

    Arguments:

        x (any): object to be copied and unloaded.

    Returns:

        any: copy of `x`.

    *Deprecated*
    """
    warn('`lightcopy` has been removed and will raise an exception in future versions', DeprecationWarning)
    return x


__all__ = [
    'PermissionError',
    'ro_property_assert',
    'Lazy',
    'lightcopy',
    ]

