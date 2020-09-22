# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .abc import *
import numpy as np
from collections.abc import Iterable, Sequence, Set

__all__ = ['GenericAttribute', 'Attribute']

__analyzer_node_init_args__ = ('parent',)

__all__.append('AnalyzerNode')
class AnalyzerNode(object):
    __slots__ = ('_parent',)
    def __init__(self, parent=None):
        self._parent = parent
    @property
    def _eldest_parent(self):
        parent = self
        while True:
            if parent._parent is None:
                break
            else:
                parent = parent._parent
                if not isinstance(parent, AnalyzerNode):
                    #assert isinstance(parent, RWAnalyzer)
                    break
        return parent
    def _bear_child(self, cls, *args, **kwargs):
        assert issubclass(cls, AnalyzerNode)
        kwargs['parent'] = self
        return cls(*args, **kwargs)
    @property
    def initialized(self):
        return True


__all__.append('Initializer')
class Initializer(AnalyzerNode):
    __slots__ = ('_specialize',)
    def __init__(self, attribute_setter, parent=None):
        AnalyzerNode.__init__(self, parent)
        self._specialize = attribute_setter
    def specialize(self, attribute_cls, *attribute_args):
        parent = dict(parent=self._parent)
        self._specialize(attribute_cls(*attribute_args, **parent))
    @property
    def initialized(self):
        return False
    @property
    def reified(self):
        return False

GenericAttribute.register(Initializer)


__all__.append('selfinitializing_property')
def selfinitializing_property(attr_name, getter, setter, metacls=None, doc=None):
    def initializer(self, cls):
        if cls is None:
            setter(self, None)
        elif callable(cls):
            if metacls:
                def typechecked_setter(obj):
                    if not isinstance(obj, metacls):
                        raise TypeError("the argument to attribute '{}' is not a {}".format(attr_name, metacls.__name__))
                    setter(self, obj)
                setter(self, cls( typechecked_setter, parent=self ))
            else:
                setter(self, cls( setter, parent=self ) )
            if not isinstance(getter(self), Initializer):
                raise TypeError("the argument to attribute '{}' is not an Initializer".format(attr_name))
        else:
            raise TypeError("the argument to attribute '{}' is not an Initializer".format(attr_name))
    return property(getter, initializer, doc=doc)


class PipelineMethod(object):
    __slots__ = ('_proc','_args','_kwargs','_enabled','_done')
    def __init__(self, proc):
        self._proc = proc
        self._args = ()
        self._kwargs = {}
        self._enabled = False
        self._done = False
        self._result = None
    def __callable__(self, *args, **kwargs):
        if self.enabled:
            if self.done:
                self._done = args == self._args and kwargs == self._kwargs
            if not self.done:
                self._args = args
                self._kwargs = kwargs
                self._result = self._proc(*args, **kwargs)
                self._done = True
        return self._result
    @property
    def enabled(self):
        return self._enabled
    def enable(self):
        self._enabled = True
    @property
    def disabled(self):
        return not self._enabled
    def disable(self):
        self._enabled = False
    @property
    def done(self):
        return self._done


__all__.append('null_index')
def null_index(i):
    if i is None:
        return True
    elif isinstance(i, Iterable):
        # expect a single-element sequence which value should be 0,
        # idealy of integer type, but no type check is performed here
        # because of the numerous integer types in NumPy
        it = iter(i)
        try:
            i = next(it)
        except StopIteration:
            # empty
            return False
        else:
            if i != 0:
                return False
            try:
                i = next(it)
            except StopIteration:
                return True
            else:
                # multiple indices; return False even if all are null
                return False
    elif callable(i):
        return i(0)
    else:
        return i == 0

__all__.append('indexer')
def indexer(i, it, return_index=False):
    if i is None:
        if return_index:
            yield from enumerate(it)
        else:
            yield from it
    elif callable(i):
        for j, item in enumerate(it):
            if i(j):
                if return_index:
                    yield j, item
                else:
                    yield item
    elif isinstance(i, (Sequence, np.ndarray)):
        # no need for `__getitem__`, but iteration follows the ordering in `i`
        postponed = dict()
        j, it = -1, iter(it)
        for k in i:
            try:
                while j<k:
                    if j in i:
                        postponed[j] = item
                    j, item = j+1, next(it)
            except StopIteration:
                raise IndexError('index is out of bounds: {}'.format(k))
            if j != k:
                try:
                    item = postponed.pop(k)
                except KeyError:
                    if k < 0:
                        raise IndexError('negative values are not supported in a sequence of indices')
                    else:
                        raise IndexError('duplicate index: {}'.format(k))
            if return_index:
                yield k, item
            else:
                yield item
    elif isinstance(i, Set):
        i = set(i) # copy and make mutable
        for j, item in enumerate(it):
            if j in i:
                if return_index:
                    yield j, item
                else:
                    yield item
                i.remove(j)
        if i:
            raise IndexError(('some indices are out of bounds: '+', '.join(['{}'])).format(*tuple(i)))
    elif np.isscalar(i):
        it = iter(it)
        if i == -1:
            try:
                while True:
                    item = next(it)
            except StopIteration:
                pass
        else:
            j = -1
            try:
                while j<i:
                    j, item = j+1, next(it)
            except StopIteration:
                raise IndexError('index is out of bounds: {}'.format(i))
        if return_index:
            yield i, item
        else:
            yield item
    else:
        raise TypeError('unsupported index type')

