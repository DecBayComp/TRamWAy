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
            if isinstance(cls, InitializerMethod):
                cls.assign( self )
                return
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
def indexer(i, it, return_index=False, has_keys=False):
    if i is None:
        if return_index:
            if has_keys:
                for k in it:
                    yield k, it[k]
            else:
                yield from enumerate(it)
        else:
            if has_keys:
                for k in it:
                    yield it[k]
            else:
                yield from it
    elif callable(i):
        if has_keys:
            if return_index:
                for j in it:
                    if i(j):
                        yield j, item
            else:
                for j in it:
                    if i(j):
                        yield item
        else:
            if return_index:
                for j, item in enumerate(it):
                    if i(j):
                        yield j, item
            else:
                for j, item in enumerate(it):
                    if i(j):
                        yield item
    elif isinstance(i, (Sequence, np.ndarray)):
        if has_keys:
            if return_index:
                for k in i:
                    yield k, it[k]
            else:
                for k in i:
                    yield it[k]
        else:
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
        if has_keys:
            for j in it:
                if j in i:
                    if return_index:
                        yield j, it[j]
                    else:
                        yield it[j]
                    i.remove(j)
        else:
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
        if has_keys:
            try:
                item = it[i]
            except KeyError:
                raise IndexError('index is out of bounds: {}'.format(i)) from None
        else:
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


__all__.append('first')
def first(it, **kwargs):
    if callable(it):
        it = it(**kwargs)
    return next(iter(it))


__all__.append('single')
def single(it, **kwargs):
    if callable(it):
        it = it(**kwargs)
    it = iter(it)
    elem = next(it)
    try:
        next(it)
    except StopIteration:
        # good
        pass
    else:
        raise RuntimeError('not a singleton') from None
    return elem


__all__.append('Proxy')
class Proxy(object):
    __slots__ = ('_proxied',)
    def __init__(self, proxied):
        self._proxied = proxied
    def __len__(self):
        return self._proxied.__len__()
    def __iter__(self):
        return self._proxied.__iter__()
    @property
    def _parent(self):
        return self._proxied._parent
    @_parent.setter
    def _parent(self, par):
        self._proxied._parent = par
    def __getattr__(self, attrname):
        return getattr(self._proxied, attrname)
    def __setattr__(self, attrname, val):
        if attrname == '_proxied':
            object.__setattr__(self, '_proxied', val)
        else:
            setattr(self._proxied, attrname, val)


__all__.append('InitializerMethod')
class InitializerMethod(object):
    """
    Useful for explicit typing.
    """
    __slots__ = ('attrname', 'method', 'args', 'kwargs')
    def __init__(self, method, attrname=None):
        self.method = method
        if attrname is None:
            attrname = method.__module__.split('.')[-1]
        self.attrname = attrname
        self.args = self.kwargs = None
    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self
    def assign(self, parent_analyzer):
        if not (isinstance(self.args, tuple) and isinstance(self.kwargs, dict)):
            raise RuntimeError('initializer method has not been called')
        parent_attribute = getattr(parent_analyzer, self.attrname)
        self.method(parent_attribute, *self.args, **self.kwargs)
    def reassign(self, parent_analyzer):
        import importlib
        mod = importlib.import_module(self.method.__module__)
        try:
            clsname = str(self.method)[::-1].split('.', 1)[-1][::-1].split()[-1]
            cls = getattr(mod, clsname)
            assert isinstance(cls, type) and issubclass(cls, Initializer)
        except:
            raise AttributeError("cannot find a proper initializer for the '{}' attribute".format(self.attrname))
        attr = getattr(type(parent_analyzer), self.attrname)
        # TODO: add a freset method to the properties returned by selfinitializing_property
        attr.fset(parent_analyzer, cls)
        self.assign( parent_analyzer )

