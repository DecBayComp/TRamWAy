# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import rwa.lazy as rwa
from . import base
from . import abc


class InstancesView(base.InstancesView):

    __slots__ = ('peek', )

    def __init__(self, analyses, peek=False):
        base.InstancesView.__init__(self, analyses)
        self.peek = peek

    def __getitem__(self, label):
        instance = base.InstancesView.__getitem__(self, label)
        if rwa.islazy(instance):
            if issubclass(instance.type, abc.Analyses):
                instance = instance.shallow()
                self.__setitem__(label, instance)
            elif self.peek:
                instance = instance.deep()
                self.__setitem__(label, instance)
        return instance

    def get(self, label, default=None):
        instance = base.InstancesView.get(self, label, None)
        if instance is None:
            instance = default
        elif rwa.islazy(instance):
            if issubclass(instance.type, abc.Analyses):
                instance = instance.shallow()
                self.__setitem__(label, instance)
            elif self.peek:
                instance = instance.deep()
                self.__setitem__(label, instance)
        return instance

    def pop(self, label, default=None):
        return rwa.lazyvalue(base.InstancesView.pop(self, label, default), deep=self.peek)


class Analyses(base.Analyses):
    __slots__ = ()

    @property
    def type(self):
        return rwa.lazytype(self._data)

    @property
    def data(self):
        if rwa.islazy(self._data):
            self._data = self._data.peek(deep=True)
        return self._data

    # copy/paste
    @data.setter
    def data(self, d):
        self._data = d
        self._instances = {}
        self._comments = {}
        self._metadata = {}

    @property
    def comments(self):
        if rwa.islazy(self._comments):
            self._comments = self._comments.deep()
            if rwa.islazy(self._metadata):
                self._metadata = self._metadata.deep()
            if rwa.islazy(self._instances):
                self._instances = self._instances.shallow()
        return base.CommentsView(self)

    @property
    def metadata(self):
        if rwa.islazy(self._metadata):
            self._metadata = self._metadata.deep()
            if rwa.islazy(self._comments):
                self._comments = self._comments.deep()
            if rwa.islazy(self._instances):
                self._instances = self._instances.shallow()
        if self._metadata is None:
            self._metadata = {}
        return self._metadata

    # copy/paste
    @metadata.setter
    def metadata(self, d):
        self._metadata = {} if d is None else d

    @property
    def instances(self):
        if rwa.islazy(self._instances):
            self._instances = self._instances.shallow()
            if rwa.islazy(self._comments):
                self._comments = self._comments.deep()
            if rwa.islazy(self._metadata):
                self._metadata = self._metadata.deep()
        return InstancesView(self)

    def __str__(self):
        return base.format_analyses(self, node=rwa.lazytype)

    def terminate(self):
        """
        Close the opened file if any and delete all the handles.
        """
        def _terminate(obj, ok=False):
            if rwa.islazy(obj):
                if ok:
                    obj.store.handle = None
                else:
                    obj.store.close()
                    ok = True
            elif isinstance(obj, abc.Analyses):
                obj = obj._instances
                if isinstance(obj, dict): # implicit: not rwa.islazy(obj)
                    for k in obj:
                        ok |= _terminate(obj[k], ok)
                else:
                    ok |= _terminate(obj, ok)
            return ok
        _terminate(self)



def label_paths(analyses, filter, lazy=False):
    """
    Find label paths for analyses matching a criterion.

    Arguments:

        analyses (tramway.core.analyses.base.Analyses):
            hierarchy of analyses, with `instances` possibly containing
            other :class:`~tramway.core.analyses.base.Analyses` instances.

        filter (type or callable):
            criterion over analysis data.

        lazy (bool):
            if applying `filter` function to a :class:`rwa.lazy.LazyPeek`,
            whether to pass the lazy or the evaluated form.

    Returns:

        list of tuples:
            list of label paths to matching analyses.

    """
    if isinstance(filter, type):
        _type = filter
        filter = lambda node: issubclass(rwa.lazytype(node), _type)
    elif callable(filter) and not lazy:
        _filter = filter
        filter = lambda node: _filter(rwa.lazyvalue(node, deep=True))
    return base.label_paths(analyses, filter)


def find_artefacts(analyses, filters, labels=None, quantifiers=None, lazy=False, return_subtree=False):
    """
    Find related artefacts.

    Filters are applied to find data elements (artefacts) along a single path specified by `labels`.

    Arguments:

        analyses (tramway.core.analyses.base.Analyses): hierarchy of analyses.

        filters (type or callable or tuple or list): list of criteria, a criterion being
            a boolean function or a type.

        labels (list): label path.

        quantifiers (str or tuple or list): list of quantifers, a quantifier for now being
            either '*first*', '*last*' or '*all*'; a quantifier should be defined for each
            filter; default is '*last*' (admits value ``None``).

        lazy (bool):
            if applying a filter function to a :class:`rwa.lazy.LazyPeek`,
            whether to pass the lazy or the evaluated form.

        return_subtree (bool): return as extra output argument the analysis subtree corresponding
            to the deepest matching artefact.

    Returns:

        tuple: matching data elements/artefacts, and optionally analysis subtree.

    Examples:

        .. code-block:: python

            cells, maps = find_artefacts(analyses, (CellStats, Maps))

            maps, maps_subtree = find_artefacts(analyses, Maps, return_subtree=True)

    """
    if not isinstance(filters, (tuple, list)):
        filters = (filters,)
    fullnode = lazy or all( isinstance(f, (type, tuple, list)) for f in filters )
    if fullnode:
        # force closure at definition time (otherwise `t` and `f` are overwritten)
        def typefilter(t):
            return lambda a: issubclass(rwa.lazytype(a._data), t)
        def directfilter(f):
            return lambda a: f(rwa.lazyvalue(a, deep=True))
        _filters = []
        for _filter in filters:
            if isinstance(_filter, (type, tuple, list)):
                _type = _filter
                _filter = typefilter(_type)
            elif callable(_filter) and not lazy:
                _filter = directfilter(_filter)
            _filters.append(_filter)
        filters = _filters
    artefacts = base.find_artefacts(analyses, filters, labels, quantifiers, fullnode,
        return_subtree)
    if lazy:
        return artefacts
    else:
        if return_subtree:
            artefacts = list(artefacts)
            subtree = artefacts.pop()
        artefacts = [ rwa.lazyvalue(a, deep=True) for a in artefacts ]
        if return_subtree:
            artefacts.append(subtree)
        return tuple(artefacts)


__all__ = [
    'InstancesView',
    'Analyses',
    'find_artefacts',
    'label_paths',
    ]


