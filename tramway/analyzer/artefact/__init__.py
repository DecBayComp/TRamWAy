# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from functools import wraps

from tramway.core.analyses import abc
#from tramway.core.analyses.base import Analyses
from tramway.core.analyses.lazy import Analyses

import pkg_resources
import platform
import time


def standard_metadata():
    return dict(
        os = platform.system(),
        python = platform.python_version(),
        tramway = pkg_resources.get_distribution('tramway').version,
        datetime = time.strftime('%Y-%m-%d %H:%M:%S UTC%z'),
        )

class Analysis(object):
    __slots__ = ('_parent', 'label', '_data', '_subtree')
    def __init__(self, data, parent=None):
        self._parent = parent
        self.label = None
        self._data = data
        self._subtree = None
    @property
    def parent(self):
        parent = self._parent
        if isinstance(parent, Analysis):
            parent = parent._subtree
        elif not isinstance(parent, abc.Analyses):
            try: # roi
                parent = parent._spt_data
            except AttributeError:
                pass
            parent = parent.analyses
        return parent
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, d):
        self._data = d
        self._parent = self._subtree
        self._subtree = None
        self.label = None
    @property
    def subtree(self):
        return self._subtree
    @property
    def is_input(self):
        return self._subtree is None
    @property
    def is_output(self):
        return not self.is_input
    def commit_as_analysis(self, label, comment=None):
        label = self.parent.autoindex(label)
        subtree = Analyses(self.data, standard_metadata())
        self.parent.add(subtree, label, comment)
        self._subtree = self.parent[label]
        return label
    @classmethod
    def get_analysis(cls, parent, label):
        artefact = cls(None, parent)
        artefact.label = label
        artefact._subtree = artefact.parent[label]
        artefact._data = artefact.subtree.data
        return artefact
    @property
    def analyses(self):
        if self.is_input:
            return self.parent
        else:
            return self.subtree
    def get_child(self, label):
        return Analysis.get_analysis(self, label)
    def get_parent(self):
        if isinstance(self._parent, Analysis):
            return self._parent
        else:
            return None


def _getter(pos, kw=None, default=None):
    if kw is None:
        def _get(_args, _kwargs):
            return _args[pos]
    else:
        def _get(_args, _kwargs):
            try:
                return _args[pos]
            except IndexError:
                return _kwargs.get(kw, default)
    return _get
def _setter(pos, kw=None):
    if kw is None:
        def _set(_args, _kwargs, val):
            return _args[:pos]+(val,)+_args[pos+1:]
    else:
        def _set(_args, _kwargs, val):
            try:
                _args[pos]
            except IndexError:
                _kwargs[kw] = val
            else:
                return _args[:pos]+(val,)+_args[pos+1:]
    return _set

def _wrap_method(met, _get, _set):
    @wraps(met)
    def wrapper_method(self, *args, **kwargs):
        _arg = _get(args, kwargs)
        wrap = isinstance(_arg, Analysis)
        if wrap:
            wrapper = _arg
            args = _set(args, kwargs, wrapper.data)
        res = met(self, *args, **kwargs)
        if wrap:
            res = type(wrapper)(res, wrapper.analyses)
        return res
    return wrapper_method

def _wrap(pos, kw=None):
    return lambda met: _wrap_method(met, _getter(pos, kw), _setter(pos, kw))

def analysis(met_or_pos=None, kw=None):
    if met_or_pos is None:
        if kw is not None:
            raise ValueError('keyword is defined whereas argument position is not')
        return _wrap(0)
    elif isinstance(met_or_pos, int):
        pos = met_or_pos - 1
        if kw is None:
            return _wrap(pos)
        elif isinstance(kw, str):
            return _wrap(pos, kw)
        else:
            raise TypeError('keyword is not a string')
    elif callable(met_or_pos):
        return _wrap_method(met_or_pos, _getter(0), _setter(0))
    else:
        raise TypeError('argument index is not an integer')



def commit_as_analysis(label, analysis, parent=None):
    if not isinstance(analysis, Analysis):
        if parent is None:
            raise ValueError('could not find the analysis tree')
        analysis = Analysis(analysis, parent)
    analysis.commit_as_analysis(label)
    return analysis


__all__ = ['Analysis', 'analysis', 'commit_as_analysis', 'standard_metadata']

