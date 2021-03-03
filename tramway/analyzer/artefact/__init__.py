# -*- coding: utf-8 -*-

# Copyright © 2020-2021, Institut Pasteur
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
        if label is None:
            from tramway.analyzer.attribute import single
            try:
                label = single(artefact.parent.labels)
            except RuntimeError:
                raise RuntimeError('undefined label for analysis artefact') from None
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
    def get_child(self, label=None):
        return Analysis.get_analysis(self, label)
    def get_parent(self):
        if isinstance(self._parent, Analysis):
            return self._parent
        else:
            return None
    @classmethod
    def rebase_tree(cls, tree, origin_tree):
        """ Merges `tree` into pre-existing `origin_tree`.

        `tree` is modified in-place.
        As a consequence, the resulting tree is `tree`,
        and NOT `origin_tree`, although the operation
        is conceptually easier to describe as `tree` being
        merged into `origin_tree`, or equivalently
        `origin_tree` being updated with `tree`.

        `tree` takes priority and an artefact in `origin_tree`
        is taken into account only if the corresponding label does
        not already exist in `tree`.

        `rebase_tree` has been designed to rebase `tree` from
        `origin_tree` and pulling any change.
        """
        subtree0, subtree1 = tree, origin_tree
        if subtree0:
            labels0 = set(list(subtree0.labels))
            labels1 = set(list(subtree1.labels))
            for label in labels0 & labels1:
                cls.rebase_tree(subtree0[label], subtree1[label])
            for label in labels1 - labels0:
                subtree0[label] = subtree1[label]
                try:
                    subtree0.comments[label] = subtree1.comments[label]
                except KeyError:
                    pass
        else:
            if subtree1:
                for label in subtree1:
                    subtree0[label] = subtree1[label]
                    try:
                        subtree0.comments[label] = subtree1.comments[label]
                    except KeyError:
                        pass
    @classmethod
    def save(cls, filepath, tree, **kwargs):
        """
        Wrapper for :func:`~tramway.core.hdf5.store.save_rwa` that overrides
        the default values for `compress` (:const:`False`) and `append` (:const:`True`).

        `append=None` is interpreted as `append=False` and a `PendingDeprecationWarning`
        warning is thrown.

        `append` behavior itself is also overriden.
        It is replaced by that of :meth:`rebase_tree`.
        As a consequence, :meth:`save` also understands keyworded argument `rebase`,
        and `append` is treated as a synonym for `rebase`.
        """
        import os.path
        from tramway.core.hdf5.store import load_rwa, save_rwa
        kwargs['compress'] = kwargs.get('compress', False)
        append = kwargs.pop('append', None)
        rebase = kwargs.pop('rebase', None)
        if rebase is None:
            rebase = True if append is None else append
        elif not (append is None or append == rebase):
            raise ValueError('`append` and `rebase` do not have compatible values')
        if rebase and os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
            existing_tree = load_rwa(filepath)
            cls.rebase_tree(tree, existing_tree)
        save_rwa(filepath, tree, **kwargs)


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

