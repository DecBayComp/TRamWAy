# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from . import abc
from . import base
from . import lazy
import warnings
import os.path


class EventDict(dict):
    """ dictionnary that can trigger an event on `__setitem__`. """
    def __init__(self, d, preprocess, postprocess):
        dict.__init__(self, d)
        self.preprocess = preprocess
        self.postprocess = postprocess
    def __setitem__(self, k, v):
        dict.__setitem__(self, k, self.preprocess(v))
        self.postprocess(self, k)


class WithState(object):
    """ complementary class for flagging an analysis as modified.
    
    Any slotted child class should define attribute `_modified` """
    __slots__ = ()
    def __init__(self, *args, cls=base.Analyses):
        cls.__init__(self, *args)
        self._modified = False
    def flag_as_modified(self):
        self._modified = True
    def reset_modification_flag(self, recursive=False):
        self._modified = False
        if recursive:
            for child in self.instances.values():
                child.reset_modification_flag(True)
    def modified(self, recursive=False):
        if self._modified:
            return True
        elif recursive:
            return any( child.modified(True) for child in self.instances.values() )


def with_state(cls, handler):
    """ derivate class `cls` so that it also inherits from `WithState`.

    If `cls` is a class instance instead, then it copies `cls` into an object
    that inhertits from `type(cls)` and `WithState`.

    The resulting class or class instance features a `statefree` method that
    returns a copy of `self` in the original parent type.
    """
    if isinstance(cls, type):
        assert cls in (base.Analyses, lazy.Analyses)
        class AnalysesWithState(cls, WithState):
            __slots__ = ('_modified',)
            def __init__(self, data=None, metadata=None):
                WithState.__init__(self, data, metadata, cls=base.Analyses)
                self._instances = EventDict(self._instances, handler.preprocess(self), handler.postprocess(self))
            def statefree(self):
                analyses = cls(self.data, self.metadata)
                analyses._comments = self._comments
                analyses._instances = { label: self[label].statefree() for label in self.labels }
                return analyses
        return AnalysesWithState
    else:
        analyses = cls
        if isinstance(analyses, Analyses):
            analyses = analyses.analyses
            assert not isinstance(analyses, Analyses)
            return analyses
        cls = with_state(type(analyses), handler)
        def _with_state(_analyses):
            assert isinstance(_analyses, (base.Analyses, lazy.Analyses))
            _state = cls(_analyses._data, _analyses._metadata)
            _state._comments = _analyses._comments
            _state._instances = EventDict(
                    { _label: _with_state(_analyses[_label]) for _label in _analyses.labels },
                    handler.preprocess(_state), handler.postprocess(_state),
                    )
            return _state
        return _with_state(analyses)


class AnalysesProxy(object):
    """ proxy for analyses objects """
    __slots__ = ('_analyses',)
    def __init__(self, data=None, metadata=None, cls=base.Analyses):
        if isinstance(data, cls):
            self.analyses = data
            if metadata:
                warnings.warn('ignoring argument `metadata`')
        else:
            self.analyses = cls(data, metadata)
    @property
    def analyses(self):
        return self._analyses
    @analyses.setter
    def analyses(self, a):
        self._analyses = a
    @property
    def data(self):
        return self.analyses.data
    @data.setter
    def data(self, d):
        self.analyses.data = d
    @property
    def _data(self):
        return self.analyses._data
    @_data.setter
    def _data(self, d):
        self.analyses._data = d
    @property
    def artefact(self):
        return self.analyses.artefact
    @artefact.setter
    def artefact(self, a):
        self.analyses.artefact = a
    @property
    def metadata(self):
        return self.analyses.metadata
    @metadata.setter
    def metadata(self, d):
        self.analyses.metadata = d
    @property
    def instances(self):
        return self.analyses.instances
    @property
    def _instances(self):
        return self.analyses._instances
    @property
    def comments(self):
        return self.analyses.comments
    @property
    def _comments(self):
        return self.analyses._comments
    @property
    def labels(self):
        return self.analyses.labels
    def keys(self):
        return self.analyses.keys()
    def autoindex(self, pattern=None):
        return self.analyses.autoindex(pattern)
    def add(self, analysis, label=None, comment=None, raw=False):
        self.analyses.add(analysis, label, comment, raw)
    def __nonzero__(self):
        return self.analyses.__nonzero__()
    def __len__(self):
        return self.analyses.__len__()
    def __missing__(self, label):
        return self.analyses.__missing__(label)
    def __iter__(self):
        return self.analyses.__iter__()
    def __contains__(self, label):
        return self.analyses.__contains__(label)
    def __getitem__(self, label):
        return self.analyses.__getitem__(label)
    def __setitem__(self, label, analysis):
        self.analyses.__setitem__(label, analysis)
    def __delitem__(self, label):
        self.analyses.__delitem__(label)
    def __str__(self):
        return self.analyses.__str__()

abc.Analyses.register(AnalysesProxy)

class LazyAnalysesProxy(AnalysesProxy):
    __slots__ = ()
    def __init__(self, data=None, metadata=None, cls=lazy.Analyses):
        AnalysesProxy.__init__(self, data, metadata, cls)
    @property
    def type(self):
        return self.analyses.type
    def terminate(self):
        self.analyses.terminate()


__autosavecapable_slots__ = ('_default_autosave_policy','_active_autosave_policy')
class AutosaveCapable(object):
    """
    Abstract class.

    Children classes, if slotted, should define:
    
    * `_default_autosave_policy` (bool or str),
    * `_active_autosave_policy` (bool or str),

    An autosave policy can take various values if of ``str`` type:

    * *'on completion'* saves once on normal completion of the entire process (default)
    * *'on termination'* saves once on termination of the entire process, whether it is successful or not
    * *'on every step'* saves on every successful step

    If `_active_autosave_policy` is ``True``, then `_default_autosave_policy` applies.

    """
    __slots__ = ()
    def __init__(self, autosave=True):
        """
        Arguments:

            autosave (bool or str): default autosaving policy

        """
        self._default_autosave_policy = autosave
        self._active_autosave_policy = None
    @property
    def autosave(self):
        if self._active_autosave_policy is None or self._active_autosave_policy is True:
            return bool(self._default_autosave_policy)
        else:
            return bool(self._active_autosave_policy)
    @autosave.setter
    def autosave(self, flag):
        if not (isinstance(flag, bool) or isinstance(flag, str)):
            raise TypeError('autosave supports only boolean and string values')
        self._default_autosave_policy = flag
    @property
    def autosave_policy(self):
        if isinstance(self._active_autosave_policy, bool):
            if self._default_autosave_policy is False:
                return None
            elif self._default_autosave_policy is True:
                return 'on completion'
            else:
                return self._default_autosave_policy
        else:
            return self._active_autosave_policy
    @property
    def active(self):
        return bool(self._active_autosave_policy)
    @property
    def save_on_completion(self):
        policy = self.autosave_policy
        return policy and \
                (policy.endswith('completion') or policy.endswith('termination'))
    @property
    def save_on_every_step(self):
        policy = self.autosave_policy
        return policy and policy.endswith('step')
    @property
    def force_save(self):
        policy = self.autosave_policy
        return policy and policy.endswith('termination')
    def __enter__(self):
        self.reset_modification_flag(True)
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.modified(True):
            self.reset_modification_flag(True)
            if self.force_save or (exc_type is None and self.save_on_completion):
                self.save()
        self._active_autosave_policy = None
    def autosaving(self, policy=None):
        if policy is None and self.autosave:
            policy = True
        self._active_autosave_policy = policy
        return self
    def save(self):
        """ should call `self.reset_modification_flag(True)` """
        raise NotImplementedError('abstract method')


class Analyses(LazyAnalysesProxy, AutosaveCapable):
    """ autosaving analyses. 
    
    Argument and attribute `rwa_file` designate the output file."""
    __slots__ = __autosavecapable_slots__ + ('rwa_file','save_options','hooks')
    def __init__(self, data=None, metadata=None, rwa_file=None, autosave=False):
        if isinstance(data, AutosaveCapable):
            raise TypeError('nested autosave-capable objects')
        LazyAnalysesProxy.__init__(self, data, metadata)
        AutosaveCapable.__init__(self, autosave)
        self.rwa_file = rwa_file
        self.save_options = dict(force=True, compress=False)
        self.hooks = []
    @property
    def analyses(self):
        return self._analyses
    @analyses.setter
    def analyses(self, a):
        self._analyses = with_state(a, self.handler)
    def add(self, analysis, label=None, comment=None, raw=False):
        if not (raw or isinstance(analysis, abc.Analyses)):
            analysis = lazy.Analyses(analysis)
        if isinstance(analysis, abc.Analyses) and not isinstance(analysis, WithState):
            analysis = with_state(analysis, self.handler)
        self.analyses.add(analysis, label, comment, raw)
    @classmethod
    def from_rwa_file(cls, input_file, output_file=None, **kwargs):
        from tramway.core.hdf5.store import load_rwa
        if output_file is None:
            output_file = input_file
        analyses = load_rwa(input_file, lazy=True,
                force_load_spt_data=False)
        analyses = cls(analyses, rwa_file=output_file, **kwargs)
    def save(self, out_of_context=False):
        if not (out_of_context or self.active):
            raise RuntimeError("method 'save' called from outside the context")
        for hook in self.hooks:
            hook(self.analyses)
        if self.rwa_file:
            from tramway.core.hdf5.store import save_rwa
            save_rwa(os.path.expanduser(self.rwa_file), self.analyses.statefree(), **self.save_options)
            self.analyses.reset_modification_flag(True)
        else:
            warnings.warn('no output file defined', RuntimeWarning)
    # WithState proxy
    def flag_as_modified(self):
        self.analyses.flag_as_modified()
    def modified(self, recursive=False):
        return self.analyses.modified(recursive)
    def reset_modification_flag(self, recursive=False):
        self.analyses.reset_modification_flag(recursive)
    # handler methods; not for direct use
    def preprocess(self, analyses):
        def _preprocess(analysis):
            if isinstance(analysis, abc.Analyses) and not isinstance(analysis, WithState):
                analysis = with_state(analysis, self.handler)
            return analysis
        return _preprocess
    def postprocess(self, analyses):
        def _postprocess(*args):
            if self.active:
                if self.save_on_every_step:
                    self.save()
                else:
                    analyses.flag_as_modified()
        return _postprocess
    @property
    def handler(self):
        class PP(object):
            def __init__(_self):
                _self.preprocess = self.preprocess
                _self.postprocess = self.postprocess
        return PP()
    def statefree(self):
        return self.analyses.statefree()

__all__ = ['Analyses', 'AutosaveCapable']

