# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import *
from ..artefact import analysis
from .abc import *
from tramway.helper.tessellation import Tessellate
from copy import deepcopy
from warnings import warn
from .post import TessellationPostProcessingInitializer


def proxy_property(propname, level='default', doc=None):
    """
    Property factory function similar to builtin *property*,
    dedicated to the :class:`TessellerProxy` class.

    For :const:`'tessellate'` level properties, the default level is safe.

    For :const:`'__init__'` level properties, the level must be specified.

    For standard attributes, it is safer to make the level explicit,
    so that some conflicts may be detected earlier (e.g. at module loading).
    """
    if level == 'tessellate':
        def _get(obj):
            return obj._tessellate_kwargs[propname]
        def _set(obj, val):
            assert propname in obj._tessellate_kwargs
            obj._tessellate_kwargs[propname] = val
            obj._explicit_kwargs[propname] = val
    elif level == '__init__':
        def _get(obj):
            return obj._init_kwargs.get(propname, None)
        def _set(obj, val):
            assert propname not in obj._tessellate_kwargs
            obj._init_kwargs[propname] = val
            obj._explicit_kwargs[propname] = val
            obj.reset()
    elif level == 'attr':
        def _get(obj):
            return getattr(obj.tesseller, propname)
        def _set(obj, val):
            setattr(obj.tesseller, propname, val)
            obj._explicit_kwargs[propname] = val
    else:
        def _get(obj):
            return TessellerProxy.__getattr__(obj, propname)
        def _set(obj, val):
            TessellerProxy.__setattr__(obj, propname, val)
    return property(_get, _set, doc=doc)


class TessellerProxy(AnalyzerNode):
    """
    Encapsulates a tessellation plugin as defined in the :mod:`tramway.tessellation` package.

    Attributes are managed with the following rules:

    * if an attribute is set using a proxy property,
      the value is flagged as :const:`'explicit'` and overrides any default value;
    * the arguments in :attr:`_init_kwargs` are passed to the wrapped tesseller's :meth:`__init__`;
      these arguments are intended to prevent :meth:`__init__` from crashing if the later requires
      some arguments to be defined; non-explicit values are likely to be overriden and
      both explicit and non-explicit values may be altered by :meth:`__init__`;
    * the arguments in :attr:`_tessellate_kwargs` are passed to the wrapped tesseller's
      :meth:`tessellate` method; all available arguments should be defined in the proxy's
      :meth:`__init__`, with default values;
    * the wrapped tesseller's attributes can be accessed using proxy properties;
      as a consequence, any key in :attr:`_explicit_kwargs`, that is not in :attr:`_init_kwargs`
      or :attr:`_tessellate_kwargs`, is considered as an actual attribute;
    * an argument should not be defined both in :attr:`_init_kwargs` and :attr:`_tessellate_kwargs`;
    * explicit :meth:`__init__` arguments take precedence over standard attributes;

    """
    __slots__ = ('_tesseller', '_init_kwargs', '_tessellate_kwargs', '_explicit_kwargs', 'alg_name',
            '_post_processing')
    def __init__(self, cls, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._tesseller = cls
        self._reset_kwargs()
        self._explicit_kwargs = {}
        self.alg_name = None
        # self-initializing properties
        self._post_processing = None
        self.post_processing  = TessellationPostProcessingInitializer
    def _reset_kwargs(self):
        self._init_kwargs = dict(ref_distance=None)
        self._tessellate_kwargs = {}
    @property
    def cls(self):
        if isinstance(self._tesseller, type):
            return self._tesseller
        elif isinstance(self._tesseller, tuple):
            return type(self._tesseller[0])
        else:
            return type(self._tesseller)
    def _get_tesseller(self):
        if isinstance(self._tesseller, type):
            cls = self._tesseller
            self._tesseller = cls(**self._init_kwargs)
            for attr in self._iter_explicit_attributes():
                setattr(self._tesseller, attr, self._explicit_kwargs[attr])
        if isinstance(self._tesseller, tuple):
            return self._tesseller[0]
        else:
            return self._tesseller
    tesseller = property(_get_tesseller)
    @property
    def reified(self):
        return isinstance(self._tesseller, tuple)
    def reset(self, clear_explicit_attrs=False):
        """ resets the tesseller to let the calibration parameters vary.
        """
        if not isinstance(self._tesseller, type):
            self._tesseller = self.cls
        self._reset_kwargs()
        if clear_explicit_attrs:
            self._explicit_kwargs = {}
        else:
            for k in self._explicit_kwargs:
                if k in self._tessellate_kwargs:
                    self._tessellate_kwargs[k] = self._explicit_kwargs[k]
                if k in self._init_kwargs:
                    self._init_kwargs[k] = self._explicit_kwargs[k]
    def _iter_explicit_attributes(self):
        for attr in self._explicit_kwargs:
            if not (attr in self._init_kwargs or attr in self._tessellate_kwargs):
                yield attr
    def calibrate(self, spt_dataframe):
        if not isinstance(self._tesseller, type):
            self.reset()
        helper = Tessellate()
        helper.prepare_data(spt_dataframe)
        #
        kwargs = dict(self._init_kwargs)
        kwargs.update(self._tessellate_kwargs)
        helper.setup = dict(make_arguments=kwargs)
        #
        kwargs = helper.standard_parameters(self._init_kwargs.pop('ref_distance', None),
                rel_min_distance=self._init_kwargs.pop('rel_min_distance', None),
                rel_avg_distance=self._init_kwargs.pop('rel_avg_distance', None),
                rel_max_distance=self._init_kwargs.pop('rel_max_distance', None),
                min_location_count=self._init_kwargs.pop('min_location_count', None),
                avg_location_count=self._init_kwargs.pop('avg_location_count', None),
                max_location_count=self._init_kwargs.pop('max_location_count', None),
                )
        for k in self._explicit_kwargs:
            if k in kwargs:
                kwargs[k] = self._explicit_kwargs[k]
        helper.parse_args(kwargs)
        #
        for k,v in helper.tessellation_kwargs.items():
            #if v is None:
            #    continue
            if k in self._tessellate_kwargs:
                self._tessellate_kwargs[k] = v
            self._init_kwargs[k] = v
        if helper.scaler is not None:
            self._init_kwargs['scaler'] = helper.scaler
        #
        assert isinstance(self._tesseller, type)
        self._tesseller = (self.tesseller, helper.colnames)
    @analysis
    def tessellate(self, spt_dataframe):
        """ Grows and returns the tessellation.
        """
        if not isinstance(self._tesseller, tuple):
            self.calibrate(spt_dataframe)
        tesseller = deepcopy(self.tesseller)
        tesseller.tessellate(spt_dataframe[self.colnames], **self._tessellate_kwargs)
        #
        if self.post_processing.initialized:
            tesseller = self.post_processing.post_process(tesseller, spt_dataframe[self.colnames])
        #
        return tesseller
    @property
    def colnames(self):
        if not isinstance(self._tesseller, tuple):
            raise RuntimeError('the tesseller is not calibrated')
        return self._tesseller[1]
    def bc_update_params(self, params):
        if self.alg_name is not None:
            params['method'] = self.alg_name
        if self._init_kwargs:
            exclude = ('scaler',)
            params['tessellation'] = { k:self._init_kwargs[k]
                    for k in self._init_kwargs
                    if k not in exclude }

    scaler          = proxy_property('scaler',           'attr')
    ref_distance    = proxy_property('ref_distance', '__init__')

    @property
    def resolution(self):
        if self.ref_distance is None:
            return None
        else:
            return 2. * self.ref_distance
    @resolution.setter
    def resolution(self, res):
        if res is None:
            self.ref_distance = None
        else:
            self.ref_distance = .5 * res

    def _get_post_processing(self):
        return self._post_processing
    def _set_post_processing(self, merger):
        self._post_processing = merger
    post_processing = selfinitializing_property('post_processing', _get_post_processing, _set_post_processing, TessellationPostProcessing)

    @property
    def _mpl_impl(self):
        from .mpl import Mpl
        return Mpl
    @property
    def mpl(self):
        """ tramway.analyzer.tesseller.mpl.Mpl: Matplotlib utilities """
        return self._mpl_impl(self)

    def __getattr__(self, attrname):
        """ Beware that it ignores :attr:`_init_kwargs` symbols;
        :meth:`__init__` arguments should be made available defining a proxy property with
        the :const:`'__init__'` flag."""
        try:
            val = self._tessellate_kwargs[attrname]
        except KeyError:
            val = getattr(self.tesseller, attrname)
        return val
    def __setattr__(self, attrname, val):
        """ Beware that it ignores :attr:`_init_kwargs` symbols.

        Important note: __slots__ must be defined instead of __dict__
            so that object.__setattr__ fails with AttributeError.
        """
        # special setters for self-initializing properties
        if attrname == 'post_processing' and isinstance(self.post_processing, Initializer):
            self.post_processing.from_callable(val)
            return
        #
        try:
            AnalyzerNode.__setattr__(self, attrname, val)
        except AttributeError:
            if attrname in ('cls','tesseller','post_processing','mpl'):
                raise AttributeError(attrname+' is read-only')
            if attrname in self._tessellate_kwargs:
                self._tessellate_kwargs[attrname] = val
            else:
                try:
                    setattr(self.tesseller, attrname, val)
                except AttributeError:
                    self._init_kwargs[attrname] = val
            self._explicit_kwargs[attrname] = val

Tesseller.register(TessellerProxy)



__all__ = ['TessellerProxy', 'proxy_property']

