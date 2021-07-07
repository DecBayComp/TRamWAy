# -*- coding: utf-8 -*-

# Copyright © 2020-2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

from ..attribute import *
from ..artefact import analysis
from .. import attribute
from .abc import *
from tramway.inference import plugins
from tramway.helper.inference import Infer

class MapperAttribute(object):
    """ Mixin class for accessing the `localization_precision`,
    `localization_error` and `temperature` attributes.
    """
    __slots__ = ()

    @property
    def localization_precision(self):
        return self._parent.localization_precision
    @localization_precision.setter
    def localization_precision(self, sigma):
        self._parent.localization_precision = sigma

    @property
    def localization_error(self):
        return self._parent.localization_error
    @localization_error.setter
    def localization_error(self, sigma2):
        self._parent.localization_error = sigma2

    @property
    def temperature(self):
        return self._parent.temperature
    @temperature.setter
    def temperature(self, T):
        self._parent.temperature = T


class MapperPlugin(AnalyzerNode, MapperAttribute):
    __slots__ = ('_name','_module','_setup','_mapper','_kwargs')
    def __init__(self, plugin, **kwargs):
        init_kwargs = {}
        for k in attribute.__analyzer_node_init_args__:
            try:
                arg = kwargs.pop(k)
            except KeyError:
                pass
            else:
                init_kwargs[k] = arg
        AnalyzerNode.__init__(self, **init_kwargs)
        self._name = self._module = self._mapper = None
        self._setup, self._kwargs = {}, {}
        if isinstance(plugin, str):
            self._name = plugin
            try:
                plugin = plugins[plugin]
            except KeyError:
                raise KeyError('no such plugin: {}'.format(self._name))
        if callable(plugin):
            self._setup = kwargs
            self._mapper = plugin
        elif isinstance(plugin, tuple):
            setup, self._module = plugin
            self._setup = dict(setup) # copy
            self._mapper = getattr(self._module, self._setup.pop('infer'))
    @property
    def name(self):
        return self._name
    @property
    def setup(self):
        return self._setup
    def get_plugin_arg(self, attrname):
        try:
            return self._setup[attrname]
        except KeyError:
            return self._kwargs.get(attrname, None)
    def __getattr__(self, attrname):
        return self.get_plugin_arg(attrname)
    def set_plugin_arg(self, attrname, val):
        if attrname in self._setup:
            self._setup[attrname] = val
        else:
            self._kwargs[attrname] = val
            if attrname.endswith('time_prior'):
                self.time.enable_regularization()
    def __setattr__(self, attrname, val):
        try:
            AnalyzerNode.__setattr__(self, attrname, val)
        except AttributeError:
            self.set_plugin_arg(attrname, val)
    @analysis
    def infer(self, sampling):
        helper = Infer()
        helper.prepare_data(sampling)
        distr_kwargs, infer_kwargs = {}, {}
        for k in self._kwargs:
            if k in ('new_cell','new_group','include_empty_cells','grad','rgrad','cell_sampling'):
                distr_kwargs[k] = self._kwargs[k]
            if k not in ('new_cell','new_group','include_empty_cells','grad','cell_sampling'):
                infer_kwargs[k] = self._kwargs[k]
        infer_kwargs['sigma'] = self._parent.spt_data.localization_precision
        if 'cell_sampling' not in distr_kwargs and \
                self.time.initialized and not self.time.regularize_in_time:
            distr_kwargs['cell_sampling'] = 'connected'
        cells = helper.distribute(**distr_kwargs)
        helper.name, helper.setup, helper._infer = self.name, self.setup, self._mapper
        cells = helper.overload_cells(cells)
        maps = helper.infer(cells, **infer_kwargs)
        return maps
    @property
    def time(self):
        return self._parent.time

    @property
    def _mpl_impl(self):
        from .mpl import Mpl
        return Mpl
    @property
    def mpl(self):
        """ tramway.analyzer.mapper.mpl.Mpl: Matplotlib utilities """
        return self._mpl_impl(self)

    @property
    def _plotly_impl(self):
        from .plotly import Plotly
        return Plotly
    @property
    def plotly(self):
        """ tramway.analyzer.mapper.plotly.Plotly: Plotly utilities """
        return self._plotly_impl(self)

Mapper.register(MapperPlugin)

