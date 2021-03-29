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
from ..artefact import analysis, Analysis
from .. import attribute
from .abc import *
#from . import stdalg as mappers
from tramway.inference import plugins
from tramway.helper.inference import Infer


class MapperInitializer(Initializer):
    __slots__ = ()
    def from_plugin(self, plugin, **kwargs):
        self.specialize( MapperPlugin, plugin, **kwargs )
    def from_callable(self, cls, **kwargs):
        self.from_plugin(cls, **kwargs)
    def from_maps(self, label_or_maps=None, exclude_attrs=(), verbose=False):
        """
        Sets the mapper up on basis of parameters in a Maps object.

        If a label is passed instead of a Maps object (or nothing),
        the corresponding artefact is sought for in the analysis tree,
        starting from the first branch (first sampling) in the first
        tree.

        This method has been designed for the *stochastic.dv* plugin.
        Application to other plugins has not been tested.

        Arguments:

            label_or_maps (*Maps* or *Analysis* or *int* or *str* or *tuple* or *list*):
                maps object or label(s) of the maps artefact to be found in any
                registered analysis tree

            exclude_attrs (iterable):
                attributes of the maps object to be ignored

            verbose (bool):
                prints the plugin name and loaded attributes

        """
        if label_or_maps is None or isinstance(label_or_maps, (int, str)):
            label = label_or_maps
            sampling = first(self._eldest_parent.roi.as_support_regions()).get_sampling()
            maps = sampling.get_child(label)
        elif isinstance(label_or_maps, (tuple, list)):
            sampling_label, labels = label_or_maps[0], label_or_maps[1:]
            sampling = first(self._eldest_parent.roi.as_support_regions()).get_sampling(sampling_label)
            #
            node = sampling
            for label in labels:
                node = node.get_child(label)
            maps = node
        else:
            maps = label_or_maps
        if isinstance(maps, Analysis):
            maps = maps.data
        plugin = maps.mode
        if plugin is None:
            raise ValueError('undefined plugin name')
        self.from_plugin(plugin)
        self = self._eldest_parent.mapper
        exclude_attrs = list(exclude_attrs) # copy
        exclude_attrs += ['mode', 'runtime', 'posteriors', 'niter', 'resolution', 'sigma']
        attrs = []
        for attr in maps.__dict__:
            if not (attr[0] == '_' or attr in exclude_attrs):
                val = getattr(maps, attr)
                if val is not None:
                    setattr(self, attr, val)
                    attrs.append((attr, val))
        if verbose:
            logger = self._eldest_parent.logger
            if attrs:
                logger.info("loading plugin: '{}'\nwith parameters:\n - {}".format(
                    plugin,
                    "\n - ".join([
                        ("{}: '{}'" if isinstance(val, str) else "{}: {}").format(key, val) \
                                for key, val in attrs
                    ])))
            else:
                logger.info("plugin:\t'{}'\nwith no parameters".format(plugin))

    @property
    def _mpl_impl(self):
        from .mpl import Mpl
        return Mpl
    @property
    def mpl(self):
        """ tramway.analyzer.mapper.mpl.Mpl: Matplotlib utilities """
        return self._mpl_impl(self)


class MapperPlugin(AnalyzerNode):
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
    def __getattr__(self, attrname):
        try:
            return self._setup[attrname]
        except KeyError:
            return self._kwargs.get(attrname, None)
    def __setattr__(self, attrname, val):
        try:
            AnalyzerNode.__setattr__(self, attrname, val)
        except AttributeError:
            if attrname in self._setup:
                self._setup[attrname] = val
            else:
                self._kwargs[attrname] = val
                if attrname.endswith('time_prior'):
                    self.time.enable_regularization()
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

Mapper.register(MapperPlugin)


__all__ = [ 'Mapper', 'MapperInitializer', 'MapperPlugin' ]

