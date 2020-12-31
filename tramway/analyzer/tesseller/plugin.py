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
from .proxy import *
from tramway.tessellation import plugins


class TessellerPlugin(TessellerProxy):
    """
    Wraps any plugin referenced in `tramway.tessellation.plugins`.
    """
    __slots__ = ('_module','_setup')
    def __init__(self, name, **kwargs):
        try:
            setup, module = plugins[name]
        except KeyError:
            raise KeyError("no such tesseller plugin: '{}'".format(name))
        cls = setup['make']
        if isinstance(cls, str):
            cls = getattr(module, cls)
        TessellerProxy.__init__(self, cls, **kwargs)
        self._reset_kwargs()
        self._module, self._setup = module, setup
    def _reset_kwargs(self):
        TessellerProxy._reset_kwargs(self)
        self._init_kwargs = { attr: None for attr in \
                ('ref_distance',
                 'min_distance','avg_distance','max_distance',
                 'min_probability','avg_probability','max_probability',
                ) }


def tesseller_plugin(name):
    """
    Translate plugin names into `RWAnalyzer`-ready tessellers as
    defined in the *tessellers* module, if available,
    else returns a generic `TessellerPlugin` initializer.
    """
    from . import proxied
    if name == 'grid':
        return proxied.Squares
    elif name == 'hexagon':
        return proxied.Hexagons
    elif name == 'kmeans':
        return proxied.KMeans
    elif name in ('gwr', 'gas'):
        return proxied.GWR
    else:
        class Cls(TessellerPlugin):
            __slots__ = () # important for __setattr__ to work properly!
            def __init__(self, **kwargs):
                TessellerPlugin.__init__(self, name, **kwargs)
        return Cls

__all__ = ['TessellerPlugin', 'tesseller_plugin']

