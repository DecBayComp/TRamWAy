
from .abc import *
from .proxy import *
from tramway.tessellation import plugins


class TessellerPlugin(TessellerProxy):
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
        self._module, self._setup = module, setup
        self._init_kwargs = { attr: None for attr in \
                ('min_distance','avg_distance','max_distance',
                 'min_probability','avg_probability','max_probability',
                ) }


def tesseller_plugin(name):
    pass
    #if name in ('grid', 'hexagon', 'kmeans', 'gwr'):

__all__ = ['TessellerPlugin', 'tesseller_plugin']

