
from ..attribute import *
from .abc import Tesseller
from .proxy import TessellerProxy
from .plugin import TessellerPlugin
from . import stdalg as tessellers
from .post import cell_mergers


class TessellerInitializer(Initializer):
    __slots__ = ()
    def from_plugin(self, plugin):
        self.specialize( TessellerPlugin, plugin )
    def from_callable(self, cls):
        """
        Argument:

            cls (callable): no-argument callable object (usually a class)
                that returns a :class:`Tesseller` object.

        """
        if issubclass(cls, TessellerProxy):
            self.specialize( cls )
        else:
            self.specialize( TessellerProxy, cls )


__all__ = ['Tesseller', 'TessellerInitializer', 'TessellerProxy', 'TessellerPlugin', 'tessellers',
        'cell_mergers']

