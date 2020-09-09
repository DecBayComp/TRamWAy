
from ..attribute import *
from .abc import Tesseller
from .proxy import TessellerProxy
from .plugin import TessellerPlugin
from . import stdalg as tessellers


class TessellerInitializer(Initializer):
    __slots__ = ()
    def from_plugin(self, plugin):
        self.specialize( TessellerPlugin, plugin )
    def from_callable(self, cls):
        """
        Argument:

            cls (callable): 0-argument callable that returns an object
                with method :met:`tessellate`; usually a class.

        """
        if issubclass(cls, TessellerProxy):
            self.specialize( cls )
        else:
            self.specialize( TessellerProxy, cls )


__all__ = ['Tesseller', 'TessellerInitializer', 'TessellerProxy', 'TessellerPlugin', 'tessellers']

