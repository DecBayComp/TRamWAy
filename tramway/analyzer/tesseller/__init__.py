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

