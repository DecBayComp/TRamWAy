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
from .abc import *
from .proxy import TessellerProxy
from .plugin import TessellerPlugin
from . import proxied as tessellers
from .post import cell_mergers


class TessellerInitializer(Initializer):
    """
    Initializer class for the :class:`~tramway.analyzer.RWAnalyzer`
    :attr:`~tramway.analyzer.RWAnalyzer.tesseller` main attribute.

    The :attr:`~tramway.analyzer.RWAnalyzer.tesseller` attribute
    self-modifies on calling any of the *from_...* methods.

    The easiest way to define a tesseller is to set the
    :attr:`~tramway.analyzer.RWAnalyzer.tesseller` attribute with
    any of the tesseller classes provided by the :mod:`tessellers` module:

    .. code-block:: python

        from tramway.analyzer import *

        a = RWAnalyzer()
        a.tesseller = tessellers.KMeans

    Note that :mod:`tessellers` is made available by importing :mod:`tramway.analyzer`.

    """
    __slots__ = ()
    def from_plugin(self, plugin):
        self.specialize( TessellerPlugin, plugin )
    def from_callable(self, cls):
        """
        Argument:

            cls (callable):
                no-argument callable object (usually a class)
                that returns a :class:`Tesseller` object.

        """
        if isinstance(cls, type) and issubclass(cls, TessellerProxy):
            self.specialize( cls )
        else:
            self.specialize( TessellerProxy, cls )

    @property
    def _mpl_impl(self):
        from .mpl import Mpl
        return Mpl
    @property
    def mpl(self):
        """ tramway.analyzer.tesseller.mpl.Mpl: Matplotlib utilities """
        return self._mpl_impl(self)


__all__ = ['Tesseller', 'TessellerInitializer', 'TessellerProxy', 'TessellerPlugin', 'tessellers',
        'TessellationPostProcessing', 'cell_mergers']

