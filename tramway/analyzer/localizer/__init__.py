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


class UNetLocalizer(AnalyzerNode):
    """ Loads the weights of a U-Net network for image deconvolution.
    """
    __slots__ = ('_weights_locator',)
    def __init__(self, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._weights_locator = 0
    @property
    def weights_locator(self):
        """ Uniform resource locator for the U-Net weights. """
        return self._weights_locator
    @weights_locator.setter
    def weights_locator(self, url):
        self._weights_locator = url
    def localize(self, stack):
        raise NotImplementedError

Localizer.register(UNetLocalizer)


class LocalizerInitializer(Initializer):
    """ initializer class for the `RWAnalyzer.localizer` main analyzer attribute.

    The `RWAnalyzer.localizer` attribute self-modifies on calling *from_...* methods.

    """
    __slots__ = ()
    def from_UNet(self):
        """ Loads a trained U-Net network for image deconvolution.
        
        See also :class:`UNetLocalizer`."""
        self.specialize( UNetLocalizer )
    def from_unet(self):
        """ Alias for `from_UNet`. """
        self.from_UNet()


__all__ = [ 'Localizer', 'LocalizerInitializer', 'UNetLocalizer' ]

