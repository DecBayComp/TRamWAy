# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..abc import Initializer
from . import merger as cell_mergers

class TessellationPostProcessingInitializer(Initializer):
    __slots__ = ()
    def from_callable(self, cls):
        """
        Argument:

            cls (callable): no-argument callable object (usually a class)
                that returns a :class:`TessellationPostProcessing` object.

        """
        self.specialize( cls )

