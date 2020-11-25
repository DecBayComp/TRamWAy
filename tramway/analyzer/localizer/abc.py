# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute.abc import Attribute, abstractmethod

class Localizer(Attribute):
    """
    Abstract base class for the :attr:`~tramway.analyzer.RWAnalyzer.localizer` attribute
    of an :class:`~tramway.analyzer.RWAnalyzer` object.
    """
    def localize(self, stack):
        pass

