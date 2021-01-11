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
# for relative imports in sub-packages:
from ..attribute import AnalyzerNode, Initializer

class Tesseller(Attribute):
    """
    Abstract base class for the :attr:`~tramway.analyzer.RWAnalyzer.tesseller` attribute
    of an :class:`~tramway.analyzer.RWAnalyzer` object.
    """
    @abstractmethod
    def tessellate(self, spt_dataframe):
        pass
    @property
    @abstractmethod
    def resolution(self):
        r"""
        *float*: Desired spatial resolution in :math:`\mu m`
        """
        pass
    @resolution.setter
    @abstractmethod
    def resolution(self, res):
        pass

class TessellationPostProcessing(Attribute):
    """
    Abstract base class for the :attr:`~tramway.analyzer.RWAnalyzer.tesseller.post`
    attribute of a :class:`Tesseller` object.
    """
    @abstractmethod
    def post_process(self, tessellation, spt_dataframe):
        pass

