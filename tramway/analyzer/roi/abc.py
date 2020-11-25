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

class ROI(Attribute):
    """
    Abstract base class for the :attr:`~tramway.analyzer.RWAnalyzer.roi` attribute
    of an :class:`~tramway.analyzer.RWAnalyzer` object.
    """
    @abstractmethod
    def __iter__(self):
        return self.as_individual_roi()
    @abstractmethod
    def as_individual_roi(self, index=None, collection=None):
        """ 
        Generator function; iterates over the individual ROI.

        `index` is the index in the collection and `collection`
        should be defined if multiple collections of ROI are defined.

        .. note::

            Iterating over the individual ROI may be useful for visualization;
            favor :meth:`~tramway.analyzer.roi.ROI.as_support_regions` for
            data processing.

        """
        pass
    @abstractmethod
    def as_support_regions(self, index=None):
        """
        Generator function; iterates over the support regions.

        Support regions are the regions to be processed.
        An individual ROI operates as a window on a support region.
        """
        pass

