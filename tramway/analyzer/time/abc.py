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

class Time(Attribute):
    """
    Abstract base class for the :attr:`~tramway.analyzer.RWAnalyzer.time` attribute
    of an :class:`~tramway.analyzer.RWAnalyzer` object.
    """
    @abstractmethod
    def segment(self, spt_dataframe, tessellation=None):
        """
        Segments the SPT data, combines the segmentation with a spatial tessellation if any,
        and returns a :class:`~tramway.tessellation.time.TimeLattice` object.
        """
        pass
    @abstractmethod
    def as_time_segments(self, sampling, maps=None, index=None, return_index=False, return_times=True):
        """ Generator function; yields single-segment sampling and map objects.

        Slices `sampling` and `maps` and yields tuples with elements in the following order:

        * segment index (*int*), if `return_index` is :const:`True`,
        * segment start and stop times *(float, float)*, if `return_times` is :const:`True` (default),
        * segment :class:`~tramway.tessellation.base.Partition` object, from `sampling`,
        * segment maps (*pandas.DataFrame*) from `maps`, if `maps` is defined.

        `index` is a selector on the segment index, either as an *int*, a *set* of *int*,
        a *sequence* of *int*, or a *callable* that takes a segment index (*int*) as input argument
        and returns a *bool*.
        """
        pass
    @property
    @abstractmethod
    def dt(self):
        """
        *float*: See :attr:`~tramway.analyzer.spt_data.SPTParameters.dt`
        """
        pass
    @dt.setter
    @abstractmethod
    def dt(self, dt):
        pass

