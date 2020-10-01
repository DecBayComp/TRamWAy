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
    @abstractmethod
    def segment(self, spt_dataframe, tessellation=None):
        pass
    @abstractmethod
    def as_time_segments(self, sampling, maps=None, index=None, return_index=False, return_times=True):
        """
        slices `sampling` and `maps` and returns an iterator of multiple elements in the following order:

        * segment index (*int*), if `return_index` is ``True``,
        * segment start and stop times *(float, float)*, if `return_times` is ``True`` (default),
        * segment :class:`~tramway.tessellation.base.Partition` object, from `sampling`,
        * segment maps (:class:`pandas.DataFrame`) from `maps`, if `maps` is defined.

        `index` is a selector on the segment index, either as an *int* or a *sequence* of *int*s,
        or a boolean *callable* that takes a segment index (*int*) as input argument.
        """
        pass
    @property
    @abstractmethod
    def dt(self):
        pass
    @dt.setter
    @abstractmethod
    def dt(self, dt):
        pass

