
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

        `index` is a selector on the segment index, either as an *int* or a boolean callable that takes
        a segment index as input argument.
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

