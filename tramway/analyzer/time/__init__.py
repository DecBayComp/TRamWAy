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
from ..artefact import analysis
from .abc import Time
import tramway.tessellation.window as window


class DT(object):
    """
    implements the default behavior of methods common to the initializer and the
    specialized attributes.

    It features the `dt` attribute and its alias `time_step`, and the `as_time_segments`
    slicer.

    The default implementation suits the initializer's behavior,
    i.e. a single all-in time segment.
    """
    __slots__ = ()
    def enable_regularization(self):
        """
        this method is called before running the inference plugins that define
        *time_prior* parameters.
        """
        raise AttributeError('no time segmentation defined; cannot regularize in time')
    def n_time_segments(self, sampling):
        """
        returns the number of time segments that `sampling` includes,
        under the hypothesis that `sampling` was generated following
        this segments definition (*self*).
        """
        return 1
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
        if not (index is None or index == 0):
            raise ValueError('no time segments defined')
        if return_times:
            try:
                t = sampling.points['t']
                all_times = np.r_[t.min(), t.max()]
            except (AttributeError, KeyError):
                all_times = None
        if maps is None:
            if return_times:
                if return_index:
                    yield 0, all_times, sampling
                else:
                    yield all_times, sampling
            else:
                if return_index:
                    yield 0, sampling
                else:
                    yield sampling
        else:
            if return_times:
                if return_index:
                    yield 0, all_times, sampling, maps
                else:
                    yield all_times, sampling, maps
            else:
                if return_index:
                    yield 0, sampling, maps
                else:
                    yield sampling, maps
    @property
    def spt_data(self):
        return self._parent.spt_data
    @property
    def dt(self):
        """
        imaging frame duration (or inter-frame time interval), in seconds (*float*).
        """
        return self.spt_data.dt
    @dt.setter
    def dt(self, dt):
        self.spt_data.dt = dt
    @property
    def time_step(self):
        """ alias for `dt` """
        return self.spt_data.time_step
    @time_step.setter
    def time_step(self, dt):
        self.spt_data.time_step = dt

class TimeInitializer(Initializer, DT):
    """
    initializer class for the `RWAnalyzer.time` main analyzer attribute.

    The `RWAnalyzer.time` attribute self-modifies on calling *from_...* methods.
    """
    __slots__ = ()
    def __init__(self, *args, **kwargs):
        Initializer.__init__(self, *args, **kwargs)
    def from_sliding_window(self, duration, shift=None):
        """
        Defines a sliding time window.

        Arguments:

            duration (float): window duration in seconds

            shift (float): time shift between successive segments, in seconds;
                by default, equals to `duration`

        See also :class:`SlidingWindow`.
        """
        self.specialize( SlidingWindow, duration, shift )

class SlidingWindow(AnalyzerNode, DT):
    """
    Specialization of the `RWAnalyzer.time` attribute.

    Defines the :meth:`segment` method that segments the SPT data into time segments
    and combines with the spatial tessellation.
    """
    __slots__ = ('_duration', '_shift', '_start_time', '_regularize_in_time')
    @property
    def reified(self):
        return True
    def __init__(self, duration=None, shift=None, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._duration = None if duration is None else float(duration)
        self._shift = None if shift is None else float(shift)
        self._start_time = None
        self._regularize_in_time = False
    @property
    def duration(self):
        """ window duration in seconds (*float*). """
        return self._duration
    @duration.setter
    def duration(self, d):
        self._duration = None if d is None else float(d)
    @property
    def shift(self):
        """ time shift between successive time segments, in seconds (*float*). """
        return self.duration if self._shift is None else self._shift
    @shift.setter
    def shift(self, s):
        self._shift = None if s is None else float(s)
    @property
    def window_duration(self):
        """ alias for `duration`. """
        return self.duration
    @window_duration.setter
    def window_duration(self, d):
        self.duration = d
    @property
    def window_shift(self):
        """ alias for `shift`. """
        return self.shift
    @window_shift.setter
    def window_shift(self, s):
        self.shift = s
    @property
    def start_time(self):
        """ start time for running the sliding window.
        By default, the window starts from the first data point in the input data
        (usually the ROI-cropped data, NOT the entire SPT dataset). """
        if self._start_time == 'sync':
            return self.spt_data.bounds.loc['min','t']
        return self._start_time
    @start_time.setter
    def start_time(self, t0):
        self._start_time = float(t0) if isinstance(t0, int) else t0
    def sync_start_times(self):
        """ aligns the start times of all the ROI to the same minimum time.

        Beware this may load all the SPT data files.
        Setting the *bounds* attribute of the `RWAnalyzer.spt_data` main attribute
        discards the need for screening the SPT data."""
        self._start_time = self.spt_data.bounds.loc['min','t']
    @property
    def time_window_kwargs(self):
        kwargs = dict(duration=self.duration, shift=self.shift, start_time=self.start_time)
        if self.regularize_in_time:
            kwargs['time_dimension'] = True
        return kwargs
    @analysis
    def segment(self, spt_dataframe, tessellation=None):
        """
        segments the SPT data, combines the segmentation with a spatial tessellation if any,
        and returns a `Partition` object.
        """
        tess = window.SlidingWindow(**self.time_window_kwargs)
        tess.spatial_mesh = tessellation
        try:
            spt_dataframe = spt_dataframe[self.tesseller.colnames]
        except AttributeError:
            pass
        tess.tessellate(spt_dataframe)
        return tess
    @property
    def tesseller(self):
        return self._parent.tesseller
    def enable_regularization(self):
        self._regularize_in_time = True
    @property
    def regularize_in_time(self):
        """ boolean property; ``True`` if time regularization is enabled. """
        return self._regularize_in_time
    def n_time_segments(self, sampling):
        return len(sampling.tessellation.time_lattice)
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
        if return_index:
            def _indexer(*args):
                for i, res in indexer(*args, return_index=True):
                    yield (i,)+res
        else:
            _indexer = indexer
        it = sampling.tessellation.split_segments(sampling, return_times=return_times)
        if maps is not None:
            maps = sampling.tessellation.split_segments(maps.maps)
            if return_times:
                ts, partitions = zip(*it)
                it = zip(ts, partitions, maps)
            else:
                partitions = it
                it = zip(partitions, maps)
        for seg in _indexer(index, it):
            yield seg
    @property
    def spt_data(self):
        return self._parent.spt_data

Time.register(SlidingWindow)


__all__ = ['TimeInitializer', 'Time', 'SlidingWindow']

