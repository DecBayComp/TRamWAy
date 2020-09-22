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
    __slots__ = ()
    def enable_regularization(self):
        raise AttributeError('no time segmentation defined; cannot regularize in time')
    def n_time_segments(self, sampling):
        return 1
    def as_time_segments(self, sampling, maps=None, index=None, return_index=False, return_times=True):
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
        return self.spt_data.dt
    @dt.setter
    def dt(self, dt):
        self.spt_data.dt = dt
    @property
    def time_step(self):
        return self.spt_data.time_step
    @time_step.setter
    def time_step(self, dt):
        self.spt_data.time_step = dt

class TimeInitializer(Initializer, DT):
    __slots__ = ()
    def __init__(self, *args, **kwargs):
        Initializer.__init__(self, *args, **kwargs)
    def from_sliding_window(self, duration, shift=None):
        self.specialize( SlidingWindow, duration, shift )

class SlidingWindow(AnalyzerNode, DT):
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
        return self._duration
    @duration.setter
    def duration(self, d):
        self._duration = None if d is None else float(d)
    @property
    def shift(self):
        return self.duration if self._shift is None else self._shift
    @shift.setter
    def shift(self, s):
        self._shift = None if s is None else float(s)
    @property
    def start_time(self):
        if self._start_time == 'sync':
            return self.spt_data.bounds.loc['min','t']
        return self._start_time
    @start_time.setter
    def start_time(self, t0):
        self._start_time = float(t0) if isinstance(t0, int) else t0
    def sync_start_times(self):
        self._start_time = self.spt_data.bounds.loc['min','t']
    @property
    def time_window_kwargs(self):
        kwargs = dict(duration=self.duration, shift=self.shift, start_time=self.start_time)
        if self.regularize_in_time:
            kwargs['time_dimension'] = True
        return kwargs
    @analysis
    def segment(self, spt_dataframe, tessellation=None):
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
        return self._regularize_in_time
    def n_time_segments(self, sampling):
        return len(sampling.tessellation.time_lattice)
    def as_time_segments(self, sampling, maps=None, index=None, return_index=False, return_times=True):
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

