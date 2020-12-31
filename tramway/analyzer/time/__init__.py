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
from ..artefact import analysis, Analysis
from .abc import Time
import tramway.tessellation.window as window
import numpy as np
import pandas as pd


class DT(object):
    """
    Implements the default behavior of methods common to the initializer and the
    specialized attributes.

    It gives access to the frame interval (the :attr:`dt` and :attr:`time_step`
    attributes are aliases of :attr:`frame_interval`),
    and features the :meth:`as_time_segments` slicer.

    The default implementation suits the initializer's behavior,
    i.e. a single all-in time segment.
    """
    __slots__ = ()
    def enable_regularization(self):
        """
        This method is called before running the inference plugins that define
        *time_prior* parameters.
        """
        raise AttributeError('no time segmentation defined; cannot regularize in time')
    def n_time_segments(self, sampling):
        """
        Returns the number of time segments (*int*) that `sampling` includes,
        under the hypothesis that `sampling` was generated following
        this segments definition (*self*).
        """
        return 1
    def as_time_segments(self, sampling, maps=None, index=None, return_index=False, return_times=True):
        if not (index is None or index == 0):
            raise ValueError('no time segments defined')
        if isinstance(sampling, Analysis):
            sampling = sampling.data
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
    as_time_segments.__doc__ = Time.as_time_segments.__doc__
    @property
    def spt_data(self):
        return self._parent.spt_data
    @property
    def dt(self):
        """
        *float*: See :attr:`~tramway.analyzer.spt_data.SPTParameters.dt`
        """
        return self.spt_data.dt
    @dt.setter
    def dt(self, dt):
        self.spt_data.dt = dt
    @property
    def time_step(self):
        """
        *float*: See :attr:`~tramway.analyzer.spt_data.SPTParameters.time_step`
        """
        return self.spt_data.time_step
    @time_step.setter
    def time_step(self, dt):
        self.spt_data.time_step = dt
    @property
    def frame_interval(self):
        """
        *float*: See :attr:`~tramway.analyzer.spt_data.SPTParameters.frame_interval`
        """
        return self.spt_data.frame_interval
    @frame_interval.setter
    def frame_interval(self, dt):
        self.spt_data.frame_interval = dt

class TimeInitializer(Initializer, DT):
    """
    Initializer :class:`Time` class for the :class:`~tramway.analyzer.RWAnalyzer`
    :attr:`~tramway.analyzer.RWAnalyzer.time` main attribute.

    The :attr:`~tramway.analyzer.RWAnalyzer.time` attribute self-modifies on
    calling *from_...* methods.
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
    def from_sampling(self, sampling):
        """
        Extracts the time segmentation parameters stored in a
        :class:`~tramway.tessellation.base.Partition` object
        and tries to initialize the parent
        :attr:`~tramway.analyzer.RWAnalyzer.time` attribute
        correspondingly.

        This may fail,
        either silently or raising a :class:`ValueError` exception,
        as this method covers only cases of sliding windows.
        """
        if isinstance(sampling, Analysis):
            sampling = sampling.data
        segmentation = sampling.tessellation
        if isinstance(segmentation, window.SlidingWindow):
            duration, shift, start_time = segmentation.duration, segmentation.shift, segmentation.start_time
        else:
            # experimental!
            import tramway.tessellation.time as time
            if isinstance(segmentation, time.TimeLattice):
                segments = segmentation.time_lattice
                if segments is None or len(segments)==0:
                    raise ValueError('no time segments defined')
                durations = np.diff(segments, axis=1)
                duration = durations[0]
                if not np.all(durations == duration):
                    raise ValueError('varying time segment duration is not supported')
                shifts = np.diff(segments[:,0])
                if shifts.size == 0:
                    shift = None
                else:
                    shift = shifts[0]
                    if not np.all(shifts == shift):
                        raise ValueError('varying time shift is not supported')
                start_time = segments[0,0]
                t0 = sampling.points['t'].min()
                if start_time == t0:
                    start_time = None
            else:
                self._parent.logger.warning('unsupported time segmentation type: '+str(type(segmentation)))
                return
        self.specialize( SlidingWindow, duration, shift )
        if start_time is not None:
            self._parent.time.start_time = start_time


class SlidingWindow(AnalyzerNode, DT):
    """
    Specialization for the :attr:`~tramway.analyzer.RWAnalyzer.time` attribute.

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
        """ *float*: Window duration in seconds """
        return self._duration
    @duration.setter
    def duration(self, d):
        self._duration = None if d is None else float(d)
    @property
    def shift(self):
        """ *float*: Time shift between successive time segments, in seconds """
        return self.duration if self._shift is None else self._shift
    @shift.setter
    def shift(self, s):
        self._shift = None if s is None else float(s)
    @property
    def window_duration(self):
        """ *float*: Alias for :attr:`duration` """
        return self.duration
    @window_duration.setter
    def window_duration(self, d):
        self.duration = d
    @property
    def window_shift(self):
        """ *float*: Alias for :attr:`shift` """
        return self.shift
    @window_shift.setter
    def window_shift(self, s):
        self.shift = s
    @property
    def start_time(self):
        """
        *float*: Start time for running the sliding window.
            By default, the window starts from the first data point in the input data
            (usually the ROI-cropped data, NOT the entire SPT dataset).
        """
        if self._start_time == 'sync':
            return self.spt_data.bounds.loc['min','t']
        return self._start_time
    @start_time.setter
    def start_time(self, t0):
        self._start_time = float(t0) if isinstance(t0, int) else t0
    def sync_start_times(self):
        """ aligns the start times of all the ROI to the same minimum time.

        Beware this may load all the SPT data files.
        Setting the :attr:`~tramway.analyzer.spt_data.SPTData.bounds` attribute
        of the :attr:`~tramway.analyzer.RWAnalyzer.spt_data` main attribute
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
        tess = window.SlidingWindow(**self.time_window_kwargs)
        tess.spatial_mesh = tessellation
        try:
            spt_dataframe = spt_dataframe[self.tesseller.colnames]
        except AttributeError:
            pass
        tess.tessellate(spt_dataframe, time_only=True)
        return tess
    segment.__doc__ = Time.segment.__doc__
    @property
    def tesseller(self):
        return self._parent.tesseller
    def enable_regularization(self):
        self._regularize_in_time = True
    @property
    def regularize_in_time(self):
        """ *bool*: :const:`True` if time regularization is enabled """
        return self._regularize_in_time
    def n_time_segments(self, sampling):
        return len(sampling.tessellation.time_lattice)
    def as_time_segments(self, sampling, maps=None, index=None, return_index=False, return_times=True):
        if return_index:
            if return_times or maps is not None:
                def _indexer(*args):
                    for i, res in indexer(*args, return_index=True):
                        yield (i,)+res
            else:
                def _indexer(*args):
                    yield from indexer(*args, return_index=True)
        else:
            _indexer = indexer
        if isinstance(sampling, Analysis):
            sampling = sampling.data
        it = sampling.tessellation.split_segments(sampling, return_times=return_times)
        if maps is not None:
            if isinstance(maps, Analysis):
                maps = maps.data
            try:
                maps = maps.maps
            except AttributeError:
                pass
            maps = sampling.tessellation.split_segments(maps)
            if return_times:
                ts, partitions = zip(*it)
                it = zip(ts, partitions, maps)
            else:
                partitions = it
                it = zip(partitions, maps)
        yield from _indexer(index, it)
    @property
    def spt_data(self):
        return self._parent.spt_data
    def self_update(self, op):
        self._parent._time = op(self)
    def segment_label(self, map_label, times, sampling):
        """
        Makes a label combining the input label as prefix and time-related suffix.
        """
        if isinstance(sampling, Analysis):
            sampling = sampling.data
        if isinstance(times, int):
            segment_index = times
            times = sampling.tessellation.time_lattice[segment_index]
        format_time = lambda t: '{:.3f}'.format(t).rstrip('0').rstrip('.')
        if map_label:
            return '{} -- t={}-{}s'.format(map_label, *[ format_time(t) for t in times ])
        else:
            return 't={}-{}s'.format(*[ format_time(t) for t in times ])
    def combine_segments(self, combined_output_label, combined_sampling, commit=True, permissive=False):
        analyses = combined_sampling.subtree
        sampling = combined_sampling.data
        try:
            single_segment_output, labels = [], []
            for times in sampling.tessellation.time_lattice:
                single_segment_label = self.segment_label(combined_output_label, times, sampling)
                labels.append(single_segment_label)
                single_segment_output.append(analyses[single_segment_label].data)
        except KeyError:
            if single_segment_output:
                if permissive:
                    self._eldest_parent.logger.warning('not all segments are available; combining aborted')
                else:
                    raise KeyError('not all segments are available; combining aborted') from None
            else:
                if permissive:
                    return None
                else:
                    labels = list(analyses.labels)
                    if labels and any( l.startswith(combined_output_label+' -- t=') for l in labels ):
                        raise KeyError("'{}' segments do not start at the expected time; expected label: '{}'; found labels: {}".format(combined_output_label, single_segment_label, str(labels))) from None
                    else:
                        raise KeyError("no '{}' segments found".format(combined_output_label)) from None
        assert single_segment_output
        from tramway.inference.base import Maps
        ok = [ isinstance(m, Maps) for m in single_segment_output ]
        if all(ok):
            pass
        elif any(ok):
            raise TypeError('cannot combine heterogeneous types')
        else:
            raise TypeError('cannot combine values of type: {}'.format(type(single_segment_output[0])))
        ncells = sampling.tessellation.spatial_mesh.number_of_cells
        import copy
        it = iter(single_segment_output)
        maps = copy.deepcopy(next(it))
        maps.runtime = None
        maps.posteriors = None
        df, s = [maps.maps], 0
        while True:
            try:
                m = next(it)
            except StopIteration:
                break
            s += 1
            m = m.maps.copy()
            m.index += s*ncells
            df.append(m)
        maps.maps = pd.concat(df)
        if commit:
            for label in labels:
                del analyses[label]
            analyses[combined_output_label] = maps
        return maps

Time.register(SlidingWindow)


__all__ = ['Time', 'DT', 'TimeInitializer', 'SlidingWindow']

