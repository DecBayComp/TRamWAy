
from .time import TimeLattice
from ..core import isstructured, hdf5
from collections import OrderedDict
import numpy as np
import pandas as pd


__all__ = ['setup', 'SlidingWindow']


setup = {
    'make_arguments': OrderedDict((
        ('duration', dict(type=float, help="window width in seconds (or in frames)")),
        ('shift', dict(type=float, help="time shift between consecutive segments, in seconds (or in frames)")),
        ('frames', dict(action='store_true', help="regard the --duration and --shift arguments as numbers of frames instead of timestamps")),
        )),
    'window_compatible': False,
    }


class SlidingWindow(TimeLattice):

    __slots__ = ('duration', 'shift', 'start_time')

    def __init__(self, scaler=None, duration=None, shift=None, frames=False, time_label=None,
        time_dimension=None, start_time=None):
        TimeLattice.__init__(self, scaler, time_label=time_label, time_dimension=time_dimension)
        if duration is None:
            raise ValueError("'duration' is required")
        elif np.isclose(max(0, duration), 0):
            raise ValueError("'duration' is too short")
        if shift is None:
            shift = duration
        elif np.isclose(max(0, shift), 0):
            raise ValueError("'shift' is too small")
        if frames:
            duration = int(duration)
            shift = int(shift)
        else:
            duration = float(duration)
            shift = float(shift)
        self.duration = duration
        self.shift = shift
        self.start_time = start_time

    def cell_index(self, points, *args, **kwargs):
        if self.time_lattice is None:
            time_col = kwargs.get('time_col', 't')
            if isstructured(points):
                ts = points[time_col]
                if isinstance(ts, (pd.Series, pd.DataFrame)):
                    ts = ts.values
            else:
                ts = points[:,time_col]
            t0, t1 = ts.min(), ts.max()
            if self.start_time is not None:
                t0 = self.start_time
            duration, shift = self.duration, self.shift
            if isinstance(duration, int):
                dt = np.median(np.diff(np.unique(ts)))
                duration *= dt
                shift *= dt
                dt /= 10.
            else:
                dt = 1e-7 # precision down to a microsecond (quantum < microsecond)
            nsegments = max(1., np.round((t1 - t0 - duration) / shift) + 1.)
            t1 = t0 + (nsegments - 1.) * shift + duration
            t0s = np.arange(t0, t1 - duration + dt, shift)
            t1s = t0s + duration
            self.time_lattice = np.stack((t0s, t1s), axis=-1)
        return TimeLattice.cell_index(self, points, *args, **kwargs)


import sys
if sys.version_info[0] < 3:

    import rwa
    sliding_window_exposes = hdf5.time_lattice_exposes + list(SlidingWindow.__slots__)#['duration', 'shift']
    rwa.hdf5_storable(rwa.default_storable(SlidingWindow, exposes=sliding_window_exposes),
        agnostic=True)

    __all__.append('sliding_window_exposes')

