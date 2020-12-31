
from ..attribute import single
from ..artefact  import Analysis
from ..roi       import ROI
from collections import OrderedDict
import numpy  as np
import pandas as pd


def as_time_profiles(self, sampling, maps, space_bin_index=None, segment_indices=None,
        return_times=False, return_bin_index=False):
    if isinstance(sampling, ROI):
        roi = sampling
        sampling = roi.get_sampling()
    if isinstance(maps, (int, str)):
        if isinstance(sampling, Analysis):
            map_label = maps
            maps = sampling.get_child(map_label)
    times = {}
    profiles = OrderedDict()
    for _s, _t, _sampling, _maps in self.time.as_time_segments(sampling, maps,
            index=segment_indices, return_index=True, return_times=True):
        if space_bin_index is None:
            space_bin_index = _maps.index
        elif np.isscalar(space_bin_index):
            space_bin_index = [ space_bin_index ]
        for b in space_bin_index:
            if b in _maps.index:
                _values = _maps.loc[[b]].copy()
                _values.index = [ _s ]
                if b not in profiles:
                    times[b] = []
                    profiles[b] = []
                times[b].append(_t)
                profiles[b].append(_values)
    for b in profiles:
        profile = pd.concat(profiles[b])
        ret = []
        if return_bin_index:
            ret.append(b)
        if return_times:
            ts = np.stack(times[b], axis=0)
            ret.append(ts)
        ret.append(profile)
        yield tuple(ret)

def get_time_profile(self, sampling, maps, space_bin_index, segment_indices=None,
        return_times=False):
    return single(as_time_profiles(self, sampling, maps, space_bin_index, segment_indices, return_times, False))


__all__ = ['as_time_profiles', 'get_time_profile']

