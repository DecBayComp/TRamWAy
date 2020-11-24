
from .collections import Collections
import numpy as np

"""
This module exports functions for the identification of regions of interest.
"""


def set_contiguous_time_support_by_count(points, space_bounds, time_window, min_points,
        min_segments=1, group_overlapping_roi=False, start_stop_min_points=None):
    """
    Arguments:

        points (DataFrame): location or translocation data

        space_bounds (dict or sequence): (collections of) boundaries,
            each boundary being a pair of NumPy arrays (lower, upper)

        time_window (float or pair of floats): window duration or
            window duration and shift

        min_points (int): desired minimum number of data rows per time segment

        min_segments (int): desired minimum number of selected time segments.

        group_overlapping_roi (bool): see :class:`~tramway.helper.roi.RoiCollections`

    Returns:

        dict or list: similar to `points`, with an additional column for time
            in each bound array.

    """
    noname = not isinstance(space_bounds, dict)
    if noname:
        space_bounds = {'': tuple(space_bounds)}

    roi = Collections(group_overlapping_roi)
    for label in space_bounds:
        roi[label] = space_bounds[label]

    try:
        duration, shift = time_window
    except (ValueError, TypeError):
        duration = shift = time_window
    min_duration = duration + (min_segments - 1) * shift

    if start_stop_min_points is None:
        start_stop_min_points = min_points

    n_space_dims = sum([ c in points.columns for c in 'xyz' ])

    regions = { label: [] for label in roi }
    for r in roi.regions:
        units = roi.regions.region_to_units(r)

        region_weight = sum([ len(u) for u in units.values() ]) # TODO: relative surface area instead
        threshold = start_stop_min_points * region_weight

        times = roi.regions.crop(r, points)['t'].values
        if min_points and len(times) < min_points * region_weight * min_segments:
            continue

        counts = np.zeros(times.size, dtype=np.int)
        for t in range(times.size):
            counts[t] = np.sum(times[t]-duration<=times[max(0,t-threshold):t+1])

        ok = np.flatnonzero(threshold<=counts)
        if ok.size==0:
            continue

        first_t, last_t = ok[0], ok[-1]

        if times[last_t] - times[first_t] + duration < min_duration:
            continue

        if min_points:
            if min_points != start_stop_min_points:
                threshold = min_points * region_weight

                ok = threshold<=counts[first_t:last_t+1]

                assert ok[0] and ok[-1]
                if not np.all(ok):
                    ok_first = first_t + 1 + np.flatnonzero(~ok[:-1] & ok[1:])
                    ok_last = first_t + np.flatnonzero(ok[:1] & ~ok[1:])
                    gap = times[ok_first] - times[ok_last]
                    assert np.all(0 < gap)
                    max_gap = window - 2 * shift # TODO: make it an input argument
                    gap_ok = gap <= max_gap
                    if not np.all(gap_ok):
                        first_t = np.r_[first_t,  ok_first[~gap_ok]]
                        last_t = np.r_[ok_last[~gap_ok], last_t]
                        segment_dur = times[ok_last] - times[ok_first] # + window
                        assert np.all(0 <= segment_dur)
                        segment_ok = min_duration <= segment_dur
                        if not np.all(segment_ok):
                            first_t = first_t[~segment_ok]
                            last_t = last_t[~segment_ok]

        if np.isscalar(first_t):
            first_ts, last_ts = [first_t], [last_t]
        else:
            first_ts, last_ts = first_t, last_t

        start_times, end_times = [], []
        for first_t, last_t in zip(first_ts, last_ts):

            start_time = times[first_t]-duration
            if start_time<times[0]:
                start_time = None

            if last_t+1==times.size:
                end_time = None
            else:
                end_time = times[last_t]

            nsegments = None
            if start_time is None:
                if end_time is None:
                    start_time = times[0]
                    end_time = times[-1]
                else:
                    nsegments = np.floor((end_time - times[first_t]) / shift) + 1
                    start_time = end_time - duration - (nsegments - 1) * shift
            elif end_time is None:
                end_time = times[-1]
            else:
                nsegments = np.floor((end_time - start_time - duration) / shift) + 1
                total_duration = duration + (nsegments - 1) * shift
                time_margin = .5 * (end_time - start_time - total_duration)
                start_time += time_margin
                end_time -= time_margin
            if nsegments is None:
                nsegments = np.floor((end_time - start_time - duration) / shift) + 1
            if nsegments<min_segments:
                continue

            start_times.append(start_time)
            end_times.append(end_time)

        for label in units:
            bounds = space_bounds[label]
            new_bounds = regions[label]
            for i in units[label]:
                for start_time, end_time in zip(start_times, end_times):
                    lower_bounds, upper_bounds = [ np.r_[r, t]
                        for r, t in zip(bounds[i], (start_time, end_time)) ]
                    new_bounds.append((lower_bounds, upper_bounds))

    if noname:
        return regions['']
    else:
        return regions

__all__ = [ 'set_contiguous_time_support_by_count' ]

