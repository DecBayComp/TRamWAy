
from .collections import Collections
import scipy.sparse as sparse
import numpy as np

"""
This module exports functions for the identification of regions of interest.
"""


def epanechnikov_density(xy, eval_at, target_pattern_size):
    from sklearn.neighbors import KernelDensity

    bandwidth = .5 * target_pattern_size # the bandwidth is similar to a radius (not a diameter)
    estimator = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)

    estimator.fit(xy)
    log_density = estimator.score_samples(eval_at)

    return log_density


def density_based_roi(locations, min_kernel_density=.1,
        target_pattern_size=.3, step_size_factor=.1, dr=4,
        kernel_density=epanechnikov_density):
    xy = locations[['x','y']].values
    
    extent = np.ravel(np.r_[xy.min(axis=0, keepdims=True), xy.max(axis=0, keepdims=True)])

    step = step_size_factor * target_pattern_size

    if extent is None:
        xymin, xymax = xy.min(axis=0), xy.max(axis=0)
    else:
        xymin, xymax = [extent[0], extent[1]], [extent[2], extent[3]]

    grid_x, grid_y = np.arange(xymin[0], xymax[0], step), np.arange(xymin[1], xymax[1], step)
    _x, _y = np.meshgrid(grid_x, grid_y, indexing='ij')
    grid = np.c_[_x.reshape((-1,1)),_y.reshape((-1,1))]

    log_density = epanechnikov_density(xy, grid, target_pattern_size)
    log_density = log_density.reshape(len(grid_x),len(grid_y))
    
     # find local maxima using logical convolution
    sizx, sizy = log_density.shape[0]-2*dr, log_density.shape[1]-2*dr
    _x = lambda x: (log_density.shape[0] if x==0 else x)
    _y = lambda y: (log_density.shape[1] if y==0 else y)
    mask = np.ones((sizx, sizy), dtype=bool)
    for dx in range(-dr,dr):
        for dy in range(-dr,dr):
            if (not (dx==0 and dy==0)) and (dx+dy <= dr):
                mask = mask & \
                        (log_density[dr+dx:_x(dx-dr),dr+dy:_y(dy-dr)] < log_density[dr:-dr,dr:-dr])

    local_max = sparse.coo_matrix(mask)
    selected_nz = np.log(min_kernel_density) <= log_density[local_max.row,local_max.col]
    if not np.any(selected_nz):
        print('global max density is below the threshold: {} < {}'.format(
            np.exp(log_density[local_max.row,local_max.col].max()), min_kernel_density))
    selected_i, selected_j = local_max.row[selected_nz], local_max.col[selected_nz]

    # define the roi as center points
    roi_centers = [ np.r_[grid_x[dr+i],grid_y[dr+j]] for i,j in zip(selected_i, selected_j) ]
    if roi_centers:
        if len(roi_centers)<10:
            print('few roi found: {}'.format(len(roi_centers)))
    else:
        raise RuntimeError('no roi found')
        
    return roi_centers


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
        if not group_overlapping_roi:
            assert region_weight == 1
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

            min_start_time = times[first_t] - duration
            max_start_time = max(times[0], min_start_time)

            min_end_time = times[last_t]
            if last_t+1 == times.size:
                max_end_time = times[-max(1, threshold)] + duration
            else:
                max_end_time = min_end_time

            max_total_duration = max_end_time - min_start_time
            nsegments = np.floor((max_total_duration - duration) / shift) + 1
            total_duration = duration + (nsegments - 1) * shift
            time_margin = .5 * (max_total_duration - total_duration)
            start_time = min(min_start_time + time_margin, max_start_time)
            end_time = max_end_time# - time_margin # no need to discard the trailing data

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


def _enumerate(it):
    for ix, elem in enumerate(it):
        yield (ix,)+tuple(elem)

def group_roi(a, *args, overlap=.75, return_matches_only=False):
    """ Returns the smallest rectangles that contain the ROI grouped depending on whether
    they overlap or not.

    A ROI should be a pair (tuple) of `numpy.ndarray` for (lower-,upper-) bounds.
    
    Time is handled just like any space coordinate.
    As a consequence, if time bounds are also specified, the corresponding values
    should be scaled so that time can be artificially related to space in the volume
    calculation carried out in the estimatation of the amount of overlap.
    """
    if args:
        b = args[0]
        if args[1:]:
            raise ValueError('arguments `overlap` and `return_matches_only` should be keyworded')
    else:
        b = a
    intra_grouping = a is b
    if intra_grouping:
        already_grouped = set()
    grouped_roi = []
    for _a, _a_lb, _a_ub in _enumerate(a):
        if intra_grouping and _a in already_grouped:
            continue
        #
        _a_area = np.prod(_a_ub - _a_lb)
        #
        _matches = []
        for _b, _b_lb, _b_ub in _enumerate(b):
            if intra_grouping and _b == _a:
                continue
            #
            _lb, _ub = np.maximum(_a_lb, _b_lb), np.minimum(_a_ub, _b_ub)
            if np.all(_lb < _ub):
                _b_area = np.prod(_b_ub - _b_lb)
                _max_area = min(_a_area, _b_area)
                _intersect_area = np.prod(_ub - _lb)
                _overlap = _intersect_area / _max_area
                if overlap <= _overlap:
                    _matches.append((_b_lb, _b_ub))
                    if intra_grouping:
                        already_grouped.add(_b)
        _lb, _ub = _a_lb, _a_ub
        if _matches:
            # compute the minimum rectangle that contains the union
            while _matches:
                _other_lb, _other_ub = _matches.pop()
                _lb, _ub = np.minimum(_lb, _other_lb), np.maximum(_ub, _other_ub)
        elif return_matches_only:
            continue
        grouped_roi.append((_lb, _ub))
    return grouped_roi


__all__ = [ 'set_contiguous_time_support_by_count', 'epanechnikov_density', 'density_based_roi',
        'group_roi' ]

