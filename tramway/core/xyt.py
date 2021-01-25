# -*- coding: utf-8 -*-

# Copyright © 2017-2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import os
import numpy as np
import pandas as pd
import warnings
import itertools
from .exceptions import *
import re


def _translocations(df, sort=True): # very slow; may be marked as deprecated
    def diff(df):
        df = df.sort_values('t')
        t = df['t'][1:]
        df = df.diff()
        df['t'] = t
        return df
    return df.groupby(['n'], sort=False).apply(diff)

def translocations(df, sort=False):
    '''Each trajectories should be represented by consecutive rows sorted by time.

    Returns displacements without the associated position information.'''
    if sort:
        return _translocations(df) # not exactly equivalent
        #raise NotImplementedError
    i = 'n'
    xyz = ['x', 'y']
    if 'z' in df.columns:
        xyz.append('z')
    dxyz = [ 'd'+col for col in xyz ]
    if all( col in df.columns for col in dxyz ):
        jump = df[dxyz]
    else:
        ixyz = xyz + [i]
        jump = df[ixyz].diff()
        #df[xyz] = jump[xyz]
        #df = df[jump[i] != 0]
        #return df
        jump = jump[jump[i] == 0][xyz]
    return jump#np.sqrt(np.sum(jump * jump, axis=1))


def iter_trajectories(trajectories, trajnum_colname='n', asslice=False, asarray=False, order=None):
    """
    Yields the different trajectories in turn.

    If ``asslice=True``, the indices corresponding to the trajectory are returned instead,
    or as first output argument if ``asarray=True``, as a slice (first index, last index + 1).

    `order` can be *'start'* to ensure that trajectories are yielded by ascending start time.
    """
    if not isinstance(trajectories, pd.DataFrame):
        raise TypeError('trajectories is not a DataFrame')
    if trajectories.empty:
        return ()

    if asarray or not asslice:
        other_cols = [ col for col in trajectories.columns if col != trajnum_colname ]
    if asarray:
        dat = trajectories[other_cols].values
    if asslice:
        if asarray:
            from_slice = lambda a,b: (a,b), dat[a:b]
        else:
            from_slice = lambda a,b: (a,b)
    else:
        if asarray:
            from_slice = lambda a,b: dat[a:b]
        else:
            from_slice = lambda a,b: trajectories.iloc[a:b]

    if order is None:

        curr_traj_num = trajectories[trajnum_colname].iat[0] - 1
        for i, num in enumerate(trajectories[trajnum_colname]):
            if num == curr_traj_num:
                stop += 1
            elif num < curr_traj_num:
                raise IndexError('trajectories are not ascendingly sorted')
            else:
                if 0<i:
                    yield from_slice(start, stop)
                start = i
                stop = start + 1
                curr_traj_num = num
        yield from_slice(start, stop)

    elif order == 'start':

        new_n = 0<np.diff(trajectories['n'].values)
        new_n = np.flatnonzero(new_n)
        if new_n.size:
            new_n += 1
            traj_ids = np.stack((np.r_[0,new_n], np.r_[new_n,len(trajectories)]), axis=1)
            traj_ts = trajectories['t'].iloc[traj_ids[:,0]].values
            traj_ids = traj_ids[np.argsort(traj_ts)]
        else:
            traj_ids = np.array([[0,len(trajectories)]])
        for i,j in traj_ids:
            yield from_slice(i,j)


def iter_frames(points, asslice=False, as_trajectory_slices=False, dt=None, skip_empty_frames=True):
    """
    Yields series of row indices, each series corresponding to a different frame.
    """
    if not isinstance(points, pd.DataFrame):
        raise TypeError('input data is not a DataFrame')
    ts = points['t'].values
    t = ts.min()
    if not np.isclose(t, ts[0]):
        raise ValueError('first location or trajectory starts later than t0')
    if 'n' in points:
        # trajectories or translocations; locations are grouped by trajectory
        if asslice:
            raise ValueError('rows are not contiguous in time; cannot represent frames as row slices')
        it = iter_trajectories(points, asslice=True, order='start')
        exhausted = False
        active_trajs = []
        while True:
            try:
                traj_ids = next(it)
            except StopIteration:
                exhausted = True
                next_traj = None
                break
            if np.isclose(ts[traj_ids[0]], t):
                active_trajs.append(traj_ids)
            else:
                next_traj = traj_ids
                break
        if dt is None:
            dt = np.median(np.diff(ts[traj_ids[0]:traj_ids[1]]))
        elif dt <= 0:
            raise ValueError('dt is not strictly positive')
        while True:
            frame = []
            trajs = []
            still_active = []
            for traj in active_trajs:
                traj_ts = ts[traj[0]:traj[1]]
                row_ix = int(np.round((t-traj_ts[0])/dt))
                if not np.isclose(traj_ts[row_ix], t):
                    traj_dt = np.diff(traj_ts)
                    ok = np.isclose(traj_dt , dt)
                    if np.any(ok):
                        raise ValueError('a trajectory is not contiguous in time:\n{}'.format(points[traj[0]:traj[1]]))
                    else:
                        raise ValueError('dt may not be properly set: {}'.format(dt))
                frame.append(traj[0]+row_ix)
                trajs.append(traj)
                if t+dt/2<traj_ts[-1]:
                    still_active.append(traj)
            if frame:
                if as_trajectory_slices:
                    yield trajs
                else:
                    yield np.array(frame)
            else:
                raise RuntimeError('no trajectories found in frame')
            #
            active_trajs = still_active
            if active_trajs:
                t += dt
            elif exhausted:
                break
            else:
                next_t = ts[next_traj[0]]
                if not skip_empty_frames:
                    t += dt
                    while t<next_t-dt/2:
                        yield []
                        t += dt
                t = next_t
            #
            if not exhausted and np.isclose(ts[next_traj[0]], t):
                active_trajs.append(next_traj)
                while True:
                    try:
                        traj_ids = next(it)
                    except StopIteration:
                        exhausted = True
                        next_traj = None
                        break
                    if np.isclose(ts[traj_ids[0]], t):
                        active_trajs.append(traj_ids)
                    else:
                        next_traj = traj_ids
                        break
    else:
        if as_trajectory_slices:
            raise ValueError('input data are not trajectories')
        if not skip_empty_frames and (dt is None or dt == 0):
            raise ValueError('dt is undefined')
        if asslice:
            from_slice = lambda a,b: (a,b)
            empty_slice = ()
        else:
            from_slice = lambda a,b: np.arange(a,b)
            empty_slice = np.arange(0)

        i, t_i, j = 0, t, 1
        for t_j in ts:
            if np.issclose(t_j, t_i):
                j += 1
            else:
                yield from_slice(i, j)
                if not skip_empty_frames:
                    t_i += dt
                    while not np.isclose(t_j, t_i):
                        yield empty_slice
                        t_i += dt
                i, t_i = j, t_j
        if i<j:
            yield from_slice(i, j)


def load_xyt(path, columns=None, concat=True, return_paths=False, verbose=False,
        reset_origin=False, header=None, **kwargs):
    """
    Load trajectory files.

    Files are loaded with :func:`~pandas.read_csv` and should have the same number of columns
    and either none or all files should exhibit a single-line header.

    Default column names are 'n', 'x', 'y' and 't'.

    Arguments:

        path (str or list of str): path to trajectory file or directory.

        columns (list of str): column names.

        concat (bool): if multiple files are read, return a single DataFrame.

        return_paths (bool): paths to files are returned as second output argument.

        verbose (bool): print extra messages.

        reset_origin (bool or sequence): the lowest coordinate is translated to 0.
            Apply to time and space columns. Default column names are 'x', 'y', 'z'
            and 't'. A sequence overrides the default.

        header (bool): if defined, a single-line header is expected in the file(s);
            if ``False``, ignore the header;
            if ``True``, overwrite the `columns` argument with names from the header;
            if undefined, check whether a header is present and, if so, act as ``True``.

    Returns:

        pandas.DataFrame or list or tuple: trajectories as one or multiple DataFrames;
            if `tuple` (with *return_paths*), the trajectories are first, the list
            of filepaths second.

    Extra keyword arguments are passed to :func:`~pandas.read_csv`.
    """
    if columns is not None and header is True:
        raise ValueError('both column names and header are defined')
    #if 'n' not in columns:
    #    raise ValueError("trajectory index should be denoted 'n'")
    if 'sep' not in kwargs and 'delimiter' not in kwargs:
        kwargs['delim_whitespace'] = True
    if not isinstance(path, list):
        path = [path]
    paths = []
    for p in path:
        if os.path.isdir(p):
            paths.append([ os.path.join(p, f) for f in os.listdir(p) ])
        else:
            paths.append([p])
    index_max = 0
    df = []
    paths = list(itertools.chain(*paths))
    if not paths:
        if verbose:
            print('nothing to load')
        return
    _failed = []
    for f in paths:
        try:
            if verbose:
                print('loading file: {}'.format(f))
            if header is False:
                if columns is None:
                    columns = ['n', 'x', 'y', 't']
                kwargs['names'] = columns
                dff = pd.read_csv(f, header=0, **kwargs)
            else:
                with open(f, 'r') as fd:
                    first_line = fd.readline()
                if re.search(r'[a-df-zA-DF-Z_]', first_line):
                    if columns is None:
                        columns = first_line.split()
                    kwargs['names'] = columns
                    dff = pd.read_csv(f, header=0, **kwargs)
                elif header is True:
                    dff = pd.read_csv(f, header=0, **kwargs)
                    columns = dff.columns
                else:
                    if columns is None:
                        columns = ['n', 'x', 'y', 't']
                    kwargs['names'] = columns
                    dff = pd.read_csv(f, **kwargs)
        except OSError:
            _failed.append(f)
        else:
            if 'n' in columns:
                sample = dff[dff['n']==dff['n'].iloc[-1]]
                sample_dt = sample['t'].diff()[1:]
                if not all(0 < sample_dt):
                    if any(0 == sample_dt):
                        try:
                            conflicting = sample_dt.values == 0
                            conflicting = np.logical_or(np.r_[False, conflicting], np.r_[conflicting, False])
                            print(sample.loc[conflicting])
                        except:
                            pass
                        raise ValueError("some simultaneous locations are associated to a same trajectory: '{}'".format(f))
                    else:
                        warnings.warn(EfficiencyWarning("table '{}' is not properly ordered".format(f)))
                    # faster sort
                    data = np.asarray(dff)
                    dff = pd.DataFrame(data=data[np.lexsort((dff['t'], dff['n']))],
                        columns=dff.columns)
                    #sorted_dff = []
                    #for n in dff['n'].unique():
                    #       sorted_dff.append(dff[dff['n'] == n].sort_values(by='t'))
                    #dff = pd.concat(sorted_dff)
                    #dff.index = np.arange(dff.shape[0]) # optional
                if dff['n'].min() < index_max:
                    dff['n'] += index_max
                    index_max = dff['n'].max()
            undefined = dff.isnull().values.all(axis=0)
            if np.any(undefined):
                if columns == list('nxyt') and np.sum(undefined) == 1:
                    raise ValueError('the molecules are not tracked')
                else:
                    raise ValueError('too many specified columns: {}'.format(columns))
            df.append(dff)
    if df:
        for f in _failed:
            warnings.warn(f, FileNotFoundWarning)
    else:
        if paths[1:]:
            raise OSError('cannot load any of the files: {}'.format(paths))
        else:
            raise OSError('cannot load file: {}'.format(paths[0]))
    if reset_origin:
        if reset_origin == True:
            reset_origin = [ col for col in ['x', 'y', 'z', 't'] if col in columns ]
        origin = dff[reset_origin].min().values
        for dff in df[:-1]:
            origin = np.minimum(origin, dff[reset_origin].min().values)
        for dff in df:
            dff[reset_origin] -= origin
    if concat:
        if df[1:]:
            df = pd.concat(df)
        else:
            df = df[0]
    if return_paths:
        return (df, paths)
    else:
        return df


def crop(points, box, by=None, add_deltas=True, keep_nans=False, no_deltas=False, keep_nan=None,
        preserve_index=False):
    """
    Remove locations outside a bounding box.

    When a location is discarded, the corresponding trajectory is splitted into two distinct
    trajectories.

    Important: the locations of any given trajectory should be contiguous and ordered.

    Arguments:

        locations (pandas.DataFrame): locations with trajectory indices in column 'n',
            times in column 't' and coordinates in the other columns;
            delta columns are ignored

        box (array-like): origin and size of the space bounding box

        by (str): for translocations only;
            '*start*' or '*origin*': crop by translocation origin; keep the associated destinations;
            '*stop*' or '*destination*': crop by translocation destinations; keep the associated
            origins;
            trajectories with a single non-terminal point outside the bounding box are
            not splitted

        add_deltas (bool): add 'dx', 'dy', ..., 'dt' columns is they are not already present;
            deltas are associated to the translocation origins

        keep_nans/keep_nan (bool): adding deltas generates NaN; keep them

        no_deltas (bool): do not consider any column as deltas

        preserve_index (bool): do not split the trajectories with out-of-bound locations,
            do not re-index the trajectories

    Returns:

        pandas.DataFrame: filtered (trans-)locations
    """
    box = np.asarray(box)
    dim = int(box.size / 2)
    support_lower_bound = box[:dim]
    support_size = box[dim:]
    support_upper_bound = support_lower_bound + support_size
    not_coord_cols = ['n', 't']
    if no_deltas:
        delta_cols = []
    else:
        delta_cols = [ c for c in points.columns \
                if c[0]=='d' and c[1:] != 'n' and c[1:] in points.columns ]
        not_coord_cols += delta_cols
    coord_cols = [ c for c in points.columns if c not in not_coord_cols ]
    if len(coord_cols) != dim:
        raise ValueError('the bounding box has dimension {} while the following coordinate columns were found: {}'.format(dim, coord_cols))
    within = np.all(np.logical_and(support_lower_bound <= points[coord_cols].values,
        points[coord_cols].values <= support_upper_bound), axis=1)
    if add_deltas or by:
        paired_dest = points['n'].diff().values==0
        paired_src = np.r_[paired_dest[1:], False]
    points = points.copy()
    if add_deltas:
        cols_with_deltas = [ c[1:] for c in delta_cols ]
        cols_to_diff = [ c for c in points.columns if c not in ['n']+cols_with_deltas+delta_cols ]
        deltas = points[cols_to_diff].diff().shift(-1)
        deltas = deltas[paired_src]
        deltas.columns = [ 'd'+c for c in cols_to_diff ]
        points = points.join(deltas)
    if by:
        if by in ('start', 'origin'):
            within[paired_dest] |= within[paired_src]
        elif by in ('stop', 'destination'):
            within[paired_src] |= within[paired_dest]
        else:
            raise ValueError('unsupported value for argument `by`')
    if 'n' in points.columns:
        n = points['n'] + np.cumsum(np.logical_not(within), dtype=points.index.dtype)
        single_point = 0 < n.diff().values[1:]
        single_point[:-1] = np.logical_and(single_point[:-1], single_point[1:])
        ok = np.r_[True, np.logical_not(single_point)]
        points = points[ok]
        within = within[ok]
        if not preserve_index:
            n -= (n.diff().fillna(0).astype(int) - 1).clip(lower=0).cumsum()
            points['n'] = n
    points = points[within]
    if keep_nan is None:
        keep_nan = keep_nans
    if not keep_nan:
        points.dropna(inplace=True)
    if not preserve_index:
        points.index = np.arange(points.shape[0])
    return points


def reindex_trajectories(trajectories, trajnum_colname='n', dt=None):
    """
    Splits the trajectories with missing time steps and assigns different indices to
    the different segments.

    Works with trajectories and translocations.
    """
    are_translocations = 'dt' in trajectories.columns and not np.any(np.isnan(trajectories['dt']))
    trajs = []
    for traj in iter_trajectories(trajectories, trajnum_colname):
        if dt is None:
            if are_translocations:
                dt = traj['dt'].min()
            else:
                dt = traj['t'].diff().min()
            print(dt)
            if dt.size == 0:
                dt = None
            elif np.isclose(dt, 0):
                raise ValueError('multiple rows for the same time point and trajectory')
        if dt is None:
            holes = np.array([])
        elif are_translocations:
            holes = 1.1 * dt <= traj['dt'].values
            holes[-1] = False
            holes = np.flatnonzero(holes)
        else:
            frame_ids = np.round(traj['t'].values/dt)
            steps = np.diff(frame_ids)
            holes = np.flatnonzero(1 < steps)
        if holes.size:
            holes += 1
            i = 0
            for j in holes:
                trajs.append(j-i)
                i = j
            trajs.append(len(traj)-i)
        else:
            trajs.append(len(traj))
    n = np.repeat(np.arange(1,len(trajs)+1), trajs)
    trajs = trajectories.copy()
    trajs[trajnum_colname] = n
    return trajs


def discard_static_trajectories(trajectories, min_msd=None, trajnum_colname='n', full_trajectory=False, verbose=False, localization_error=None):
    """
    Arguments:

        trajectories (DataFrame): trajectory or translocation data with
            columns :const:`'n'` (trajectory number),
            spatial coordinates :const:`'x'` and :const:`'y'` (and optionaly
            :const:`'z'`), and time :const:`'t'`;
            delta columns, if available (translocations), are used instead
            for calculating the displacement length.

        min_msd (float): minimum mean-square-displacement (usually set to the localization error).

        trajnum_colname (str): column name for the trajectory number.

        full_trajectory (float or bool): if :const:`True`, the trajectories with
            static translocations are entirely discarded;
            if False, only the static translocations are discarded, and the
            corresponding trajectories are discarded only if they end up being
            single points;
            if a `float` value, trajectories with `full_trajectory` x 100% static
            translocations or more are discarded.

        localization_error (float): alias for `min_msd`; for backward compatibility.

    Returns:

        DataFrame: filtered trajectory data with a new row index.

    """
    if min_msd is None:
        min_msd = localization_error
    trajs = []
    for start,stop in iter_trajectories(trajectories, trajnum_colname, asslice=True, order='start'):
        traj = trajectories.iloc[start:stop]
        delta_cols = [ col for col in trajectories.columns if col.startswith('d') and col[1:] in trajectories.columns ]
        columns_with_deltas = [ col[1:] for col in delta_cols ]
        if columns_with_deltas:
            dr = traj[[ 'd'+col for col in trajectories.columns if col in 'xyz' ]].values
        else:
            r = traj[[ col for col in trajectories.columns if col in 'xyz' ]].values
            dr = np.diff(r, axis=0)
        js = np.mean(dr * dr, axis=1)
        if columns_with_deltas:
            js[np.isnan(js)] = -1
            static = js < min_msd
        else:
            static = np.r_[False, js < min_msd]
        if verbose and np.any(static):
            print('trajectory {:.0f} exhibits static translocations'.format(traj[trajnum_colname].iat[0]))
        if full_trajectory:
            if isinstance(full_trajectory, bool):
                if np.any(static):
                    # discard the entire trajectory
                    continue
            else:
                threshold_ratio = full_trajectory
                if columns_with_delta:
                    ratio = np.sum(static)/static.size
                else:
                    ratio = (np.sum(static)-1)/(static.size-1)
                if threshold_ratio <= ratio:
                    # discard the entire trajectory
                    continue
        else:
            traj = traj.loc[~static]
        if (columns_with_deltas and 0<len(traj)) or 1<len(traj):
            trajs.append(traj)
    return pd.concat(trajs, ignore_index=True) # preserve column types


def load_mat(path, columns=None, varname='plist', dt=None, coord_scale=None, pixel_size=None):
    """
    Load SPT data from MatLab V7 file.

    Arguments:

        path (str): file path.

        columns (sequence of str): data column names;
            default is ['t', 'x', 'y'].

        varname (str): record name.

        dt (float): frame interval in seconds.

        coord_scale (float): convertion factor for spatial coordinates.

        pixel_size (float): deprecated; superseded by `coord_scale`.

    Returns:

        pandas.DataFrame: SPT data.
    """
    import h5py
    with h5py.File(path, 'r') as f:
        spt_data = f[varname][...]
    if columns is None:
        if spt_data.shape[0]==3:
            columns = list('txy')
        else:
            raise NotImplementedError('cannot infer the column names')
    spt_data = pd.DataFrame(spt_data.T, columns=columns)
    if dt is not None:
        spt_data['t'] = spt_data['t'] * dt
    if coord_scale is None:
        if pixel_size is not None:

            warnings.warn('attribute pixel_size is deprecated; use coord_scale instead', DeprecationWarning)
            coord_scale = pixel_size
    if coord_scale is not None:
        spt_data[list('xy')] = spt_data[list('xy')] * coord_scale
    return spt_data


def trajectories_to_translocations(points, exclude_columns=['n']):
    """
    Appends delta columns (*'dx'*, *'dy'*, *'dt'*, etc) and removes
    the last location of each trajectory.

    See also `translocations_to_trajectories`.
    """
    exclude_columns = list(exclude_columns)
    delta_cols = [ c for c in points.columns \
            if c[0]=='d' and c[1:] not in exclude_columns and c[1:] in points.columns ]
    paired_dest = points['n'].diff().values==0
    paired_src = np.r_[paired_dest[1:], False]
    points = points.copy()
    cols_with_deltas = [ c[1:] for c in delta_cols ]
    cols_to_diff = [ c for c in points.columns if c not in exclude_columns+cols_with_deltas+delta_cols ]
    deltas = points[cols_to_diff].diff().shift(-1)
    deltas = deltas[paired_src]
    deltas.columns = [ 'd'+c for c in cols_to_diff ]
    points = points.join(deltas)
    points.dropna(inplace=True)
    points.index = np.arange(points.shape[0])
    return points

def translocations_to_trajectories(points):
    """
    Reintroduces the last location of each trajectory, and discards
    the delta columns.

    See also `trajectories_to_translocations`.
    """
    delta_cols = [ c for c in points.columns \
            if c[0]=='d' and c[1:] != 'n' and c[1:] in points.columns ]
    cols_with_deltas = [ c[1:] for c in delta_cols ]
    trans_n = points['n'].values
    traj_n = []
    traj_xyt = []
    r = 0
    while r<points.shape[0]:
        n = trans_n[r]
        start = r
        while r<points.shape[0] and trans_n[r]==n:
            r += 1
        stop = r
        trans = points.iloc[start:stop]
        traj = trans[cols_with_deltas].values
        traj_end = traj[-1] + trans[delta_cols].values[-1]
        traj_n.append(np.full((stop-start+1,1), n))
        traj_xyt.append(traj)
        traj_xyt.append(traj_end[np.newaxis,:])
    n = pd.DataFrame(np.vstack(traj_n), columns=['n'])
    xyt = pd.DataFrame(np.vstack(traj_xyt), columns=cols_with_deltas)
    return n.join(xyt)


__all__ = [
    'translocations',
    'iter_trajectories',
    'iter_frames',
    'trajectories_to_translocations',
    'translocations_to_trajectories',
    'load_xyt',
    'load_mat',
    'crop',
    'discard_static_trajectories',
    'reindex_trajectories',
    ]

