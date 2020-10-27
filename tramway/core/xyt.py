# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
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


def _translocations(df, sort=True): # very slow; may soon be deprecated
    def diff(df):
        df = df.sort_values('t')
        t = df['t'][1:]
        df = df.diff()
        df['t'] = t
        return df
    return df.groupby(['n'], sort=False).apply(diff)

def translocations(df, sort=False):
    '''each trajectories should be represented by consecutive rows sorted by time.'''
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


def iter_trajectories(trajectories, trajnum_colname='n', asslice=False, asarray=False):
    if not isinstance(trajectories, pd.DataFrame):
        raise TypeError('trajectories is not a DataFrame')

    if asarray or not asslice:
        other_cols = [ col != trajnum_colname for col in trajectories.columns ]
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


def crop(points, box, by=None, add_deltas=True, keep_nans=False, no_deltas=False):
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

        keep_nans (bool): adding deltas generates NaN; keep them

        no_deltas (bool): do not consider any column as deltas

    Returns:

        pandas.DataFrame: filtered locations
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
        points['n'] += np.cumsum(np.logical_not(within), dtype=points.index.dtype)
        single_point = 0 < points['n'].diff().values[1:]
        single_point[:-1] = np.logical_and(single_point[:-1], single_point[1:])
        ok = np.r_[True, np.logical_not(single_point)]
        points = points[ok]
        within = within[ok]
        points['n'] -= (points['n'].diff().fillna(0).astype(int) - 1).clip(lower=0).cumsum()
    points = points[within]
    if not keep_nans:
        points.dropna(inplace=True)
    points.index = np.arange(points.shape[0])
    return points


def discard_static_trajectories(trajectories, min_msd=None, trajnum_colname='n', full_trajectory=False, verbose=False, localization_error=None):
    """
    Arguments:

        trajectories (DataFrame): trajectory data with columns 'n' (trajectory number),
            spatial coordinates 'x' and 'y' (and optionaly 'z'), and time 't'.

        min_msd (float): minimum mean-square-displacement (usually set to the localization error).

        trajnum_colname (str): column name for the trajectory number.

        full_trajectory (bool): if True, the trajectories with static translocations
            are entirely discarded; if False, only the static translocations are
            discarded, and the corresponding trajectories are discarded only if they
            end up being single points.

        localization_error (float): alias for `min_msd`; for backward compatibility.

    Returns:

        DataFrame: filtered trajectory data with a new row index.

    """
    if min_msd is None:
        min_msd = localization_error
    trajs = []
    for start,stop in iter_trajectories(trajectories, trajnum_colname, asslice=True):
        traj = trajectories.iloc[start:stop]
        r = traj[[ col for col in trajectories.columns if col in 'xyz' ]].values
        dr = np.diff(r, axis=0)
        js = np.mean(dr * dr, axis=1)
        static = np.r_[False, js < min_msd]
        if verbose and np.any(static):
            print('trajectory {:.0f} exhibits static translocations'.format(traj[trajnum_colname].iat[0]))
        if full_trajectory and np.any(static):
            # discard the entire trajectory
            continue
        traj = traj.iloc[~static]
        if 1<len(traj):
            trajs.append(traj)
    return pd.DataFrame(np.vstack([ traj.values for traj in trajs ]), columns=trajectories.columns)


def load_mat(path, columns=None, varname='plist', dt=None, coord_scale=None, pixel_size=None):
    """
    Load SPT data from MatLab V7 file.

    Arguments:

        path (str): file path.

        columns (sequence of str): data column names.

        varname (str): record name.

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


__all__ = [
    'translocations',
    'iter_trajectories',
    'load_xyt',
    'load_mat',
    'crop',
    'discard_static_trajectories',
    ]

