# -*- coding: utf-8 -*-

# Copyright © 2017-2018, Institut Pasteur
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
    ixyz = xyz + [i]
    jump = df[ixyz].diff()
    #df[xyz] = jump[xyz]
    #df = df[jump[i] != 0]
    #return df
    jump = jump[jump[i] == 0][xyz]
    return jump#np.sqrt(np.sum(jump * jump, axis=1))


def load_xyt(path, columns=['n', 'x', 'y', 't'], concat=True, return_paths=False, verbose=False):
    """
    Load trajectory files.

    Files are loaded with :func:`~pandas.read_table` with explicit column names.

    Arguments:

        path (str or list of str): path to trajectory file or directory.

        columns (list of str): column names.

        concat (bool): if multiple files are read, return a single DataFrame.

        return_paths (bool): paths to files are returned as second output argument.

        verbose (bool): print extra messages.

    Returns:

        pandas.DataFrame or list or tuple: trajectories as one or multiple DataFrames;
            if `tuple` (with *return_paths*), the trajectories are first, the list
            of filepaths second.
    """
    #if 'n' not in columns:
    #    raise ValueError("trajectory index should be denoted 'n'")
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
    for f in paths:
        try:
            if verbose:
                print('loading file: {}'.format(f))
            dff = pd.read_table(f, names=columns)
        except OSError:
            warnings.warn(f, FileNotFoundWarning)
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
                        raise ValueError("some indices refer to multiple simultaneous trajectories in table: '{}'".format(f))
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
            df.append(dff)
    if concat:
        df = pd.concat(df)
    if return_paths:
        return (df, paths)
    else:
        return df


def crop(points, box, by=None):
    """
    Remove locations outside a bounding box.

    When a location is discarded, the corresponding trajectory is splitted into two distinct
    trajectories.

    Arguments:

        locations (pandas.DataFrame): locations with trajectory indices in column 'n',
            times in column 't' and coordinates in the other columns

        box (array-like): origin and size of the bounding box

        by (str): for translocations only;
            '*start*': crop by translocation starting point; keep the associated
            destination points;
            '*stop*': crop by translocation destination point; keep the associated
            origins;
            trajectories with a single non-terminal point outside the bounding box are
            not splitted

    Returns:

        pandas.DataFrame: filtered locations
    """
    box = np.asarray(box)
    dim = int(box.size / 2)
    support_lower_bound = box[:dim]
    support_size = box[dim:]
    support_upper_bound = support_lower_bound + support_size
    coord_cols = [ c for c in points.columns if c not in ['n', 't'] ]
    within = np.all(np.logical_and(support_lower_bound <= points[coord_cols].values,
        points[coord_cols].values <= support_upper_bound), axis=1)
    if by:
        paired_dest = points['n'].diff().values==0
        paired_src = np.r_[paired_dest[1:], False]
        if by == 'start':
            within[paired_dest] |= within[paired_src]
        elif by == 'stop':
            within[paired_src] |= within[paired_dest]
        else:
            raise ValueError('unsupported value for argument `by`')
    points = points.copy()
    if 'n' in points.columns:
        points['n'] += np.cumsum(np.logical_not(within), dtype=points.index.dtype)
        single_point = 0 < points['n'].diff().values[1:]
        single_point[:-1] = np.logical_and(single_point[:-1], single_point[1:])
        ok = np.r_[True, np.logical_not(single_point)]
        points = points.iloc[ok]
        within = within[ok]
        points['n'] -= (points['n'].diff() - 1).clip_lower(0).cumsum()
    points = points.iloc[within]
    points.index = np.arange(points.shape[0])
    return points


__all__ = [
    'translocations',
    'load_xyt',
    'crop',
    ]

