# -*- coding:utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#    Contributor: Maxime Duval

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""
This module is where functions processing (extracting features) from random
walks are defined.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
import tqdm

from .rw_features import *
from .batch_generation import *


def feature_processing(c_drop={'t_max', 't_min', 'size', 'is_dt_cst', 'dt'},
                       c_lin_scale={'area'},
                       c_sqrt_scale={'step_max', 'perimeter'}, **kwargs):
    """Feature processing function that is applied to a DataFrame of the raw
    features extracted from random walks.

    Parameters
    ----------
    c_drop : set of the columns we want to drop from a DataFrame.
    c_lin_scale : set of the columns we want to divide by the duration of the
        random walk.
    c_sqrt_scale : set of the columns we want to divide by the square root of
        the duration of the random walk.

    Returns
    -------
    Function that can be applied to a DataFrame.
    """
    if 'windows' in kwargs:
        def get_func(df, ws=list(kwargs['windows'])):
            time_cols = [(df[f't_max_{w}'] -
                          df[f't_min_{w}']).astype(np.float64) for w in ws]
            c_drop_all = list(
                map(lambda x: [x + f'_{w}' for w in ws], list(c_drop)))
            c_drop_all = set([x for l in c_drop_all for x in l])
            c_drop_all.add('n')

            c_lin_scale_all = list(
                map(lambda x: [x + f'_{w}' for w in ws], list(c_lin_scale)))
            c_lin_scale_all = set([x for l in c_lin_scale_all for x in l])

            c_sqrt_scale_all = list(
                map(lambda x: [x + f'_{w}' for w in ws], list(c_sqrt_scale)))
            c_sqrt_scale_all = set([x for l in c_sqrt_scale_all for x in l])

            df = df.drop(labels=list(
                set(df.columns).intersection(c_drop_all)), axis=1)
            c_scale = c_lin_scale_all.union(c_sqrt_scale_all)
            df_noscale = df[df.columns.difference(c_scale)]
            df_linscale = [
                df[df.columns.intersection(
                    set([f'{x}_{w}' for x in list(c_lin_scale)]))].add_suffix(
                    '_t') for w in ws]
            df_sqrtscale = [df[df.columns.intersection(
                set([f'{x}_{w}' for x in list(c_sqrt_scale)]))].add_suffix(
                '_sqrt') for w in ws]
            df_scaled = (pd.merge(df_linscale[i].divide(time_cols[i],
                                                        axis='index'),
                                  df_sqrtscale[i].divide(np.sqrt(time_cols[i]),
                                                         axis='index'),
                                  left_index=True, right_index=True)
                         for i in range(len(time_cols)))
            return pd.concat((df_noscale, *df_scaled),
                             axis=1).sort_index(axis=1)
        return get_func
    else:
        def get_func(df):
            c_scale = c_lin_scale.union(c_sqrt_scale)
            time_col = (df['t_max'] - df['t_min']).astype(np.float64)
            df = df.drop(labels=list(
                set(df.columns).intersection(c_drop)), axis=1)
            df_noscale = df[df.columns.difference(c_scale)]
            df_linscale = df[c_lin_scale].add_suffix('_t')
            df_sqrtscale = df[c_sqrt_scale].add_suffix('_sqrt')
            df_scaled = pd.merge(df_linscale.divide(time_col, axis='index'),
                                 df_sqrtscale.divide(np.sqrt(time_col),
                                                     axis='index'),
                                 left_index=True, right_index=True)
            return pd.concat((df_noscale, df_scaled),
                             axis=1).sort_index(axis=1)
        return get_func


def get_features_from_group(args):
    """Function introduced to allow the parametrization of the feature
    extraction : the bool zero_time, if True, makes sure that the starting time
    of the random walk is 0.
    Single parameter to be able to use this function with multiprocessing imap.
    """
    RW_df, zero_time = args
    return get_all_features(RandomWalk(RW_df, zero_time=zero_time))


def extract_features(RWs, nb_process=4, func_feat_process=None, pbar=True):
    """Extracts features from a pandas DataFrame collecting different
    trajectories of random walks.

    Parameters
    ----------
    RWs : pandas DataFrame. Columns : n (Index of the random walk), t, and
        dimensions (x, and/or, y, and/or z).
    nb_process : int or None. Number of processes to use if not None.
    func_feat_process : function to apply to raw features extracted from the
        random walk. Use case : to get rid of unused features in the VAE.

    Returns
    -------
    df : pandas DataFrame of the features extracted from trajectories.
        Index is the id of the trajectory, columns are the names of the
        features extracted.
    """
    df_trajs = RWs.groupby('n')
    n_trajs = df_trajs.agg('count').count().x.astype(int)
    if nb_process is None:
        if pbar:
            mes = 'creating rws'
            list_RWobj = [RandomWalk(group, zero_time=True)
                          for _, group in tqdm.tqdm_notebook(df_trajs,
                                                             total=n_trajs,
                                                             desc=mes)]
            raw_features = list(map(get_all_features, tqdm.tqdm_notebook(
                                list_RWobj, total=n_trajs,
                                desc='extracting features')))
        else:
            list_RWobj = [RandomWalk(group, zero_time=True)
                          for _, group in df_trajs]
            raw_features = list(map(get_all_features, list_RWobj))
    else:
        list_args = [(group.copy(), True) for _, group in df_trajs]
        if pbar:
            with mp.Pool(nb_process) as p:
                raw_features = list(tqdm.tqdm_notebook(
                    p.imap(get_features_from_group, list_args),
                    total=n_trajs,
                    desc='creating objects and extracting features'))
        else:
            with mp.Pool(nb_process) as p:
                raw_features = list(p.imap(get_features_from_group, list_args))
    df = pd.DataFrame.from_dict(raw_features)
    df['n'] = RWs.n.unique()
    df.set_index('n', inplace=True)
    if func_feat_process is not None:
        df = func_feat_process(df)
    return df


def create_and_extract(args):
    """Function that extracts features from a single random walk.
    Used for multiprocessing with a generator.
    """
    rw, rw_dict_prm, kwargs = args
    if 'windows' in kwargs:
        ws = kwargs['windows']
        N = len(rw)
        rw_obj = RandomWalk(rw, zero_time=True, check_useless=False)
        feat_scales = [get_all_features(rw_obj,
                                        id_min=int(N/2-w/2),
                                        id_max=int(N/2+w/2)) for w in ws]
        rw_feat = {}
        for i, w in enumerate(ws):
            rw_feat.update({f'{k}_{w}': v for k, v in feat_scales[i].items()})
    elif 'time' in kwargs:
        rw_feat = get_all_features(RandomWalk(rw, zero_time=True),
                                   time_evol=True,
                                   check_useless=False, **kwargs)
    else:
        rw_feat = get_all_features(RandomWalk(rw, zero_time=True,
                                              check_useless=False))
    rw_feat['n'] = rw.n.iloc[0]
    if not kwargs['get_rw']:
        rw = None
    return rw_feat, rw_dict_prm, rw


def features_creation(n=1000, types=[(RW_gauss_dist,
                                      {'d_l': ('float', 'uni', 0.01, 0.1),
                                       'T_max': ('float', 'exp', 0.1, 1)})],
                      ps=[1], get_rw=False, chunksize=1, nb_pos_min=3,
                      nb_process=None, func_feat_process=None, jump_max=10,
                      **kwargs):
    """Creates and directly extracts features from specified types of random
    walks. Avoids the creation of a pandas DataFrame of the trajectories :
    useful for lowering the RAM usage.

    Parameters
    ----------
    n : int, the number of trajectories we want to generate.
    type : list that describes which random walks we generate.
        Each item is a tuple with 2 elements.
        - The first is the function (see rw_simulation) which characterizes
        which type of random walk it produces.
        - The second is a dictionary whose keys are the parameters we want to
        pass to the random walk function and values the parameters describing
        how those parameter values are generated with `generate_random_number`.
    ps : parameter passed to `np.random.choice` as p, should have same
        dimension as type. Controls the probability distribution of choosing
        some type.
    get_rw : bool, optional. Whether we want to retrieve trajectories or not.
    nb_process : int or None. If int, the number of processes to use to benefit
        from parallelization.
    func_feat_process : function to apply to raw features extracted from the
        random walk. Use case : to get rid of unused features in the VAE.

    Returns
    -------
    df : pandas DataFrame of the features extracted from trajectories.
        Index is the id of the trajectory, columns are the names of the
        features extracted.
    rw_prms : dict that carries info on each trajectory generated. Each value
        is the index in df of the trajectory, each key is a dictionary
        of the parameters used to generate the random walk (except for
        default parameters). It also carries the name of the function which
        generated the random walk.
    df_rws : None or pandas DataFrame depending on get_rw, raw data of the
        trajectories.
    """
    kwargs['get_rw'] = get_rw
    rw_generator = rw_feature_generator(n, types=types, ps=ps,
                                        nb_pos_min=nb_pos_min,
                                        jump_max=jump_max, **kwargs)
    desc = 'creating and extracting rws features'
    if nb_process is None:
        raw_data = list(map(create_and_extract,
                            tqdm.tqdm_notebook(rw_generator,
                                               total=n, desc=desc)))
    else:
        with mp.Pool(nb_process) as p:
            raw_data = list(tqdm.tqdm_notebook(
                p.imap(create_and_extract, rw_generator, chunksize=chunksize),
                total=n, desc=desc))
    raw_features = []
    rw_prms = {}
    list_rws = []
    for i in range(len(raw_data)):
        raw_features.append(raw_data[i][0])
        rw_prms[i] = raw_data[i][1]
        if get_rw:
            list_rws.append(raw_data[i][2])
    df = pd.DataFrame.from_dict(raw_features)
    if func_feat_process is not None:
        df = func_feat_process(df)
    if get_rw:
        df_rws = pd.concat(list_rws).reset_index(drop=True)
    else:
        df_rws = None
    return df, rw_prms, df_rws
