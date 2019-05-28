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
This module is where functions creating multiple random walks at a time are
defined.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
import tqdm

from .rw_simulation import *
from .rw_misc import rw_is_useless


def generate_random_number(type_n, type_gen, a, b=None, p=None):
    """
    Useful function that outputs a randomly generated number or string
    depending on the given parameters

    Parameters
    ----------
    type_n : string, the type of the random number generated.
    type_gen : string, the type of random distribution chosen.
    a : number, parameter relevant to the chosen distribution.
    b : number, optional depending on the type of generation chosen.
    """
    if type_n == 'float':
        if type_gen == 'uni':
            return np.random.rand() * (b-a) + a
        elif type_gen == 'exp':
            return 10**(np.random.rand() * (np.log10(b)-np.log10(a)) +
                        np.log10(a))
        else:
            raise TypeError(f'Unrecognized type : {type_gen}')
    elif type_n == 'int':
        if type_gen == 'uni':
            return np.random.randint(a, b+1)
        elif type_gen == 'exp':
            return (10**(np.random.rand() * (np.log10(b)-np.log10(a)) +
                         np.log10(a))).astype(int)
        else:
            raise TypeError(f'Unrecognized type : {type_gen}')
    elif type_n == 'str':
        i = np.random.choice(len(a), p=p)
        return a[i]
    elif type_n == 'cst':
        return a
    else:
        raise TypeError(f'Unrecognized number type : {type_n}')


def create_random_rw(args):
    """Returns a tuple of the simulated random walk and the parameters of the
    simulation a probability distribution of possible random walks.

    Parameters
    ----------
    args : tuple of (ps (list or None or int, should be accepted by
        np.random.choice as the p parameter), types (see
        "rw_feature_generator")).

    Returns
    -------
    rw : pandas DataFrame of the position/time of the simulated random walk.
    rw_dict_prms : dict of the parameters of the simulated random walk.
    """
    ps, types, nb_pos_min, jump_max, id_traj = args
    rw_id_type = np.random.choice(len(types), p=ps)
    rw_dict_prms = {}
    for prm, val in types[rw_id_type][1].items():
        rw_dict_prms[prm] = generate_random_number(*val)
    rw = types[rw_id_type][0](**rw_dict_prms)
    while rw_is_useless(rw, nb_pos_min, jump_max):
        rw_dict_prms = {}
        for prm, val in types[rw_id_type][1].items():
            rw_dict_prms[prm] = generate_random_number(*val)
        rw = types[rw_id_type][0](**rw_dict_prms)
    rw['n'] = id_traj
    rw_dict_prms['func_name'] = types[rw_id_type][0].__name__
    return (rw, rw_dict_prms)


def rw_feature_generator(n, types=[(RW_gauss_dist,
                                    {'d_l': ('float', 'exp', 0.01, 0.1),
                                     'T_max': ('float', 'exp', 0.1, 1)})],
                         ps=[1], nb_pos_min=3, jump_max=10, **kwargs):
    """
    Generator of random walks.

    Parameters
    ----------
    n : int, the number of random walks we want to generate.
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

    Yields
    ------
    A tuple of :
        - the list of functions used to extract features out of the random walk
        - the id of the random walk generated
        - rw, the random walk pandas.DataFrame object
        - ns, the feature extraction parameter
        - rw_dict_prms, the dictionary of parameters relevant to the random
        walk generation.
        - get_rw, the bool describing if we want to eventually return the
        random walk.
    """
    for i in range(n):
        yield (create_random_rw((ps, types, nb_pos_min, jump_max, i)) +
               (kwargs,))


def create_batch_rw(n=100, ps=[1], nb_process=4, nb_pos_min=2, jump_max=10,
                    chunksize=10, pbar=True,
                    types=[(RW_gauss_dist,
                            {'d_l': ('float', 'exp', 0.01, 0.1),
                             'T_max': ('float', 'exp', 0.1, 1)})]):
    """Function which uses multiprocessing to rapidly create multiple random
    walks.

    Parameters
    ----------
    n : int, the number of random walks we want to generate.
    ps : parameter passed to `np.random.choice` as p, should have same
        dimension as type. Controls the probability distribution of choosing
        some type.
    nb_process : number of CPU processes used simultaneous (makes use of
        multiprocessing built-in Python module).
    type : list that describes which random walks we generate.
        Each item is a tuple with 2 elements.
        - The first is the function (see rw_simulation) which characterizes
        which type of random walk it produces.
        - The second is a dictionary whose keys are the parameters we want to
        pass to the random walk function and values the parameters describing
        how those parameter values are generated with `generate_random_number`.

    Returns
    -------
    - pandas DataFrame of the simulated random walks.
    - dictionary of dictionaries. Each key is the index of the random walk,
        each value is a dictionary of the values taken by the parameters of the
        random walk.
    """
    prms = [(ps, types, nb_pos_min, jump_max, i) for i in range(n)]
    if pbar:
        if nb_process is None:
            raw_data = list(map(create_random_rw,
                                tqdm.tqdm_notebook(prms,
                                                   desc='generating RWs')))
        else:
            with mp.Pool(nb_process) as p:
                raw_data = list(tqdm.tqdm_notebook(
                    p.imap(create_random_rw, prms, chunksize),
                    total=n, desc='generating RWs'))
    else:
        if nb_process is None:
            raw_data = list(map(create_random_rw, prms))
        else:
            with mp.Pool(nb_process) as p:
                raw_data = list(p.imap(create_random_rw, prms, chunksize))
    RW_data = []
    RW_prm = {}
    for traj, rw in enumerate(raw_data):
        RW_prm[traj] = rw[1]
        RW_data.append(rw[0])
    return pd.concat(RW_data).reset_index(drop=True), RW_prm
