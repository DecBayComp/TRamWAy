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


def generate_random_number(type_n, type_gen, a, b=None):
    """
    Useful function that can output different random numbers based on the
    parameters given

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
        return np.random.choice(a)
    elif type_n == 'cst':
        return a
    else:
        raise TypeError(f'Unrecognized number type : {type_n}')


def rw_feature_generator(n, types=[(RW_gauss_dist,
                                    {'d_l': ('float', 'exp', 0.01, 0.1),
                                     'T_max': ('float', 'exp', 0.1, 1)})],
                         ps=[1]):
    """
    Generator of random walks.

    Parameters
    ----------
    n : int, the number of random walks we want to generate.
    type : list that describes which random walks we generate.
        Each item is tuple with 2 elements.
        - The first is the function (see RW_gen_pd_cleaner) which characterizes
        which type of random walk it produces.
        - The second is a dictionary whose keys are the parameters we want to
        pass to the random walk function and values the parameters describing
        how those parameter values are generated with `generate_random_number`.
    ps : parameter passed to `np.random.choice` as p, probability of choosing
        some element of types.
    ns : int, parameter of the feature extraction process.
    flist : list of functions that extract features from the generated random
        walk.
    get_rw : bool, whether we want to return the random walk generated (if not
        only the features and random walk parameters are returned).

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
        rw_id_type = np.random.choice(len(types), p=ps)
        rw_dict_prms = {}
        for prm, val in types[rw_id_type][1].items():
            rw_dict_prms[prm] = generate_random_number(*val)
        rw = types[rw_id_type][0](**rw_dict_prms)
        rw_dict_prms['func_name'] = types[rw_id_type][0].__name__
        yield (i, rw, rw_dict_prms)


def create_random_rw(args):
    ps, types = args
    rw_id_type = np.random.choice(len(types), p=ps)
    rw_dict_prms = {}
    for prm, val in types[rw_id_type][1].items():
        rw_dict_prms[prm] = generate_random_number(*val)
    rw = types[rw_id_type][0](**rw_dict_prms)
    rw_dict_prms['func_name'] = types[rw_id_type][0].__name__
    return (rw, rw_dict_prms)


def create_batch_rw(n=100, ps=[1], nb_process=4,
                    types=[(RW_gauss_dist,
                            {'d_l': ('float', 'exp', 0.01, 0.1),
                             'T_max': ('float', 'exp', 0.1, 1)})]):
    with mp.Pool(nb_process) as p:
        raw_data = list(tqdm.tqdm_notebook(
            p.imap(create_random_rw, [(ps, types)] * n), total=n))
    RW_data = [rw[0] for rw in raw_data]
    RW_prm = {i: rw[1] for i, rw in enumerate(raw_data)}
    for i, rw in enumerate(RW_data):
        RW_data[i]['n'] = i
    return pd.concat(RW_data).reset_index(drop=True), RW_prm
