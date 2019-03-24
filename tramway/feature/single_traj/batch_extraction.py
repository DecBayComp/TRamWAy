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


def extract_features(RWs, nb_process=4, func_feat_process=None):
    df_trajs = RWs.groupby('n')
    n_trajs = df_trajs.agg('count').count().x.astype(int)
    list_RWobj = [RandomWalk(group, zero_time=True)
                  for _, group in tqdm.tqdm_notebook(df_trajs,
                                                     total=n_trajs,
                                                     desc='creating objects')]
    if nb_process is None:
        print('here')
        raw_features = list(map(get_all_features, tqdm.tqdm_notebook(
                        list_RWobj, total=n_trajs,
                        desc='extracting features')))
    else:
        with mp.Pool(nb_process) as p:
            raw_features = list(tqdm.tqdm_notebook(
                    p.imap(get_all_features, list_RWobj),
                    total=n_trajs, desc='extracing features'))
    df = pd.DataFrame.from_dict(raw_features)
    if func_feat_process is not None:
        df.apply()
    return df


