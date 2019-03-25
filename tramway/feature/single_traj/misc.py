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
Useful functions
"""

import glob
import pickle

import tqdm
import pandas as pd


def extract_i(file):
    return file.split('\\')[3].split('_')[2].split('.')[0]


def concat_job_files(job_name, df_path='Y:\data\df', dict_path='Y:\data\dict'):
    dfs = []
    prms = {}
    df_paths = glob.glob(f'{df_path}\df_{job_name}_*')
    dict_paths = glob.glob(f'{dict_path}\dict_{job_name}_*')
    for df_path, dict_path in tqdm.tqdm_notebook(zip(df_paths, dict_paths),
                                                 total=len(df_paths)):
        try:
            dfs.append(pd.read_csv(df_path, index_col=0))
            with open(dict_path, 'rb') as handle:
                dict_i = pickle.load(handle)
            prms = {**prms, **dict_i}
        except:
            print(f'Could not add file {extract_i(df_path)}')
    df = pd.concat(dfs, sort=True)
    return prms, df
