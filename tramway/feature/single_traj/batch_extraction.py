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

import numpy as np
import pandas as pd

from .rw_features import *


def extract_features(RWs):
    df_trajs = RWs.groupby('n')
    n_trajs = df_trajs.agg('count').count().x.astype(int)
    features = [get_all_features(RandomWalk(group))
                for id_g, group in df_trajs]
    return features