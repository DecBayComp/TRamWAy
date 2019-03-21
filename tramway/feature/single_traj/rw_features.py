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
This module is where features extractible from a single random walk are
defined.
"""

import numpy as np
import pandas as pd


class RandomWalk():

    def __init__(self, RW_df):
        self.data = RW_df
        self.dims = set(RW_df.columns).intersection({'x', 'y', 'z'})
        self.length = len(self.data)
        self.position = self.data.loc[:, list(dims)].values
        self.t = self.data.t.values
        self.Dvec = self.position[:, np.newaxis] - self.position[np.newaxis, :]
        self.Dabs = np.linalg.norm(self.Dvec, axis=2)
    
    def __len__(self):
        return self.length
