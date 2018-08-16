# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
import re
import collections

def isstructured(x):
        """
        Check for named columns.

        The adjective *structured* comes from NumPy structured array.

        Arguments:
                x (any): any datatype

        Returns:
                bool: ``True`` if input argument ``x`` has named columns.

        """
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
                return True
        else:
                try:
                        return bool(x.dtype.names)
                except AttributeError:
                        return False

def columns(x):
        """
        Get column names.

        Arguments:
                x (any): 
                        datatype that satisfies :func:`isstructured`.

        Returns:
                iterable:
                        column iterator.

        Raises:
                ValueError: if no named columns are found in `x`.

        """ 
        if isinstance(x, pd.DataFrame):
                return x.columns
        elif isinstance(x, pd.Series):
                return x.index
        elif x.dtype.names:
                return x.dtype.names
        else:
                raise ValueError('not structured')

def splitcoord(varnames):
        coord = re.compile('[a-z][0-9]*$')
        vs = collections.defaultdict(list)
        for v in varnames:
                u = [ w[::-1] for w in v[::-1].split(None, 1) ]
                if u[1:] and coord.match(u[0]):
                        vs[u[1]].append(v)
                else:
                        vs[v].append(v)
        return vs


__all__ = [
        'isstructured',
        'columns',
        'splitcoord',
        ]

