# -*- coding: utf-8 -*-

# Copyright © 2018, Institut Pasteur
#   Contributor: Alexander Serov

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


"""
Small functions used throughout the package
"""


def n_pi_func(dim):
    return 5 - dim


def p(n, dim):
    n_pi = n_pi_func(dim)
    p = dim * (n + n_pi - 1.0) / 2.0 - 1.0
    return p
