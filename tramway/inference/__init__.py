# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .base import *
from .diffusivity import *
import os.path
from tramway.core.plugin import *

#__all__ = ['Local', 'Cell', 'Distributed', 'distributed', 'd_neg_posterior', 'inferD', 'dv_neg_posterior', 'inferDV', 'DV']

__plugin_package__ = 'modes'
all_modes = list_plugins(os.path.join(os.path.dirname(__file__), __plugin_package__),
		'.'.join((__package__, __plugin_package__)),
		{'infer': 'infer.*'})

