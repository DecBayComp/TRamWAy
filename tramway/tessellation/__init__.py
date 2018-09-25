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
from .time import *
from .nesting import *
from tramway.core.plugin import Plugins
import os.path


try:
    from .grid import RegularMesh
except ImportError:
    pass


plugins = Plugins(os.path.dirname(__file__), __package__, \
    {'make': Tessellation}, require='setup')

