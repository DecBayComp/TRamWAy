from .rw_simulation import *
from .rw_features import *
from .batch_generation import *
from .batch_extraction import *
from .misc import *
try:
    from .vae import *
except ImportError: # torch
    pass
from .visualization import *
try:
    from .utils_supervised import *
except ImportError: # torch
    pass

"""
Using this module or subpackage is strongly discouraged,
as this code does not properly implement all of the featured measurements.
"""

import warnings
_msg = '''\
using the tramway.feature.single_traj module is strongly discouraged,
as this code does not properly implement all of the featured measurements\
'''
warnings.warn(_msg, FutureWarning)
