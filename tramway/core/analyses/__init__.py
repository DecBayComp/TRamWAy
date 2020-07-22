
from .base import *
try: # overwrite
    from .lazy import *
except ImportError:
    pass

