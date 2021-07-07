
from . import *
from . import abc
from . import proxy
from . import proxied
from . import plugin
from . import post
from ..attribute import initializer_method

from_plugin   = initializer_method( TessellerInitializer.from_plugin   )
from_callable = initializer_method( TessellerInitializer.from_callable )

