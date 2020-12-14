
from . import *
from . import abc
from . import proxy
from . import proxied
from . import plugin
from . import post
from ..attribute import InitializerMethod

from_plugin   = InitializerMethod( TessellerInitializer.from_plugin   )
from_callable = InitializerMethod( TessellerInitializer.from_callable )

