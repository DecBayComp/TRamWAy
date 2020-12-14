
from . import *
from ..attribute import InitializerMethod

from_voronoi                = InitializerMethod( SamplerInitializer.from_voronoi                )
from_spheres                = InitializerMethod( SamplerInitializer.from_spheres                )
from_nearest_neighbors      = InitializerMethod( SamplerInitializer.from_nearest_neighbors      )
from_nearest_time_neighbors = InitializerMethod( SamplerInitializer.from_nearest_time_neighbors )

