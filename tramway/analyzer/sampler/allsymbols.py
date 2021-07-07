
from . import *
from ..attribute import initializer_method

from_voronoi                = initializer_method( SamplerInitializer.from_voronoi                )
from_spheres                = initializer_method( SamplerInitializer.from_spheres                )
from_nearest_neighbors      = initializer_method( SamplerInitializer.from_nearest_neighbors      )
from_nearest_time_neighbors = initializer_method( SamplerInitializer.from_nearest_time_neighbors )

