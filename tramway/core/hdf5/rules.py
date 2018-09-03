# -*- coding: utf-8 -*-

# Copyright © 2017-2018, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

from rwa import *

__all__ = []

# Core datatypes
from tramway.core import Lazy, Matrix, ArrayChain, Scaler
lazy_exposes = list(Lazy.__slots__)
__all__.append('lazy_exposes')
try:
        hdf5_storable(namedtuple_storable(Matrix))
        hdf5_storable(default_storable(ArrayChain))
        hdf5_storable(default_storable(Scaler), agnostic=True)
except NameError: # in rtd
        pass

from tramway.tessellation.base import Tessellation, Delaunay, Voronoi
# Delaunay
tessellation_exposes = lazy_exposes + list(Tessellation.__slots__) # not a storable
__all__.append('tessellation_exposes')
delaunay_exposes = tessellation_exposes + list(Delaunay.__slots__)#['_cell_centers']
__all__.append('delaunay_exposes')
hdf5_storable(default_storable(Delaunay, exposes=delaunay_exposes), agnostic=True)
# Voronoi
voronoi_exposes = delaunay_exposes + list(Voronoi.__slots__)#['_vertices', '_vertex_adjacency', '_cell_vertices']
__all__.append('voronoi_exposes')
try:
        hdf5_storable(default_storable(Voronoi, exposes=voronoi_exposes), agnostic=True)
except NameError: # in rtd
        pass

# RegularMesh
try:
        from tramway.tessellation.grid import RegularMesh
except ImportError:
        pass
else:
        regular_mesh_exposes = voronoi_exposes + \
                ['_diagonal_adjacency', 'lower_bound', 'upper_bound', 'count_per_dim', \
                        'min_probability', 'max_probability', 'avg_probability', \
                        'min_distance', 'avg_distance', 'grid']
        __all__.append('regular_mesh_exposes')
        try:
                hdf5_storable(default_storable(RegularMesh, exposes=regular_mesh_exposes), agnostic=True)
        except NameError: # in rtd
                pass

# KDTreeMesh
try:
        from tramway.tessellation.kdtree import KDTreeMesh
except ImportError:
        pass
else:
        kdtree_mesh_exposes = voronoi_exposes + ['_min_distance', '_avg_distance', \
                'min_probability', 'max_probability', 'max_level', 'dichotomy', 'reference_length']
        __all__.append('kdtree_mesh_exposes')
        try:
                hdf5_storable(default_storable(KDTreeMesh, exposes=kdtree_mesh_exposes), agnostic=True)
        except NameError: # in rtd
                pass

# KMeansMesh
try:
        from tramway.tessellation.kmeans import KMeansMesh
except ImportError:
        pass
else:
        kmeans_mesh_exposes = voronoi_exposes + ['_min_distance', 'avg_probability']
        __all__.append('kmeans_mesh_exposes')
        try:
                hdf5_storable(default_storable(KMeansMesh, exposes=kmeans_mesh_exposes), agnostic=True)
        except NameError: # in rtd
                pass

# GasMesh
try:
        from tramway.tessellation.gwr import GasMesh
except ImportError:
        pass
else:
        # Gas
        from tramway.tessellation.gwr.gas import Gas
        try:
                hdf5_storable(default_storable(Gas), agnostic=True)
        except NameError: # in rtd
                pass
        # ArrayGraph
        from tramway.tessellation.gwr.graph.base import Graph
        graph_exposes = list(Graph.__slots__)
        __all__.append('graph_exposes')
        from tramway.tessellation.gwr.graph.array import ArrayGraph
        array_graph_exposes = graph_exposes + list(ArrayGraph.__slots__)
        __all__.append('array_graph_exposes')
        try:
                hdf5_storable(default_storable(ArrayGraph, exposes=array_graph_exposes), agnostic=True)
        except NameError: # in rtd
                pass
        gas_mesh_exposes = voronoi_exposes + ['gas', '_min_distance', '_avg_distance', '_max_distance', \
                'min_probability']
        __all__.append('gas_mesh_exposes')
        try:
                hdf5_storable(default_storable(GasMesh, exposes=gas_mesh_exposes), agnostic=True)
        except NameError: # in rtd
                pass

# TimeLattice
try:
        from tramway.tessellation.time import TimeLattice
except ImportError:
        pass
else:
        time_lattice_exposes = tessellation_exposes + list(TimeLattice.__slots__)
        __all__.append('time_lattice_exposes')
        try:
                hdf5_storable(default_storable(TimeLattice, exposes=time_lattice_exposes), agnostic=True)
        except NameError: # in rtd
                pass

# NestedTessellations
try:
        from tramway.tessellation.nesting import NestedTessellations
except ImportError:
        pass
else:
        nested_tessellations_expose = tessellation_exposes + list(NestedTessellations.__slots__)#\
        #       [ _s for _s in NestedTessellations.__slots__ if _s not in ('child_factory',) ]
        __all__.append('nested_tessellations_expose')
        try:
                hdf5_storable(default_storable(NestedTessellations, exposes=nested_tessellations_expose), \
                        agnostic=True)
        except NameError: # in rtd
                pass


from tramway.inference.base import Local, Distributed, Cell, Locations, Translocations
local_exposes = lazy_exposes + list(Local.__slots__)
__all__.append('local_exposes')
cell_exposes = local_exposes + list(Cell.__slots__)
__all__.append('cell_exposes')
locations_expose = cell_exposes + list(Locations.__slots__)
__all__.append('locations_expose')
try:
        hdf5_storable(default_storable(Locations, exposes=locations_expose), agnostic=True)
except NameError: # in rtd
        pass
translocations_expose = cell_exposes + list(Translocations.__slots__)
__all__.append('translocations_expose')
try:
        hdf5_storable(default_storable(Translocations, exposes=translocations_expose), agnostic=True)
except NameError: # in rtd
        pass
distributed_exposes = local_exposes + list(Distributed.__slots__)
__all__.append('distributed_exposes')
try:
        hdf5_storable(default_storable(Distributed, exposes=distributed_exposes), agnostic=True)
except NameError: # in rtd
        pass

try:
        from tramway.inference.dv import DV
except ImportError:
        pass
else:
        try:
                hdf5_storable(default_storable(DV), agnostic=True)
        except NameError: # in rtd
                pass

