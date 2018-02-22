# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

from rwa import *

# Core datatypes
from tramway.core import Lazy, Matrix, ArrayChain, Scaler
lazy_exposes = list(Lazy.__slots__)
hdf5_storable(namedtuple_storable(Matrix))
hdf5_storable(default_storable(ArrayChain))
hdf5_storable(default_storable(Scaler), agnostic=True)

from tramway.tessellation.base import Tessellation, Delaunay, Voronoi
# Delaunay
tessellation_exposes = lazy_exposes + list(Tessellation.__slots__) # not a storable
delaunay_exposes = tessellation_exposes + ['_cell_centers']
hdf5_storable(default_storable(Delaunay, exposes=delaunay_exposes), agnostic=True)
# Voronoi
voronoi_exposes = delaunay_exposes + ['_vertices', '_vertex_adjacency', '_cell_vertices']
hdf5_storable(default_storable(Voronoi, exposes=voronoi_exposes), agnostic=True)

# RegularMesh
try:
	from tramway.tessellation.grid import RegularMesh
except ImportError:
	pass
else:
	regular_mesh_exposes = voronoi_exposes + \
		['_diagonal_adjacency', 'lower_bound', 'upper_bound', 'count_per_dim', \
			'min_probability', 'max_probability', 'avg_probability', 'grid']
	hdf5_storable(default_storable(RegularMesh, exposes=regular_mesh_exposes), agnostic=True)

# KDTreeMesh
try:
	from tramway.tessellation.kdtree import KDTreeMesh
except ImportError:
	pass
else:
	kdtree_mesh_exposes = voronoi_exposes + ['_min_distance', '_avg_distance', \
		'min_probability', 'max_probability', 'max_level', 'dichotomy', 'reference_length']
	hdf5_storable(default_storable(KDTreeMesh, exposes=kdtree_mesh_exposes), agnostic=True)

# KMeansMesh
try:
	from tramway.tessellation.kmeans import KMeansMesh
except ImportError:
	pass
else:
	kmeans_mesh_exposes = voronoi_exposes + ['_min_distance', 'avg_probability']
	hdf5_storable(default_storable(KMeansMesh, exposes=kmeans_mesh_exposes), agnostic=True)

# GasMesh
try:
	from tramway.tessellation.gwr import GasMesh
except ImportError:
	pass
else:
	# Gas
	from tramway.tessellation.gwr.gas import Gas
	hdf5_storable(default_storable(Gas), agnostic=True)
	# ArrayGraph
	from tramway.tessellation.gwr.graph.base import Graph
	graph_exposes = list(Graph.__slots__)
	from tramway.tessellation.gwr.graph.array import ArrayGraph
	array_graph_exposes = graph_exposes + list(ArrayGraph.__slots__)
	hdf5_storable(default_storable(ArrayGraph, exposes=array_graph_exposes), agnostic=True)
	gas_mesh_exposes = voronoi_exposes + ['gas', '_min_distance', '_avg_distance', '_max_distance', \
		'min_probability']
	hdf5_storable(default_storable(GasMesh, exposes=gas_mesh_exposes), agnostic=True)

# TimeLattice
try:
	from tramway.tessellation.time import TimeLattice
except ImportError:
	pass
else:
	time_lattice_exposes = tessellation_exposes + list(TimeLattice.__slots__)
	hdf5_storable(default_storable(TimeLattice, exposes=time_lattice_exposes), agnostic=True)

# NestedTessellations
try:
	from tramway.tessellation.nesting import NestedTessellations
except ImportError:
	pass
else:
	nested_tessellations_expose = tessellation_exposes + list(NestedTessellations.__slots__)#\
	#	[ _s for _s in NestedTessellations.__slots__ if _s not in ('child_factory',) ]
	hdf5_storable(default_storable(NestedTessellations, exposes=nested_tessellations_expose), \
		agnostic=True)


from tramway.inference.base import Local, Distributed, Cell, Locations, Translocations
local_exposes = lazy_exposes + list(Local.__slots__)
cell_exposes = local_exposes + list(Cell.__slots__)
locations_expose = cell_exposes + list(Locations.__slots__)
hdf5_storable(default_storable(Locations, exposes=locations_expose), agnostic=True)
translocations_expose = cell_exposes + list(Translocations.__slots__)
hdf5_storable(default_storable(Translocations, exposes=translocations_expose), agnostic=True)
distributed_exposes = local_exposes + list(Distributed.__slots__)
hdf5_storable(default_storable(Distributed, exposes=distributed_exposes), agnostic=True)

try:
	from tramway.inference.dv import DV
except ImportError:
	pass
else:
	hdf5_storable(default_storable(DV), agnostic=True)

