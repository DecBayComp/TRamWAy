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
from tramway.core import Lazy, Matrix, ArrayChain, Analyses
from tramway.inference.diffusivity import DV
lazy_exposes = list(Lazy.__slots__)
analyses_expose = lazy_exposes + list(Analyses.__slots__)
hdf5_storable(namedtuple_storable(Matrix))
hdf5_storable(default_storable(ArrayChain))
hdf5_storable(default_storable(DV), agnostic=True)
hdf5_storable(default_storable(Analyses, exposes=analyses_expose), agnostic=True)

# Scaler
from tramway.spatial.scaler import Scaler
hdf5_storable(default_storable(Scaler), agnostic=True)

# Gas
from tramway.spatial.gas import Gas
hdf5_storable(default_storable(Gas), agnostic=True)

# ArrayGraph
from tramway.spatial.graph.array import ArrayGraph
hdf5_storable(default_storable(ArrayGraph), agnostic=True)

from tramway.tesselation import CellStats, Tesselation, Delaunay, Voronoi, \
	RegularMesh, KDTreeMesh, KMeansMesh, GasMesh, \
	TimeLattice, NestedTesselations
# CellStats
hdf5_storable(default_storable(CellStats), agnostic=True)
# Delaunay
tesselation_exposes = lazy_exposes + list(Tesselation.__slots__) # not a storable
delaunay_exposes = tesselation_exposes + ['_cell_centers']
hdf5_storable(default_storable(Delaunay, exposes=delaunay_exposes), agnostic=True)
# Voronoi
voronoi_exposes = delaunay_exposes + ['_vertices', '_vertex_adjacency', '_cell_vertices']
hdf5_storable(default_storable(Voronoi, exposes=voronoi_exposes), agnostic=True)
# RegularMesh
regular_mesh_exposes = voronoi_exposes + \
	['_diagonal_adjacency', 'lower_bound', 'upper_bound', 'count_per_dim', \
		'min_probability', 'max_probability', 'avg_probability']
hdf5_storable(default_storable(RegularMesh, exposes=regular_mesh_exposes), agnostic=True)
# KDTreeMesh
kdtree_mesh_exposes = voronoi_exposes + ['_min_distance', '_avg_distance', \
	'min_probability', 'max_probability', 'max_level', 'dichotomy']
hdf5_storable(default_storable(KDTreeMesh, exposes=kdtree_mesh_exposes), agnostic=True)
# KMeansMesh
kmeans_mesh_exposes = voronoi_exposes + ['_min_distance', 'avg_probability']
hdf5_storable(default_storable(KMeansMesh, exposes=kmeans_mesh_exposes), agnostic=True)
# GasMesh
gas_mesh_exposes = voronoi_exposes + ['gas', '_min_distance', '_avg_distance', '_max_distance', \
	'min_probability']
hdf5_storable(default_storable(GasMesh, exposes=gas_mesh_exposes), agnostic=True)
# TimeLattice
time_lattice_exposes = tesselation_exposes + list(TimeLattice.__slots__)
hdf5_storable(default_storable(TimeLattice, exposes=time_lattice_exposes), agnostic=True)
# NestedTesselations
nested_tesselations_expose = tesselation_exposes + list(NestedTesselations.__slots__)#\
#	[ _s for _s in NestedTesselations.__slots__ if _s not in ('child_factory',) ]
hdf5_storable(default_storable(NestedTesselations, exposes=nested_tesselations_expose), agnostic=True)


from tramway.inference.base import Local, Cell, Distributed
local_exposes = lazy_exposes + list(Local.__slots__)
cell_exposes = local_exposes + list(Cell.__slots__)
hdf5_storable(default_storable(Cell, exposes=cell_exposes), agnostic=True)
distributed_exposes = local_exposes + list(Distributed.__slots__)
hdf5_storable(default_storable(Distributed, exposes=distributed_exposes), agnostic=True)

