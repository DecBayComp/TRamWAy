"""This module does not export any symbol. It implements the :class:`~tramway.io.store.storable.Storable` class for TRamWAy datatypes."""

from rwa.storable import *
from rwa.generic import *
from rwa.hdf5 import *

# Core datatypes
from tramway.core import Matrix, ArrayChain
from tramway.inference.diffusivity import DV
hdf5_storable(namedtuple_storable(Matrix))
hdf5_storable(default_storable(ArrayChain))
hdf5_storable(default_storable(DV), agnostic=True)

# Scaler
from tramway.spatial.scaler import Scaler
hdf5_storable(default_storable(Scaler), agnostic=True)

# Gas
from tramway.spatial.gas import Gas
hdf5_storable(default_storable(Gas), agnostic=True)

# ArrayGraph
from tramway.spatial.graph.array import ArrayGraph
hdf5_storable(default_storable(ArrayGraph), agnostic=True)

from tramway.spatial.dichotomy import Dichotomy, ConnectedDichotomy
dichotomy_exposes = ['base_edge', 'min_depth', 'max_depth', 'origin', 'lower_bound', 'upper_bound', \
	'min_count', 'max_count', 'subset', 'cell']
hdf5_storable(kwarg_storable(Dichotomy, dichotomy_exposes), agnostic=True)
dichotomy_graph_exposes = dichotomy_exposes + ['adjacency']
hdf5_storable(kwarg_storable(ConnectedDichotomy, exposes=dichotomy_graph_exposes), agnostic=True)

from tramway.tesselation import CellStats, Delaunay, Voronoi, RegularMesh, KDTreeMesh, KMeansMesh, \
	GasMesh
# CellStats
hdf5_storable(default_storable(CellStats), agnostic=True)
# Delaunay
delaunay_exposes = ['scaler', 'cell_adjacency', 'cell_label', 'adjacency_label', 'cell_centers']
# cell_centers should be last
hdf5_storable(default_storable(Delaunay, exposes=delaunay_exposes), agnostic=True)
# Voronoi
voronoi_exposes = ['cell_vertices', 'ridge_vertices'] + delaunay_exposes
hdf5_storable(default_storable(Voronoi, exposes=voronoi_exposes), agnostic=True)
# RegularMesh
regular_mesh_exposes = voronoi_exposes + ['lower_bound', 'upper_bound', 'count_per_dim', \
	'min_probability', 'max_probability', 'avg_probability']
hdf5_storable(default_storable(RegularMesh, exposes=regular_mesh_exposes), agnostic=True)
# KDTreeMesh
kdtree_mesh_exposes = voronoi_exposes + ['_min_distance', '_avg_distance', \
	'min_probability', 'dichotomy']
hdf5_storable(default_storable(KDTreeMesh, exposes=kdtree_mesh_exposes), agnostic=True)
# KMeansMesh
kmeans_mesh_exposes = voronoi_exposes + ['avg_probability']
hdf5_storable(default_storable(KMeansMesh, exposes=kmeans_mesh_exposes), agnostic=True)
# GasMesh
gas_mesh_exposes = voronoi_exposes + ['gas', '_min_distance', '_max_distance']
hdf5_storable(default_storable(GasMesh, exposes=gas_mesh_exposes), agnostic=True)


from tramway.inference.base import Cell, Distributed
hdf5_storable(default_storable(Cell), agnostic=True)
hdf5_storable(default_storable(Distributed), agnostic=True)

