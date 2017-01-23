
from .store.storable import *
from .store.generic import *
from .store.hdf5 import *

# Scaler
from inferencemap.spatial.scaler import Scaler
hdf5_storable(default_storable(Scaler), agnostic=True)

# Gas
from inferencemap.spatial.gas import Gas
hdf5_storable(default_storable(Gas), agnostic=True)

# ArrayGraph
from inferencemap.spatial.graph.array import ArrayGraph
hdf5_storable(default_storable(ArrayGraph), agnostic=True)

from inferencemap.spatial.dichotomy import Dichotomy, ConnectedDichotomy
dichotomy_exposes = ['base_edge', 'min_depth', 'max_depth', 'origin', 'lower_bound', 'upper_bound', \
	'min_count', 'max_count', 'subset', 'cell']
hdf5_storable(kwarg_storable(Dichotomy, dichotomy_exposes), agnostic=True)
dichotomy_graph_exposes = dichotomy_exposes + ['adjacency']
hdf5_storable(kwarg_storable(ConnectedDichotomy, exposes=dichotomy_graph_exposes), agnostic=True)

from inferencemap.tesselation import CellStats, Delaunay, Voronoi, RegularMesh, KDTreeMesh, KMeansMesh, \
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

