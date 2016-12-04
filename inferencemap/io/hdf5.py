
from .store.storable import *
from .store.generic import *
from .store.hdf5 import *

# Scaler
from inferencemap.spatial.scaler import Scaler
hdf5_storable(storable_with_new(Scaler), agnostic=True)

# Gas
from inferencemap.spatial.gas import Gas
hdf5_storable(storable_with_new(Gas), agnostic=True)

# ArrayGraph
from inferencemap.spatial.graph.array import ArrayGraph
hdf5_storable(storable_with_new(ArrayGraph), agnostic=True)

from inferencemap.tesselation import CellStats, Delaunay, Voronoi, RegularMesh, KMeansMesh, GasMesh
# CellStats
hdf5_storable(storable_with_new(CellStats), agnostic=True)
# Delaunay
delaunay_exposes = ['scaler', 'cell_centers', 'cell_adjacency', 'cell_label', 'adjacency_label']
hdf5_storable(storable_with_new(Delaunay, exposes=delaunay_exposes), agnostic=True)
# Voronoi
voronoi_exposes = delaunay_exposes + ['cell_vertices', 'ridge_vertices']
hdf5_storable(storable_with_new(Voronoi, exposes=voronoi_exposes), agnostic=True)
# RegularMesh
regular_mesh_exposes = voronoi_exposes + ['lower_bound', 'upper_bound', 'count_per_dim', \
	'min_probability', 'max_probability', 'avg_probability']
hdf5_storable(storable_with_new(RegularMesh, exposes=regular_mesh_exposes), agnostic=True)
# KMeansMesh
kmeans_mesh_exposes = voronoi_exposes + ['avg_probability']
hdf5_storable(storable_with_new(KMeansMesh, exposes=kmeans_mesh_exposes), agnostic=True)
# GasMesh
gas_mesh_exposes = voronoi_exposes + ['gas', '_min_distance', '_max_distance']
hdf5_storable(storable_with_new(GasMesh, exposes=gas_mesh_exposes), agnostic=True)

