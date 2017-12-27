# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""This module does not export any symbol. It implements the :class:`~tramway.io.store.storable.Storable` class for TRamWAy datatypes."""

from rwa.storable import *
from rwa.generic import *
from rwa.hdf5 import *

# Core datatypes
from tramway.core import Matrix, ArrayChain, Analyses
from tramway.inference.diffusivity import DV
hdf5_storable(namedtuple_storable(Matrix))
hdf5_storable(default_storable(ArrayChain))
hdf5_storable(default_storable(DV), agnostic=True)
hdf5_storable(default_storable(Analyses), agnostic=True)

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
# scaler should be first and cell_centers should be last
hdf5_storable(default_storable(Delaunay, exposes=delaunay_exposes), agnostic=True)
# Voronoi
voronoi_exposes = ['scaler', 'vertices', 'vertex_adjacency', 'cell_vertices'] + delaunay_exposes[1:]
hdf5_storable(default_storable(Voronoi, exposes=voronoi_exposes), agnostic=True)
# RegularMesh
regular_mesh_exposes = \
	['scaler', 'diagonal_adjacency'] + \
	voronoi_exposes[1:] + \
	['lower_bound', 'upper_bound', 'count_per_dim', \
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


from tramway.inference.base import Cell, Distributed, Maps
hdf5_storable(default_storable(Cell), agnostic=True)
hdf5_storable(default_storable(Distributed), agnostic=True)

def poke_maps(store, objname, self, container, visited=None, legacy=False):
	#print('poke_maps')
	sub_container = store.newContainer(objname, self, container)
	attrs = dict(self.__dict__) # dict
	#print(list(attrs.keys()))
	if legacy:
		# legacy format
		if callable(self.mode):
			store.poke('mode', '(callable)', sub_container)
			store.poke('result', self.maps, sub_container)
		else:
			store.poke('mode', self.mode, sub_container)
			store.poke(self.mode, self.maps, sub_container)
	else:
		#print("poke 'maps'")
		store.poke('maps', self.maps, sub_container, visited=visited)
		del attrs['maps']
	deprecated = {}
	for a in ('distributed_translocations','partition_file','tesselation_param','version'):
		deprecated[a] = attrs.pop(a, None)
	for a in attrs:
		if attrs[a] is not None:
			if attrs[a] or attrs[a] == 0:
				#print("poke '{}'".format(a))
				store.poke(a, attrs[a], sub_container, visited=visited)
	for a in deprecated:
		if deprecated[a]:
			warn('`{}` is deprecated'.format(a), DeprecationWarning)
			#print("poke '{}'".format(a))
			store.poke(a, deprecated[a], sub_container, visited=visited)

def peek_maps(store, container):
	#print('peek_maps')
	read = []
	mode = store.peek('mode', container)
	read.append('mode')
	try:
		maps = store.peek('maps', container)
		read.append('maps')
	except KeyError:
		# former standalone files
		if mode == '(callable)':
			maps = store.peek('result', container)
			read.append('result')
			mode = None
		else:
			maps = store.peek(mode, container)
			read.append(mode)
	maps = Maps(maps, mode=mode)
	for r in container:
		if r not in read:
			setattr(maps, r, store.peek(r, container))
	return maps

hdf5_storable(Storable(Maps, handlers=StorableHandler(poke=poke_maps, peek=peek_maps)), agnostic=True)

