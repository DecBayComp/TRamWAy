# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import *
from tramway.tesselation import *
import numpy as np
import pandas as pd
import copy


class TimeLattice(Lazy):
	"""Proxy `Tesselation` for time lattice expansion.
	"""
	__slots__ = ['spatial_mesh', 'time_lattice', 'time_edge', \
		'_cell_adjacency', '_cell_label', '_adjacency_label', \
		'_cell_centers']
	__lazy__ = ['cell_adjacency', 'cell_label', 'adjacency_label', \
		'cell_centers']

	def __init__(self, mesh=None, lattice=None, label=None):
		self.spatial_mesh = mesh
		self.time_lattice = lattice
		self.time_edge = label
		self._cell_adjacency = None
		self._cell_label = None
		self._adjacency_label = None
		self._cell_centers = None

	def tesselate(self, points, **kwargs):
		self.spatial_mesh.tesselate(points, **kwargs)

	def cell_index(self, points, *args, **kwargs):
		time_col = kwargs.pop('time_col', 't')
		exclude = kwargs.pop('exclude_cells_by_location_count', None)
		spatial_index = self.spatial_mesh.cell_index(points, *args, **kwargs)
		ncells = self.spatial_mesh.cell_adjacency.shape[0]
		nsegments = self.time_lattice.shape[0]
		if isstructured(points):
			ts = points[time_col]
			if isinstance(ts, (pd.Series, pd.DataFrame)):
				ts = ts.values
		else:
			ts = points[:,time_col]
		if exclude:
			location_count = np.zeros((ncells, nsegments), dtype=int)
		ps, cs = [], []
		for t in range(nsegments):
			t0, t1 = self.time_lattice[t]
			segment = np.logical_and(t0 <= ts, ts < t1)
			pts, = np.nonzero(segment)
			if isinstance(spatial_index, np.ndarray):
				ids = spatial_index[segment]
			else:
				raise NotImplementedError
			if exclude:
				vs, count = np.unique(ids, return_counts=True)
				location_count[vs, t] = count
			if ids.size:
				ps.append(pts)
				cs.append(t * ncells + ids)
		ps = np.concatenate(ps)
		cs = np.concatenate(cs)
		if exclude:
			i, t = exclude(location_count).nonzero()
			ok = np.ones(cs.size, dtype=bool)
			for c in t * ncells + i:
				ok[cs == c] = False
			ps = ps[ok]
			cs = cs[ok]
		return (ps, cs)

	# scaler property
	@property
	def scaler(self):
		return self.spatial_mesh.scaler

	@scaler.setter
	def scaler(self, s):
		self.spatial_mesh.scaler = s

	# descriptors
	def descriptors(self, *args, **kwargs):
		return self.spatial_mesh.descriptors(*args, **kwargs)

	# cell_adjacency property
	@property
	def cell_adjacency(self):
		if self._cell_adjacency is None:
			nsegments = self.time_lattice.shape[0]
			if self.spatial_mesh.adjacency_label is None:
				A = sparse.triu(self.spatial_mesh.cell_adjacency, format='coo')
				ncells = A.shape[0]
				edge_max = int(A.data.max())
				if 1 < edge_max:
					raise ValueError('non boolean values in the adjacency matrix are no reference to labels')
				n_spatial_edges = A.data.size
				A = sparse.coo_matrix((np.tile(np.arange(n_spatial_edges), 2), \
						(np.r_[A.row, A.col], np.r_[A.col, A.row])), \
					shape=A.shape).tocsr()
				self._adjacency_label = np.ones(n_spatial_edges)
			else:
				A = self.spatial_mesh.adjacency.tocsr()
				edge_max = int(A.data.max())
				n_spatial_edges = int(self._adjacency_label.max())
			ncells = A.shape[0]
			active_cells, = np.where(0 < np.diff(A.indptr))
			try:
				past_edge, future_edge = self.time_edge
			except TypeError:
				past_edge = future_edge = self.time_edge
			edge_ptr = edge_max + 1
			past = sparse.coo_matrix( \
				(np.arange(edge_ptr, edge_ptr + active_cells.size), \
					(active_cells, active_cells)), \
				shape=(ncells, ncells))
			edge_ptr += active_cells.size
			future = sparse.coo_matrix( \
				(np.arange(edge_ptr, edge_ptr + active_cells.size), \
					(active_cells, active_cells)), \
				shape=(ncells, ncells))
			edge_ptr += active_cells.size
			blocks = [[A, future] + [None] * (nsegments - 2)]
			for k in range(1, nsegments - 1):
				blocks.append([None] * (k - 1) + [past, A, future] + \
					[None] * (nsegments - 2 - k))
			blocks.append([None] * (nsegments - 2) + [past, A])
			self._cell_adjacency = sparse.bmat(blocks, format='csr')
			if past_edge is None:
				past_edge = edge_max + 1
				edge_max += 1
			if future_edge is None:
				future_edge = edge_max + 1
				edge_max += 1
			dtype = self._adjacency_label.dtype
			self._adjacency_label = np.r_[self._adjacency_label, \
				np.full(active_cells.size, past_edge, dtype=dtype), \
				np.full(active_cells.size, future_edge, dtype=dtype)]
			self.time_edge = (past_edge, future_edge)
		return self._cell_adjacency

	@cell_adjacency.setter
	def cell_adjacency(self, matrix):
		self.__lazysetter__(matrix)

	# past/future properties
	@property
	def past_edge(self):
		return self.time_edge[0]

	@property
	def future_edge(self):
		return self.time_edge[1]

	# cell_label
	@property
	def cell_label(self):
		if self._cell_label is None:
			nsegments = self.time_lattice.shape[0]
			return np.tile(self.spatial_mesh.cell_label, nsegments)
		return self._cell_label

	@cell_label.setter
	def cell_label(self, label):
		self.__lazysetter__(label)

	# adjacency_label
	@property
	def adjacency_label(self):
		return self._adjacency_label

	@adjacency_label.setter
	def adjacency_label(self, label):
		self.__lazysetter__(label)

	## Delaunay properties and methods
	@property
	def cell_centers(self):
		if self._cell_centers is None:
			nsegments = self.time_lattice.shape[0]
			self._cell_centers = self.spatial_mesh.cell_centers
			ncells = self._cell_centers.shape[0]
			self._cell_centers = np.tile(self._cell_centers, (nsegments, 1))
			self._cell_centers = np.hstack((self._cell_centers, \
				np.repeat(np.mean(self.time_lattice, axis=1), ncells)[:,np.newaxis]))
		return self._cell_centers

	@cell_centers.setter
	def cell_centers(self, pts):
		self.__lazysetter__(pts)


	def split_frames(self, df, return_times=False):
		ncells = self.spatial_mesh.cell_adjacency.shape[0]
		nsegments = self.time_lattice.shape[0]
		try:
			# not tested yet
			segment, cell = np.divmod(df.index, ncells) # 1.13.0 <= numpy
		except AttributeError:
			try:
				segment = df.index // ncells
			except TypeError:
				print(df.index)
				raise
			cell = np.mod(df.index, ncells)
		ts = []
		for t in range(nsegments):
			xt = df[segment == t]
			xt.index = cell[segment == t]
			if return_times:
				ts.append((self.time_lattice[t], xt))
			else:
				ts.append(xt)
		return ts


def with_time_lattice(cells, frames, exclude_cells_by_location_count=None, **kwargs):
	dynamic_cells = copy.deepcopy(cells)
	dynamic_cells.tesselation = TimeLattice(cells.tesselation, frames)
	dynamic_cells.cell_index = dynamic_cells.tesselation.cell_index(cells.points, \
		exclude_cells_by_location_count=exclude_cells_by_location_count, **kwargs)
	return dynamic_cells

