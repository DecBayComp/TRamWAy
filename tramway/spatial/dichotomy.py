# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import *
import numpy as np
from threading import Lock
import itertools


class Dichotomy(object):
	def __init__(self, points=None, min_edge=None, base_edge=None, \
		min_depth=None, max_depth=50, max_level=None, \
		min_count=1, max_count=None, \
		origin=None, center=None, lower_bound=None, upper_bound=None, \
		subset=None, cell=None):
		self.subset_lock = Lock()
		self.cell_lock = Lock()
		if isinstance(points, Dichotomy): # useless
			obj = points
			self.base_edge = obj.base_edge
			self.min_depth = obj.min_depth
			self.max_depth = obj.max_depth
			self.origin = obj.origin.copy()
			self.lower_bound = obj.lower_bound.copy()
			self.upper_bound = obj.upper_bound.copy()
			self.min_count = obj.min_count
			self.max_count = obj.max_count
			self.unit_hypercube = obj.unit_hypercube.copy()
			self.reference_length = obj.reference_length.copy()
			self.subset = obj.subset.copy()
			self.subset_counter = obj.subset_counter
			self.cell = obj.cell.copy()
			self.cell_counter = obj.cell_counter
			#Dichotomy.__init__(self, \
			#	min_depth=points.min_depth, max_depth=points.max_depth, \
			#	min_count=points.min_count, max_count=points.max_count, \
			#	origin=points.origin.copy(), \
			#	lower_bound=points.lower_bound.copy(), upper_bound=points.upper_bound.copy(), \
			#	subset=points.subset.copy(), cell=points.cell.copy())
			return
		self.base_edge = base_edge
		self.min_depth = min_depth
		self.max_depth = max_depth
		self.origin = origin
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound
		self.min_count = min_count
		self.max_count = max_count
		try:
			if self.max_count is None:
				self.max_count = points.shape[0]
			if self.lower_bound is None:
				self.lower_bound = points.min(axis=0)
			if self.upper_bound is None:
				self.upper_bound = points.max(axis=0)
		except AttributeError:
			raise AttributeError('missing `points` input argument')
		if center is None:
			center = (self.lower_bound + self.upper_bound) / 2.0
		# define `max_edge`; `max_edge` is level-1 edge, not level 0, hence +1 in `max_depth`
		max_edge = np.max(np.maximum(self.upper_bound - center, center - self.lower_bound))
		if self.origin is None: # `center` cannot be `None`
			self.origin = center - max_edge
		if self.base_edge is None: # `max_edge` is defined
			if min_edge:
				self.max_depth = int(ceil(log(max_edge / min_edge, 2))) + 1
			self.base_edge = max_edge * 2.0 ** (1 - self.max_depth)
		else:
			if min_edge:
				raise AttributeError('`min_edge` ignored') # should be a warning instead
			self.max_depth = int(ceil(log(max_edge / self.base_edge, 2))) + 1
		if self.min_depth is None:
			if max_level is None:
				self.min_depth = 0
			else:
				self.min_depth = self.max_depth - max_level
		dim = self.origin.size
		grid = [(0, 1)] * dim
		grid = np.meshgrid(*grid, indexing='ij')
		self.unit_hypercube = np.stack([ g.flatten() for g in grid ], axis=-1)
		# `min_edge` is defined
		self.reference_length = self.base_edge * 2.0 ** np.arange(self.max_depth, -2, -1)
		if subset is None:
			if points is None:
				self.subset = {}
				self.subset_counter = 0
			else:
				self.subset = {0: (np.arange(points.shape[0]), points)}
				self.subset_counter = 1
		else:
			self.subset = subset
			if subset:
				self.subset_counter = max(subset.keys()) + 1
			else:
				self.subset_counter = 0
		if cell is None:
			self.cell = dict()
			self.cell_counter = 0
		else:
			self.cell = cell
			if cell:
				self.cell_counter = max(cell.keys()) + 1
			else:
				self.cell_counter = 0
		# already done at the beginning of __init__
		#self.subset_lock = Lock()
		#self.cell_lock = Lock()


	def split(self, points=None):
		if points is None:
			if self.subset_counter == 0:
				raise AttributeError('missing `points` input argument')
		else:
			self.subset = {0: (np.arange(points.shape[0]), points)}
			self.subset_counter = 1
			self.cell = dict()
			self.cell_counter = 0
		self._split(0, self.origin, 0)

	def _split(self, ss_ref, origin, depth):
		indices, points = self.subset[ss_ref]
		ok = depth < self.max_depth and self.min_count < points.shape[0]
		depth += 1
		ok = ok or depth < self.min_depth
		if ok:
			# split and check that every subarea has at least min_count points
			ss_refs = dict()
			lower = dict()
			for i, step in enumerate(self.unit_hypercube): # could be parallelized
				lower[i] = origin + np.float_(step) * self.reference_length[depth] # new origin
				upper = lower[i] + self.reference_length[depth] # opposite hypercube vertex
				self.subset_lock.acquire()
				r = self.subset_counter
				self.subset_counter += 1
				self.subset_lock.release()
				subset = np.all(np.logical_and(lower[i] < points, \
					points < upper), axis=1)
				self.subset[r] = (indices[subset], points[subset])
				ss_refs[i] = r # dict for future parallelization
			if self.min_depth <= depth:
				counts = [ self.subset[r][0].size for r in ss_refs.values() ]
				ok = any([ self.max_count < c for c in counts ]) or \
					all([ self.min_count < c for c in counts ])
			if ok:
				del self.subset[ss_ref] # save memory
			else:
				for r in ss_refs.values():
					del self.subset[r]
		if ok:
			return self.do_split(ss_refs, lower, depth)
		else:
			depth -= 1
			self.cell_lock.acquire()
			i = self.cell_counter
			self.cell_counter += 1
			self.cell_lock.release()
			self.cell[i] = (origin, depth, ss_ref)
			return self.dont_split(i, ss_ref, origin, depth)

	def do_split(self, ss_ref, origin, depth):
		for i in ss_ref:
			self._split(ss_ref[i], origin[i], depth)

	def dont_split(self, cell_ref, ss_ref, origin, depth):
		pass



def _face_hash(v1, n1):
	'''key that identify a same face.'''
	return tuple(v1) + tuple(n1)


class ConnectedDichotomy(Dichotomy):
	def __init__(self, *args, **kwargs):
		adjacency = kwargs.pop('adjacency', None)
		Dichotomy.__init__(self, *args, **kwargs)
		self.adjacency_lock = Lock()
		if len(args) == 1 and not kwargs and isinstance(args[0], DichotomyGraph): # useless
			obj = args[0]
			self.face_normal = obj.face_normal.copy()
			self.face_vertex = obj.face_vertex.copy()
			self.exterior = obj.exterior.copy()
			self.adjacency = obj.adjacency.copy()
			self.edge_counter = self.edge_counter
		dim = self.unit_hypercube.shape[1]
		eye = np.eye(dim, dtype=int)
		self.face_normal = np.vstack((eye[::-1], eye))
		self.face_vertex = np.vstack((np.zeros_like(eye), eye))
		def onface(vertex, face_vertex, face_normal):
			return np.sum(face_normal * (face_vertex - vertex), axis=1, keepdims=True) == 0
		self.exterior = np.concatenate([ onface(self.unit_hypercube, v, n) \
			for v, n in zip(self.face_vertex, self.face_normal) ], axis=1)
		if adjacency is None:
			self.adjacency = dict()
			self.edge_counter = 0
		else:
			self.adjacency = adjacency
			self.edge_counter = max(adjacency.keys()) + 1
		# already done earlier (above)
		#self.adjacency_lock = Lock()

	def merge_Faces(self, face1, face2, axis):
		dim = self.unit_hypercube.shape[1]
		dims = np.logical_not(axis)
		cell1 = [ (c,) + self.cell[c] for c in face1 ]
		cell2 = [ (c,) + self.cell[c] for c in face2 ]
		ok1 = np.ones(len(cell1), dtype=bool)
		ok2 = np.ones(len(cell2), dtype=bool)
		for i in range(len(cell1)):
			if ok1[i]:
				c1, o1, l1, _ = cell1[i]
				o1 = o1[dims]
				j = 0
				while j < ok2.size and ok2[j] and not np.all(np.abs(o1 - cell2[j][1][dims]) < np.spacing(1)):
					j += 1
				c2, o2, l2, _ = cell2[j]
				o2 = o2[dims]
				dk = 2 ** ((dim - 1) * abs(l1 - l2))
				self.adjacency_lock.acquire()
				k = self.edge_counter
				self.edge_counter += dk
				self.adjacency_lock.release()
				self.adjacency[k] = (c1, c2)
				ok1[i] = False
				ok2[j] = False
				if l1 < l2:
					e1 = o1 + self.reference_length[l1]
					jj = 0
					while 1 < dk and jj < ok2.size and ok2[jj]:
						c2, o2, ll2, _ = cell2[jj]
						o2 = o2[dims]
						if l2 == ll2 and \
							np.all(np.logical_and(o1 <= o2, \
								o2 < e1)):
							k += 1
							self.adjacency[k] = (c1, c2)
							ok2[jj] = False
							dk -= 1
						jj += 1
				elif l2 < l1:
					e2 = o2 + self.reference_length[l2]
					ii = 0
					while 1 < dk and ii < ok1.size and ok1[ii]:
						c1, o1, ll1, _ = cell1[ii]
						o1 = o1[dims]
						if l1 == ll1 and \
							np.all(np.logical_and(o2 <= o1, \
								o1 < e2)):
							k += 1
							self.adjacency[k] = (c1, c2)
							ok1[ii] = False
							dk -= 1
						ii += 1
		
	def do_split(self, ss_ref, origin, depth):
		interior_cells = dict()
		exterior_cells = dict()
		# split
		for i, step in enumerate(self.unit_hypercube):
			cells = self._split(ss_ref[i], origin[i], depth)
			# factorize interior/exterior
			for c, cs in enumerate(cells):
				if self.exterior[i, c]:
					Cs = exterior_cells.get(c, [])
					Cs.append(cs)
					exterior_cells[c] = Cs
				else:
					j = _face_hash(step + self.face_vertex[c], \
						self.face_normal[c])
					if j in interior_cells:
						self.merge_Faces(interior_cells[j], cs, \
							self.face_normal[c])
					else:
						interior_cells[j] = cs
		for c in exterior_cells:
			exterior_cells[c] = list(itertools.chain(*exterior_cells[c]))
		dim = self.unit_hypercube.shape[1]
		return [ exterior_cells[c] for c in range(2 * dim) ]

	def dont_split(self, cell_ref, ss_ref, origin, depth):
		dim = self.unit_hypercube.shape[1]
		return [ [cell_ref] for _ in range(2 * dim) ]


