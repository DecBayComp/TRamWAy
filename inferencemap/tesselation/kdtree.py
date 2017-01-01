
from .base import *
from inferencemap.spatial.scaler import *
from math import *
import numpy as np
import pandas as pd
from threading import Lock
import itertools
import scipy.sparse as sparse


def face_hash(v1, n1):
	'''key that identify a same face.'''
	return tuple(v1) + tuple(n1)


class KDTreeMesh(Voronoi):
	"""k-dimensional tree (quad tree in 2D) based tesselation."""
	def __init__(self, scaler=None, lower_bound=None, upper_bound=None, min_distance=None, \
		min_probability=None, max_probability=None, lower_levels=None, **kwargs):
		Voronoi.__init__(self, Scaler())
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound
		self._min_distance = min_distance
		self.min_probability = min_probability
		if min_probability and not max_probability:
			max_probability = 10.0 * min_probability
		self.max_probability = max_probability
		self.min_level = 0
		self.max_level = None
		self.lower_levels = lower_levels

	def cellIndex(self, points, knn=None, metric='chebyshev', **kwargs):
		if metric == 'chebyshev':
			# TODO: pass relevant kwargs to cdist
			D = cdist(np.asarray(self.scaler.scalePoint(points, inplace=False)), \
					self._cell_centers, metric) # , **kwargs
			dmax = self.width * self.scale[self.level[np.newaxis,:] + 1]
			I, J = np.nonzero(D <= dmax)
			if I[0] == 0 and I.size == points.shape[0] and I[-1] == points.shape[0] - 1:
				return J
			else:
				K = -np.ones(points.shape[0], dtype=J.dtype)
				K[I] = J
				return K
		else:
			return Delaunay.cellIndex(self, points, knn=knn, metric=metric, **kwargs)

	def tesselate(self, points, **kwargs):
		init = self.scaler.init
		points = self._preprocess(points)
		if self.lower_bound is None:
	 		self.lower_bound = points.min(axis=0)
		elif isinstance(points, pd.DataFrame) and not isinstance(self.lower_bound, pd.Series):
			self.lower_bound = pd.Series(self.lower_bound, index=points.columns)
		if self.upper_bound is None:
			self.upper_bound = points.max(axis=0)
		elif isinstance(points, pd.DataFrame) and not isinstance(self.upper_bound, pd.Series):
			self.upper_bound = pd.Series(self.upper_bound, index=points.columns)
		if init and self._min_distance:
			self._min_distance = self.scaler.scaleDistance(self._min_distance)
		if self.max_level is None:
			lower = np.asarray(self.lower_bound)
			upper = np.asarray(self.upper_bound)
			self.width = np.max(upper - lower)
			self.max_level = int(ceil((log(self.width) - log(self._min_distance)) / log(2.0)))
			if self.lower_levels:
				self.min_level = self.max_level - self.lower_levels
			del self.lower_levels
			center = 0.5 * (lower + upper)
			self.width = self._min_distance * 2.0 ** self.max_level
			self.origin = center - 0.5 * self.width
		self.min_count = int(round(self.min_probability * points.shape[0]))
		self.max_count = int(round(self.max_probability * points.shape[0]))
		self.dim = points.shape[1]
		i = [(0, 1)] * self.dim
		g = np.meshgrid(*i, indexing='ij')
		for c in g:
			c.shape = (c.size, 1)
		self.unit_hypercube = np.concatenate(g, axis=1) 
		eye = np.eye(self.dim, dtype=int)
		self.face_normal = np.vstack((eye[::-1], eye))
		self.face_vertex = np.vstack((np.zeros_like(eye), eye))
		def onface(vertex, face_vertex, face_normal):
			return np.sum(face_normal * (face_vertex - vertex), axis=1, keepdims=True) == 0
		self.exterior = np.concatenate([ onface(self.unit_hypercube, v, n) \
			for v, n in zip(self.face_vertex, self.face_normal) ], axis=1)
		self.scale = 2.0 ** np.arange(0, -self.max_level-2, -1)
		self.subset = {0: np.asarray(points)}
		self.subset_counter = 1
		self.subset_lock = Lock()
		self.cell = dict()
		self.cell_counter = 0
		self.cell_lock = Lock()
		self.adjacency = dict()
		self.edge_counter = 0
		self.adjacency_lock = Lock()

		self.split(0, self.origin, 0)
		
		origin, level = zip(*[ self.cell[c] for c in range(self.cell_counter) ])
		origin = np.vstack(origin)
		self.level = np.array(level)
		self._cell_centers = origin + self.width * self.scale[self.level[:, np.newaxis] + 1]
		adjacency = np.vstack([ self.adjacency[e] for e in range(self.edge_counter) \
			if e in self.adjacency ]).T
		self._cell_adjacency = sparse.csr_matrix((np.ones(adjacency.shape[1], dtype=int), \
			(adjacency[0], adjacency[1])), shape=(self.cell_counter, self.cell_counter))

		# quick and dirty Voronoi construction: vertices are introduced as many times as they
		# appear in a ridge, and as many ridges are introduced as four times the number of 
		# centers. In a second step, duplicate vertices are removed.
		def unique_rows(data, *args, **kwargs):
			uniq = np.unique(data.view(data.dtype.descr * data.shape[1]), *args, **kwargs)
			if isinstance(uniq, tuple):
				return (uniq[0].view(data.dtype).reshape(-1, data.shape[1]),) + tuple(uniq[1:])
			else:
				return uniq.view(data.dtype).reshape(-1, data.shape[1])
		n = origin.shape[0]
		self._cell_vertices = []
		self._ridge_vertices = []
		for i, v1 in enumerate(self.unit_hypercube):
			self._cell_vertices.append(origin + \
				self.width * np.float_(v1) * self.scale[self.level[:, np.newaxis]])
			for jj, v2 in enumerate(self.unit_hypercube[i+1:]):
				if np.sum(v1 != v2) == 1: # neighbors in the voronoi
					j = i + 1 + jj
					self._ridge_vertices.append(np.vstack(\
						np.hstack((np.arange(i * n, (i+1) * n)[:,np.newaxis], \
							np.arange(j * n, (j+1) * n)[:,np.newaxis]))))
		self._cell_vertices, I = unique_rows(np.concatenate(self._cell_vertices, axis=0), \
			return_inverse=True)
		self._ridge_vertices = I[np.concatenate(self._ridge_vertices, axis=0)]
		#self._postprocess()


	def mergeFaces(self, face1, face2, axis):
		dims = np.logical_not(axis)
		cell1 = [ (c,) + self.cell[c] for c in face1 ]
		cell2 = [ (c,) + self.cell[c] for c in face2 ]
		ok1 = np.ones(len(cell1), dtype=bool)
		ok2 = np.ones(len(cell2), dtype=bool)
		for i in range(len(cell1)):
			if ok1[i]:
				c1, o1, l1 = cell1[i]
				o1 = o1[dims]
				j = 0
				while j < ok2.size and ok2[j] and not np.all(np.abs(o1 - cell2[j][1][dims]) < np.spacing(1)):
					j += 1
				c2, o2, l2 = cell2[j]
				o2 = o2[dims]
				dk = 2 ** ((self.dim - 1) * abs(l1 - l2))
				self.adjacency_lock.acquire()
				k = self.edge_counter
				self.edge_counter += dk
				self.adjacency_lock.release()
				self.adjacency[k] = (c1, c2)
				ok1[i] = False
				ok2[j] = False
				if l1 < l2:
					e1 = o1 + self.width * self.scale[l1]
					jj = 0
					while 1 < dk and jj < ok2.size and ok2[jj]:
						c2, o2, ll2 = cell2[jj]
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
					e2 = o2 + self.width * self.scale[l2]
					ii = 0
					while 1 < dk and ii < ok1.size and ok1[ii]:
						c1, o1, ll1 = cell1[ii]
						o1 = o1[dims]
						if l1 == ll1 and \
							np.all(np.logical_and(o2 <= o1, \
								o1 < e2)):
							k += 1
							self.adjacency[k] = (c1, c2)
							ok1[ii] = False
							dk -= 1
						ii += 1

	def split(self, ss_ref, origin, level):
		#level = len(path)
		points = self.subset[ss_ref]
		ok = level < self.max_level and self.min_count < points.shape[0]
		level += 1
		if ok or level <= self.min_level:
			# split and check that every subarea has at least min_count points
			ss_refs = dict()
			lower = dict()
			for i, step in enumerate(self.unit_hypercube): # could be parallelized
				lower[i] = origin + self.width * np.float_(step) * self.scale[level] # new origin
				upper = lower[i] + self.width * self.scale[level] # opposite hypercube vertex
				self.subset_lock.acquire()
				r = self.subset_counter
				self.subset_counter += 1
				self.subset_lock.release()
				self.subset[r] = points[np.all(np.logical_and(lower[i] < points, \
					points < upper), axis=1)]
				ss_refs[i] = r # dict for future parallelization
			del self.subset[ss_ref] # save memory
			counts = [ self.subset[r].shape[0] for r in ss_refs.values() ]
			ok = any([ self.max_count < c for c in counts ]) or \
				all([ self.min_count < c for c in counts ])
		if ok or level <= self.min_level:
			interior_cells = dict()
			exterior_cells = dict()
			# split
			for i, step in enumerate(self.unit_hypercube):
				cells = self.split(ss_refs[i], lower[i], level)#path + [i])
				# factorize interior/exterior
				for c, cs in enumerate(cells):
					if self.exterior[i, c]:
						Cs = exterior_cells.get(c, [])
						Cs.append(cs)
						exterior_cells[c] = Cs
					else:
						j = face_hash(step + self.face_vertex[c], \
							self.face_normal[c])
						if j in interior_cells:
							self.mergeFaces(interior_cells[j], cs, \
								self.face_normal[c])
						else:
							interior_cells[j] = cs
			for c in exterior_cells:
				exterior_cells[c] = list(itertools.chain(*exterior_cells[c]))
			return [ exterior_cells[c] for c in range(2 * self.dim) ]
		else:
			level -= 1
			#for r in ss_refs.values():
			#	del subset[r]
			self.cell_lock.acquire()
			i = self.cell_counter
			self.cell_counter += 1
			self.cell_lock.release()
			self.cell[i] = (origin, level)#path)
			return [ [i] for j in range(2 * self.dim) ]

