
from .base import *
from inferencemap.spatial.scaler import *
from math import *
import numpy as np
import pandas as pd
from threading import Lock
import itertools
import scipy.sparse as sparse


def stack(v0, n1):
	'''key (suitable for using in a dict) for faces that stack along the same orthogonal direction
	with the lower corner (closest to the origin) in common.'''
	return (bvhash(v0[np.logical_not(n1)]), bvhash(n1))
def bvhash(v):
	'''hash for NumPy binary vectors.'''
	return np.sum(np.arange(0, v.size)[v == 1])


class KDTreeMesh(Voronoi):
	"""k-dimensional tree (quad tree in 2D) based tesselation."""
	def __init__(self, scaler=None, lower_bound=None, upper_bound=None, min_distance=None, \
		min_probability=None, max_probability=None, **kwargs):
		Voronoi.__init__(self, Scaler())
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound
		self._min_distance = min_distance
		self.min_probability = min_probability
		if min_probability and not max_probability:
			max_probability = 10.0 * min_probability
		self.max_probability = max_probability
		self.max_level = None

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
			self.origin = 0.5 * (lower + upper - self.width)
			self.max_level = int(floor(self.width / self._min_distance)) - 1
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
			#return np.logical_and(np.dot(face_normal, vertex) == 0, \
			#	np.dot(np.logical_xor(face_vertex, vertex), \
			#		1 - face_normal) == 1)
		self.exterior = np.concatenate([ onface(self.unit_hypercube, v, n) \
			for v, n in zip(self.face_vertex, self.face_normal) ], axis=1)
		def facing(v1, n1, v2, n2):
			return np.all(n1 == n2) and \
				np.all(v1[np.logical_not(n1)], v2[np.logical_not(n2)])
		self.facing_i = np.array([ [ [ i for i, w in enumerate(self.unit_hypercube) \
				if i != k and np.all(v[np.logical_not(n)] == w[np.logical_not(n)])][0]\
			for n in self.face_normal ] for k, v in enumerate(self.unit_hypercube) ])
		self.scale = 2.0 ** np.arange(0, -self.max_level-1, -1)
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
		self._cell_centers = origin + self.width * self.scale[self.level[:, np.newaxis]]
		adjacency = np.vstack([ self.adjacency[e] for e in range(self.edge_counter) ]).T
		self._cell_adjacency = sparse.csr_matrix((np.ones(self.edge_counter, dtype=int), \
			(adjacency[0], adjacency[1])), shape=(self.cell_counter, self.cell_counter))
		n = origin.shape[0]
		self._cell_vertices = []
		self._ridge_vertices = []
		for i, v1 in enumerate(self.unit_hypercube):
			self._cell_vertices.append(origin + self.width * np.float_(v1) * self.scale[self.level[:, np.newaxis]])
			for j, v2 in enumerate(self.unit_hypercube[i+1:]):
				if np.sum(np.logical_xor(v1, v2)) == 1: # neighbors in the voronoi
					self._ridge_vertices.append(np.vstack(\
						np.hstack(np.arange(i * n, (i+1) * n), \
							np.arange(j * n, (j+1) * n))))
		self._cell_vertices = np.concatenate(self._cell_vertices, axis=0)
		self._ridge_vertices = np.concatenate(self._ridge_vertices, axis=0)
		#self._postprocess()


	def mergeFaces(self, face1, face2, axis):
		dims = np.logical_not(axis)
		print(('faces:', face1, face2, dims))
		cell1 = [ (c, *self.cell[c]) for c in face1 ]
		cell2 = [ (c, *self.cell[c]) for c in face2 ]
		ok1 = np.ones(len(cell1), dtype=bool)
		ok2 = np.ones(len(cell2), dtype=bool)
		for i in range(len(cell1)):
			if ok1[i]:
				c1, o1, l1 = cell1[i]
				o1 = o1[dims]
				j = 0
				while j < ok2.size and ok2[j] and not np.all(np.abs(o1 - cell2[j][1][dims]) < np.spacing(1)):
					j += 1
				try:
					c2, o2, l2 = cell2[j]
				except IndexError as e:
					print((o1, [ (o2[dims] ,np.abs(o1 - o2[dims]) < np.spacing(1)) for _, o2, _ in cell2 ]))
					print(np.array([ o for o, _ in self.cell.values() ]))
					raise e
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
		if ok:
			# split and check that every subarea has at least min_count points
			level += 1
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
			print(('ok:', ok, level, points.shape[0], counts))
		else:
			print(('ok:', ok, level, points.shape[0]))
		if ok:
			interior_cells = dict()
			exterior_cells = dict()
			# split
			for i, step in enumerate(self.unit_hypercube):
				cells = self.split(ss_refs[i], lower[i], level)#path + [i])
				print(('cells:', cells))
				# factorize interior/exterior
				for c, cs in enumerate(cells):
					if self.exterior[i, c]:
						Cs = exterior_cells.get(c, [])
						Cs.append(cs)
						exterior_cells[c] = Cs
					else:
						j = stack(step, self.face_normal[c])
						print(('j:', j, level, origin, step, self.face_normal[c], interior_cells.get(j,None)))
						if j in interior_cells:
							self.mergeFaces(interior_cells[j], cs, \
								self.face_normal[c])
						else:
							interior_cells[j] = cs
			for c in exterior_cells:
				exterior_cells[c] = list(itertools.chain(*exterior_cells[c]))
			print(('exterior_cells:', exterior_cells))
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
