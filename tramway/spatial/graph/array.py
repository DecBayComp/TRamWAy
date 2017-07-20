# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""
.. =============================================
.. Alternative implementation for :class:`Graph`
.. =============================================

"""

from ..graph import *
import numpy as np
import scipy.sparse as sp
from collections import deque
import itertools
import time


class ArrayGraph(Graph):
	"""With DictGraph backend for Gas::

		%run glycine_receptor.py
		Elapsed:  mean: 40106 ms  std: 5819 ms

	With ArrayGraph backend::

		%run glycine_receptor.py
		Elapsed:  mean: 2937 ms  std: 304 ms
		ratio: 13.6554

	"""
	__slots__ = Graph.__slots__ + ['node_count_max', 'edge_count_max', '_node_counter', \
		'_edge_counter', '_free_nodes', '_free_edges', 'nodes', 'edges', 'adjacency', \
		'_fast_node']

	def __init__(self, node_defaults, edge_defaults, node_count=None, edge_count=None):
		Graph.__init__(self, node_defaults, edge_defaults)
		if node_count:
			if not edge_count:
				edge_count = node_count * 4
		elif edge_count:
			node_count = edge_count / 4
		else:
			node_count = 10000
			edge_count = node_count * 4
		self.node_count_max = node_count
		self.edge_count_max = edge_count
		self._node_counter = 0
		self._edge_counter = 0
		self._free_nodes = deque()
		self._free_edges = deque()
		self.nodes = {}
		self.edges = {}
		self.adjacency = sp.lil_matrix((self.node_count_max, self.node_count_max), \
			dtype=int)
		self._fast_node = {}

	def connect(self, n1, n2, **kwargs):
		if not self.hasNode(n1):
			raise MissingNodeError(n1)
		if not self.hasNode(n2):
			raise MissingNodeError(n2)
		if not self.edges:
			self._initEdges(**kwargs)
		e = self.adjacency[n1, n2] - 1
		if e < 0: # e is 0 does not work as expected because of types
			if self._free_edges:
				e = self._free_edges.pop()
			else:
				e = self._edge_counter
				self._edge_counter += 1
				if self.edge_count_max < self._edge_counter: # actual index is e - 1
					self._resizeEdges()
			self.adjacency[n1, n2] = e + 1
			self.adjacency[n2, n1] = e + 1
		for attr, default in self.edge_defaults.items():
			self.edges[attr][e] = kwargs.get(attr, default)

	def unsafeDisconnect(self, n1, n2):
		edge = self.adjacency[n1, n2] - 1
		self.adjacency[n1, n2] = 0
		self.adjacency[n2, n1] = 0
		self._free_edges.append(edge)
		
	def disconnect(self, n1, n2, edge=None):
		if not isinstance(n2, list):
			n2 = [n2]
		if n2:
			#print(('disconnect',(n1, type(n1)),(n2, type(n2))))
			if not self.hasNode(n1):
				raise MissingNodeError(n1)
			for n in n2:
				if self.hasNode(n):
					self.unsafeDisconnect(n1, n)
				else:	raise MissingNodeError(n)

	def findEdge(self, n1, n2):
		e = self.adjacency[n1, n2]
		if e:	return e - 1
		else:	return None

	def getNodeAttr(self, n, attr):
		#print(('getNodeAttr', n, attr))
		if not self.hasNode(n):
			raise MissingNodeError(n)
		try:
			return self.nodes[attr][n]
		except KeyError:
			raise NodeAttributeError(attr)

	def setNodeAttr(self, n, **kwargs):
		if not self.hasNode(n):
			raise MissingNodeError(n)
		for attr, val in kwargs.items():
			if attr in self.nodes:
				self.nodes[attr][n] = val
				if attr in self._fast_node:
					self._fast_node[attr][n] = np.dot(val, val)
			else: raise NodeAttributeError(attr)

	def getEdgeAttr(self, e, attr):
		if self._edgeMisses(e):
			raise MissingEdgeError(e)
		try:
			return self.edges[attr][e]
		except KeyError:
			raise EdgeAttributeError(attr)

	def setEdgeAttr(self, e, **kwargs):
		if self._edgeMisses(e):
			raise MissingEdgeError(e)
		for attr, val in kwargs.items():
			if attr in self.edges:
				self.edges[attr][e] = val
			else: raise EdgeAttributeError(attr)

	def iterEdges(self):
		"""Potentially inefficient."""
		return [e - 1 for e in set(itertools.chain(self.adjacency.data))]

	def iterEdgesFrom(self, n):
		if not self.hasNode(n):
			raise MissingNodeError(n)
		return zip([e - 1 for e in self.adjacency.data[n]], self.adjacency.rows[n])

	def iterNeighbors(self, n):
		if not self.hasNode(n):
			raise MissingNodeError(n)
		return self.adjacency.rows[n]

	def hasNode(self, n):
		return n < self._node_counter and n not in self._free_nodes

	def _edgeMisses(self, e):
		# _edge_counter is incremented before assignement, i.e. edges[*] has size at least
		# _edge_counter+1
		return self._edge_counter <= e or e in self._free_edges

	def addNode(self, **kwargs):
		if not self.nodes:
			self._initNodes(**kwargs)
		if self._free_nodes:
			n = self._free_nodes.pop()
		else:
			if self._node_counter is self.node_count_max:
				self._resizeNodes()
			n = self._node_counter
			self._node_counter += 1
		for attr, default in self.node_defaults.items():
			self.nodes[attr][n] = kwargs.get(attr, default)
		for attr in self._fast_node:
			w = self.nodes[attr][n]
			self._fast_node[attr][n] = np.dot(w, w)
		return n

	def delNode(self, n):
		#print(('delNode', n, type(n)))
		if self.hasNode(n):
			neighbors = list(self.iterNeighbors(n))
			if neighbors:
				self.disconnect(n, neighbors)
			self._free_nodes.append(n)
		else:	raise MissingNodeError(n)

	@property
	def size(self):
		return self._node_counter - len(self._free_nodes)

	def iterNodes(self):
		"""Avoid by all means!"""
		free = set(self._free_nodes)
		return [ n for n in range(0, self._node_counter) if n not in free ]

	def _initNodes(self, **kwargs):
		for attr, default in self.node_defaults.items():
			if default is None:
				if attr in kwargs:
					val = kwargs[attr]
				else: raise NodeAttributeError(attr)
			else:	val = default
			if isinstance(val, np.ndarray):
				dim = val.size
				typ = val.dtype
			else:
				dim = 1
				typ = type(val)
			self.nodes[attr] = np.zeros((self.node_count_max, dim), dtype=typ)

	def _initEdges(self, **kwargs):
		for attr, default in self.edge_defaults.items():
			if default is None:
				if attr in kwargs:
					val = kwargs[attr]
				else: raise NodeAttributeError(attr)
			else:	val = default
			if isinstance(val, np.ndarray):
				dim = val.size
				typ = val.dtype
			else:
				dim = 1
				typ = type(val)
			self.edges[attr] = np.zeros((self.edge_count_max, dim), dtype=typ)

	def _resizeNodes(self):
		raise NotImplementedError

	def _resizeEdges(self):
		raise NotImplementedError

	def export(self, sparse='csr'):
		#t = time.time()
		I = np.ones(self.node_count_max, dtype=bool)
		I[self._node_counter:] = 0
		I[self._free_nodes] = 0
		V = {}
		for attr in self.nodes:
			V[attr] = self.nodes[attr][I]
		#print('{:.0f}'.format((time.time() - t) * 1e6))
		#t = time.time()
		#Iadj, = np.nonzero(self.adjacency.rows)
		#V = {}
		#for attr in self.nodes:
		#	V[attr] = self.nodes[attr][Iadj]
		#print('{:.0f}'.format((time.time() - t) * 1e6))
		#assert np.all(I[Iadj]) and Iadj.size == np.sum(I)
		A = self.adjacency[I].tocsc()[:,I]
		if sparse is 'csc':
			A = A.tocsc()
		elif sparse is 'csr':
			A = A.tocsr()
		elif sparse is 'bsr':
			A = A.tobsr()
		elif sparse is 'coo':
			A = A.tocoo()
		else:
			raise NotImplementedError
		J = np.ones(self.edge_count_max, dtype=bool)
		J[self._edge_counter:] = 0
		J[self._free_edges] = 0
		E = {}
		for attr in self.edges:
			E[attr] = self.edges[attr][J]
		J, = np.nonzero(J)
		K = np.zeros(self._edge_counter + 1, dtype=int) # could be A.shape[0] * np.ones to
		# detect edge coding errors, but it seems there is none
		K[J] = np.arange(0, J.size)
		A.data = K[A.data - 1]
		#print((A.data.min(), A.data.max()))
		return (A, V, E)

	def squareDistance(self, attr, eta, eta2=None):
		if eta2 is None:
			eta2 = np.dot(eta, eta)
		if attr not in self._fast_node:
			self._fast_node[attr] = np.zeros(self.nodes[attr].shape[0], \
				dtype=self.nodes[attr].dtype)
			w = self.nodes[attr][:self._node_counter]
			self._fast_node[attr][:self._node_counter] = np.sum(w * w, axis=1)
		d = self._fast_node[attr][:self._node_counter] + eta2 - 2.0 * \
			np.dot(self.nodes[attr][:self._node_counter], eta)
		d[self._free_nodes] = float('nan')
		return (d, lambda i: i)

