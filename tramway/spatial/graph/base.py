# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import scipy.sparse as sparse
from .exception import *


class Graph(object):
	'''Abstract class for node-centered graph implementations. Here the concept of edge is limited to
	labels and retrieving the nodes that a given edge connects may be a very inefficient operation.'''
	__slots__ = ['node_defaults', 'edge_defaults']

	def __init__(self, node_defaults, edge_defaults):
		self.node_defaults = node_defaults
		self.edge_defaults = edge_defaults

	def connect(self, n1, n2, **kwargs):
		raise AbstractGraphError

	def disconnect(self, n1, n2, edge=None):
		raise AbstractGraphError

	def getNodeAttr(self, n, attr):
		raise AbstractGraphError
	def setNodeAttr(self, n, **kwargs):
		raise AbstractGraphError
	def getEdgeAttr(self, e, attr):
		raise AbstractGraphError
	def setEdgeAttr(self, e, **kwargs):
		raise AbstractGraphError
	@property
	def size(self):
		return len(list(self.iterNodes))
	def iterNodes(self):
		'''Returns an iterator over all the nodes of the graph. A copy of the returned
		view should be made if the structure of the graph had to be altered.'''
		raise AbstractGraphError
	def iterNeighbors(self, n):
		'''Returns an iterator over the neighbor nodes of a given node. A copy of the returned
		view should be made if the structure of the graph had to be altered.'''
		return [ n for _, n in self.iterEdgesFrom(n) ]
	def iterEdges(self):
		'''Returns an iterator over all the edges of the graph. A copy of the returned
		view should be made if the structure of the graph had to be altered.'''
		raise AbstractGraphError
	def iterEdgesFrom(self, n):
		'''Returns an iterator over the `(edge, neighbor_node)` index pairs from a given node. 
		A copy of the returned view should be made if the structure of the graph had to be altered.'''
		raise AbstractGraphError
	def hasNode(self, n):
		return n in self.iterNodes()

	def addNode(self, **kwargs):
		raise AbstractGraphError
	def delNode(self, n):
		raise AbstractGraphError
	def standsAlone(self, n):
		'''Returns `True` if node `n` has no neighbor.'''
		return not self.iterNeighbors(n)

	def findEdge(self, n1, n2):
		if self.hasNode(n1):
			e = [ e for e, n in self.iterEdgesFrom(n1) if n is n2 ]
			if e: return e[0]
			else: return None
		else:
			return None

	def areConnected(self, n1, n2):
		return self.findEdge(n1, n2) is not None

	def export(self, sparse='csr'):
		"""Returns `(A, V, E)` where `A` is an adjacency matrix as a `scipy.sparse.*_matrix` of 
		size `(n_vertices, n_vertices)` and with edge indices as defined values (including 
		explicit zeros), where * is defined by the optional input argument `sparse` (any of 'csc',
		'csr', 'bsr', 'lil', 'dok', 'coo', 'dia').
		`V` is a `dict` with `n_vertex_attributes` entries and where each value is a 
		`(n_vertices, *)` matrix (usually vector).
		`E` is a `dict` with `n_edge_attributes` entries and where each value is a `(n_edges, *)` 
		matrix (usually vector).
		Note that vertices and edges may be differently ordered compared to the current `Graph`
		implementation. Especially, `V` and `E` should be dense."""
		raise AbstractGraphError

	def squareDistance(self, attr, eta, **kwargs):
		raise AbstractGraphError

