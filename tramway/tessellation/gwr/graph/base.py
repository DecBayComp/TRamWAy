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

    def get_node_attr(self, n, attr):
        raise AbstractGraphError
    def set_node_attr(self, n, **kwargs):
        raise AbstractGraphError
    def get_edge_attr(self, e, attr):
        raise AbstractGraphError
    def set_edge_attr(self, e, **kwargs):
        raise AbstractGraphError
    @property
    def size(self):
        return len(list(self.iter_nodes))
    def iter_nodes(self):
        '''Returns an iterator over all the nodes of the graph. A copy of the returned
        view should be made if the structure of the graph had to be altered.'''
        raise AbstractGraphError
    def iter_neighbors(self, n):
        '''Returns an iterator over the neighbor nodes of a given node. A copy of the returned
        view should be made if the structure of the graph had to be altered.'''
        return [ n for _, n in self.iter_edges_from(n) ]
    def iter_edges(self):
        '''Returns an iterator over all the edges of the graph. A copy of the returned
        view should be made if the structure of the graph had to be altered.'''
        raise AbstractGraphError
    def iter_edges_from(self, n):
        '''Returns an iterator over the ``(edge, neighbor_node)`` index pairs from a given node.
        A copy of the returned view should be made if the structure of the graph had to be altered.'''
        raise AbstractGraphError
    def has_node(self, n):
        return n in self.iter_nodes()

    def add_node(self, **kwargs):
        raise AbstractGraphError
    def del_node(self, n):
        raise AbstractGraphError
    def stands_alone(self, n):
        '''Returns ``True`` if node `n` has no neighbor.'''
        return not self.iter_neighbors(n)

    def find_edge(self, n1, n2):
        if self.has_node(n1):
            e = [ e for e, n in self.iter_edges_from(n1) if n is n2 ]
            if e: return e[0]
            else: return None
        else:
            return None

    def are_connected(self, n1, n2):
        return self.find_edge(n1, n2) is not None

    def export(self, sparse='csr'):
        """Returns ``(A, V, E)`` where ``A`` is an adjacency matrix as a `scipy.sparse.*_matrix`
        of size ``(n_vertices, n_vertices)`` and with edge indices as defined values (including
        explicit zeros), where * is defined by the optional input argument `sparse` (any of
        '*csc*', '*csr*', '*bsr*', '*lil*', '*dok*', '*coo*', '*dia*').
        ``V`` is a `dict` with ``n_vertex_attributes`` entries and where each value is a
        ``(n_vertices, *)`` matrix (usually vector).
        ``E`` is a `dict` with ``n_edge_attributes`` entries and where each value is a
        ``(n_edges, *)`` matrix (usually vector).
        Note that vertices and edges may be differently ordered compared to the current `Graph`
        implementation. Especially, ``V`` and ``E`` should be dense."""
        raise AbstractGraphError

    def squareDistance(self, attr, eta, **kwargs):
        raise AbstractGraphError

