# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .exception import *
from .base import *
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, bsr_matrix, dok_matrix, coo_matrix, dia_matrix


class DictGraph(Graph):
    """Default implementation of :class:`Graph`. Memory efficient and suitable for very dynamic graph
    with frequent modifications to its structure, but possibly computationally slow for :class:`Gas`.

    Consider using alternative implementations in module :mod:`.graph.alloc`."""
    def __init__(self, node_defaults, edge_defaults):
        Graph.__init__(self, node_defaults, edge_defaults)
        self.nodes = dict([ (attr, {}) for attr in node_defaults ])
        self._node_counter = 0
        self.neighbors = {}
        self.edges = dict([ (attr, {}) for attr in edge_defaults ])
        self._edge_counter = 0

    def connect(self, n1, n2, **kwargs):
        if self.has_node(n1) and self.has_node(n2):
            e = [ e for e, n in self.neighbors[n1] if n is n2 ]
            if e: # if nodes are already connected
                e = e[0]
            else:
                e = self._edge_counter
                self.neighbors[n1].append((e, n2))
                self.neighbors[n2].append((e, n1))
                self._edge_counter += 1
            for attr, default in self.edge_defaults.items():
                self.edges[attr][e] = kwargs.get(attr, default)
        else:   raise MissingNodeError((n1, n2))
        return e

    def disconnect(self, n1, n2, edge=None):
        if self.has_node(n1) and self.has_node(n2):
            e = edge
            if e is None:
                e = [ e for e, n in self.neighbors[n2] if n is n1]
                if e: e = e[0]
                else: e = None
            if e is None: # e can be 0
                #raise MissingEdgeError((n1, n2))
                pass
            else: # if nodes are connected
                self.neighbors[n1].remove((e, n2))
                self.neighbors[n2].remove((e, n1))
                for attr in self.edges:
                    del self.edges[attr][e]
        else:   raise MissingNodeError((n1, n2))

    def find_edge(self, n1, n2):
        if self.has_node(n1):
            e = [ e for e, n in self.neighbors[n1] if n is n2 ]
            if e: return e[0]
            else: return None
        else:
            return None

    def get_node_attr(self, n, attr):
        try:
            return self.nodes[attr][n]
        except KeyError:
            if attr in self.nodes:
                raise MissingNodeError(n)
            else:   raise NodeAttributeError(n)

    def set_node_attr(self, n, **kwargs):
        for attr, val in kwargs.items():
            if attr in self.nodes:
                if n in self.nodes[attr]:
                    self.nodes[attr][n] = val
                else:   raise MissingNodeError(n)
            else: raise NodeAttributeError(attr)

    def get_edge_attr(self, e, attr):
        try:
            return self.edges[attr][e]
        except KeyError:
            if attr in self.edges:
                raise MissingEdgeError(e)
            else:   raise EdgeAttributeError(attr)

    def set_edge_attr(self, e, **kwargs):
        for attr, val in kwargs.items():
            if attr in self.edges:
                if e in self.edges[attr]:
                    self.edges[attr][e] = val
                else:
                    raise MissingEdgeError(e)
            else:
                raise EdgeAttributeError(attr)

    def iter_edges(self):
        any_attr = list(self.edges.keys())[0]
        return self.edges[any_attr].keys()

    def iter_edges_from(self, n):
        try:
            return self.neighbors[n]
        except KeyError:
            raise MissingNodeError(n)

    def has_node(self, n):
        return n in self.neighbors

    def add_node(self, **kwargs):
        n = self._node_counter
        for attr, default in self.node_defaults.items():
            self.nodes[attr][n] = kwargs.get(attr, default)
        self.neighbors[n] = []
        self._node_counter += 1
        return n

    def del_node(self, n):
        if self.has_node(n):
            for edge, neighbor in list(self.neighbors[n]):
                self.disconnect(n, neighbor, edge)
            del self.neighbors[n]
            for attr in self.nodes:
                del self.nodes[attr][n]
        else:   raise MissingNodeError(n)

    @property
    def size(self):
        return len(self.neighbors)

    def iter_nodes(self):
        return self.neighbors.keys()

    def export(self, sparse='csr'):
        matrix = dict(csc=csc_matrix, csr=csr_matrix, bsr=bsr_matrix, coo=coo_matrix, \
            dok=lambda x: coo_matrix(x).todok(), \
            dia=lambda x: coo_matrix(x).todia())
            #lil=lambda x: coo_matrix(x).tolil(), \ # lil does not support explicit zeros
        any_attr = list(self.node_defaults.keys())[0]
        node_order = list(self.nodes[any_attr].keys())
        node_map = np.zeros(self._node_counter)
        node_map[node_order] = np.arange(0, len(node_order))
        any_attr = list(self.edge_defaults.keys())[0]
        edge_order = list(self.edges[any_attr].keys())
        edge_map = np.zeros(self._edge_counter)
        edge_map[edge_order] = np.arange(0, len(edge_order))
        edges, nodes1, nodes2 = zip(*[ (e, n1, n2) for n1, ns in self.neighbors.items() \
                                for e, n2 in ns ])
        edges = edge_map[np.asarray(edges)]
        nodes1 = node_map[np.asarray(nodes1)]
        nodes2 = node_map[np.asarray(nodes2)]
        A = matrix[sparse]((edges, (nodes1, nodes2)))
        V = {}
        for attr, data in self.nodes.items():
            V[attr] = np.asarray(list(data.values()))
        E = {}
        for attr, data in self.edges.items():
            E[attr] = np.asarray(list(data.values()))
        return (A, V, E)

    def square_distance(self, attr, eta, eta2=None):
        if eta2 is None:
            eta2 = np.dot(eta, eta)
        x = np.vstack(self.nodes[attr].values())
        d = np.sum(x * x, axis=1) + eta2 - 2.0 * np.dot(x, eta)
        n = list(self.iter_nodes())
        def index_to_node(i):
            if isinstance(i, int):
                return n[i]
            else:   return [ n[j] for j in i ]
        return (d, index_to_node)

