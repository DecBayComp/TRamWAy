# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


class GraphError(Exception):
    pass
class AbstractGraphError(NotImplementedError):
    def __str__(self):
        return ':class:`Graph` is an abstract class and cannot be instanciated'
class NodeError(GraphError):
    """Consider raising any child of :class:`NodeError` instead."""
    def __init__(self, node):
        self.node = node
class EdgeError(GraphError):
    """Consider raising any child of :class:`EdgeError` instead."""
    def __init__(self, edge):
        self.edge = edge
class MissingNodeError(NodeError):
    def __str__(self):
        return 'missing node: {}'.format(self.node)
class MissingEdgeError(EdgeError):
    def __str__(self):
        return 'missing edge: {}'.format(self.edge)
class NodeAttributeError(NodeError):
    def __init__(self, attribute):
        self.attribute = attribute
    def __str__(self):
        return 'missing node attribute: {}'.format(self.attribute)
class EdgeAttributeError(EdgeError):
    def __init__(self, attribute):
        self.attribute = attribute
    def __str__(self):
        return 'missing edge attribute: {}'.format(self.attribute)


