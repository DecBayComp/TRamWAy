

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


