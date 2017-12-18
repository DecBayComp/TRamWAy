# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .lazy import Lazy


class AnalysesView(dict):
	__slots__ = ('__analyses__',)
	def __init__(self, analyses):
		self.__analyses__ = analyses
	def __nonzero__(self):
		#return self.__analyses__._instances.__nonzero__()
		return bool(self.__analyses__._instances) # Py2
	def __len__(self):
		return len(self.__analyses__._instances)
	def __missing__(self, label):
		raise KeyError('no such analysis instance: {}'.format(label))
	def __iter__(self):
		return self.__analyses__._instances.__iter__()
	def __contains__(self, label):
		return self.__analyses__._instances.__contains__(label)
	def keys(self):
		return self.__analyses__._instances.keys()

class InstancesView(AnalysesView):
	__slots__ = ()
	def __str__(self):
		return self.__analyses__._instances.__str__()
	def __getitem__(self, label):
		return self.__analyses__._instances[label]
	def __setitem__(self, label, analysis):
		self.__analyses__._instances[label] = analysis
	def __delitem__(self, label):
		self.__analyses__._instances.__delitem__(label)
		try:
			self.__analyses__._comments.__delitem__(label)
		except KeyError:
			pass
	def values(self):
		return self.__analyses__._instances.values()
	def items(self):
		return self.__analyses__._instances.items()

class CommentsView(AnalysesView):
	__slots__ = ()
	def __str__(self):
		return self.__analyses__._comments.__str__()
	def __getitem__(self, label):
		try:
			return self.__analyses__._comments[label]
		except KeyError:
			if label in self.__analyses__._instances:
				return None
			else:
				self.__missing__(label)
	def __setitem__(self, label, comment):
		if label in self.__analyses__._instances:
			if comment:
				self.__analyses__._comments[label] = comment
			else:
				self.__delitem__(label)
		else:
			self.__missing__(label)
	def __delitem__(self, label):
		self.__analyses__._comments.__delitem__(label)
	def values(self):
		return self.__analyses__._comments.values()
	def items(self):
		return self.__analyses__._comments.items()


class Analyses(Lazy):
	"""
	Generic container with labels and comments for analyses on some data.

	For example, the various sampling strategies (`instances`) explored 
	in relation with some molecule location data (`data`)
	or the various dynamics parameter maps (`instances`) infered from a same sample (`data`).

	Instances and comments are addressable with keys refered to as "labels".

	Attributes:

		data (any): common data which the instances apply to or derive from.

		instances (dict): analyses on the data; keys are natural integers or string labels.

		comments (dict): comments associated to the analyses; keys are a subset of the keys
			in `instances`.

	"""
	__slots__ = ['_data', '_instances', '_comments']

	def __init__(self, data=None):
		self._data = data
		self._instances = {}
		self._comments = {}

	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, d):
		self._data = d
		self._instances = {}
		self._comments = {}

	@property
	def instances(self):
		return InstancesView(self)

	@instances.setter
	def instances(self, a):
		raise AttributeError('read-only attribute')

	@property
	def comments(self):
		return CommentsView(self)

	@comments.setter
	def comments(self, c):
		raise AttributeError('read-only attribute')

	@property
	def labels(self):
		return self.instances.keys()

	@labels.setter
	def labels(self, l):
		raise AttributeError('read-only attribute')

	def autoindex(self):
		"""
		Determine the lowest available natural integer for use as key in `instances` and `comments`.
		"""
		i = 0
		if self.instances:
			while i in self.instances:
				i += 1
		return i

	def add(self, analysis, label=None, comment=None):
		"""
		Add an analysis.

		Adding an analysis at an existing label overwrites the existing analysis 
		instance and deletes the associated comment if any.

		Arguments:

			analysis (any): analysis instance.

			label (any): key for the analysis; calls :met:`autoindex` if undefined.

			comment (str): associated comment.

		"""
		if label is None:
			label = self.autoindex()
		self.instances[label] = analysis
		if comment:
			self.comments[label] = comment
		else:
			try:
				del self.comments[label]
			except KeyError:
				pass

	def unload(self, visited=None):
		"""
		Overloads :met:`Lazy.unload`.
		"""
		if visited is None:
			visited = set()
		elif self in visited:
			# already unloaded
			return
		visited.add(self)
		if isinstance(self.data, Lazy):
			self.data.unload(visited)
		for label in self.instances:
			if isinstance(self.instances[label], Lazy):
				self.instances[label].unload(visited)

	def __nonzero__(self):
		return self.instances.__nonzero__()
	def __len__(self):
		return self.instances.__len__()
	def __missing__(self, label):
		self.instances.__missing__(label)
	def __iter__(self):
		return self.instances.__iter__()
	def __contains__(self, label):
		return self.instances.__contains__(label)
	def __getitem__(self, label):
		return self.instances.__getitem__(label)
	def __setitem__(self, label, analysis):
		self.instances.__setitem__(label, analysis)
	def __delitem__(self, label):
		self.instances.__delitem__(label)


def select_analysis(analyses, labels):
	"""
	Extract an analysis from a hierarchy of analyses.

	The elements of an :class:`Analyses` instance can be other :class:`Analyses` objects. 
	As such, analyses are structured in a tree that exhibits as many logically-consecutive 
	layers as there are processing steps.

	Arguments:

		analyses (Analyses): hierarchy of analyses, with `instances` possibly containing
			other :class:`Analyses` instances.

		labels (int, str or sequence of int and str): analyses label(s); the first label
			addresses the first layer of analyses instances, the second label addresses
			the second layer of analyses and so on.

	Returns:

		Analyses: copy of the analyses along the path defined by `labels`. 
	"""
	if not isinstance(labels, (tuple, list)):
		labels = [labels]
	analysis = instance = Analyses(analyses.data)
	for label in labels:
		analysis, = \
			analysis.instances[label], analysis.comment[label] = \
			analyses.instances[label].copy(), analyses.comment[label]
		analyses = analyses.instances[label]
	return instance

