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
import itertools
import copy
import traceback
import warnings


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
	def get(self, label, default=None):
		return self.__analyses__._instances.get(label, default)
	def pop(self, label, default=None):
		analysis = self.get(label, default)
		self.__delitem__(label)
		return analysis

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
	in relation with some molecule location data (`data` or equivalently `artefact`)
	or the various dynamics parameter maps (`instances`) infered from a same sample (`data` or 
	`artefact`).

	Instances and comments are addressable with keys refered to as "labels".

	Attributes:

		data/artefact (any): common data which the instances apply to or derive from.

		instances (dict): analyses on the data; keys are natural integers or string labels.

		comments (dict): comments associated to the analyses; keys are a subset of the keys
			in `instances`.

	"""
	__slots__ = ('_data', '_instances', '_comments')

	def __init__(self, data=None):
		Lazy.__init__(self)
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
	def artefact(self):
		return self.data

	@artefact.setter
	def artefact(self, a):
		self.data = a

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

	def autoindex(self, pattern=None):
		"""
		Determine the lowest available natural integer for use as key in `instances` and `comments`.

		If `pattern` is an integer, `autoindex` returns the pattern unchanged.

		Arguments:

			pattern (str): label with a *'\*'* to be replaced by a natural integer.

		Returns:

			int or str: index or label.
		"""
		if pattern:
			try:
				pattern = int(pattern) # this also checks type
			except ValueError:
				pass
			if not isinstance(pattern, int) and '*' in pattern:
				f = lambda i: pattern.replace('*', str(i))
			else:
				return pattern
		else:
			f = lambda i: i
		i = 0
		if self.instances:
			while f(i) in self.instances:
				i += 1
		return f(i)

	def add(self, analysis, label=None, comment=None):
		"""
		Add an analysis.

		Adding an analysis at an existing label overwrites the existing analysis 
		instance and deletes the associated comment if any.

		Arguments:

			analysis (any): analysis instance.

			label (any): key for the analysis; calls :meth:`autoindex` if undefined.

			comment (str): associated comment.

		"""
		label = self.autoindex(label)
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
		Overloads :meth:`Lazy.unload`.
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


def map_analyses(fun, analyses, label=False, comment=False, depth=False, allow_tuples=False):
	with_label, with_comment, with_depth = label, comment, depth
	def _fun(x, **kwargs):
		y = fun(x, **kwargs)
		if not allow_tuples and isinstance(y, tuple):
			raise ValueError('type conflict: function returned a tuple')
		return y
	def _map(analyses, label=None, comment=None, depth=0):
		kwargs = {}
		if with_label:
			kwargs['label'] = label
		if with_comment:
			kwargs['comment'] = comment
		if with_depth:
			kwargs['depth'] = depth
		node = _fun(analyses.data, **kwargs)
		if analyses.instances:
			depth += 1
			tree = []
			for label in analyses.instances:
				child = analyses.instances[label]
				comment = analyses.comments[label]
				if isinstance(child, Analyses):
					tree.append(_map(child, label, comment, depth))
				else:
					if with_label:
						kwargs['label'] = label
					if with_comment:
						kwargs['comment'] = comment
					if with_depth:
						kwargs['depth'] = depth
					tree.append(_fun(child, **kwargs))
			return (node, tuple(tree))
		else:
			return node
	return _map(analyses)


def extract_analysis(analyses, labels):
	"""
	Extract an analysis from a hierarchy of analyses.

	The elements of an :class:`~tramway.core.analyses.Analyses` instance can be other 
	:class:`~tramway.core.analyses.Analyses` objects. 
	As such, analyses are structured in a tree that exhibits as many logically-consecutive 
	layers as there are processing steps.

	Arguments:

		analyses (tramway.core.analyses.Analyses):
			hierarchy of analyses, with `instances` possibly containing
			other :class:`~tramway.core.analyses.Analyses` instances.

		labels (int, str or sequence of int and str):
			analyses label(s); the first label addresses the first layer of 
			analyses instances, the second label addresses the second layer of 
			analyses and so on.

	Returns:

		tramway.core.analyses.Analyses: copy of the analyses along the path defined by `labels`. 
	"""
	if not isinstance(labels, (tuple, list)):
		labels = [labels]
	analysis = instance = Analyses(analyses.data)
	for label in labels:
		analysis.instances[label] = copy.copy(analyses.instances[label])
		try:
			analysis.comments[label] = analyses.comments[label]
		except KeyError:
			pass
		analysis, analyses = analysis.instances[label], analyses.instances[label]
	return instance


def _append(s, ls):
	"""for internal use in :fun:`label_paths`."""
	if ls:
		ss = []
		for ok, _ls in ls:
			if isinstance(ok, bool):
				l, _ls = _ls, None
			else:
				ok, l = ok
			_s = list(s) # copy
			_s.append(l)
			if ok:
				ss.append([tuple(_s)])
			if _ls:
				ss.append(_append(_s, _ls))
		return itertools.chain(*ss)
	else:
		return []

def label_paths(analyses, filter):
	"""
	Find label paths for analyses matching a criterion.

	Arguments:

		analyses (tramway.core.analyses.Analyses):
			hierarchy of analyses, with `instances` possibly containing
			other :class:`~tramway.core.analyses.Analyses` instances.

		filter (type or callable):
			criterion over analysis data.

	Returns:

		list of tuples:
			list of label paths to matching analyses.

	"""
	if isinstance(filter, type):
		_type = filter
		_map = lambda node, label: (isinstance(node, _type), label)
	elif callable(filter):
		_map = lambda node, label: (filter(node), label)
	else:
		raise TypeError('`filter` is neither a type nor a callable')
	_, labels = map_analyses(_map, analyses, label=True, allow_tuples=True)
	return list(_append([], labels))


def find_artefacts(analyses, filters, labels=None, quantifiers=None):
	"""
	Find related artefacts.

	Filters are applied to find data elements (artefacts) along a single path specified by `labels`.

	Arguments:

		analyses (tramway.core.analyses.Analyses): hierarchy of analyses.

		filters (type or callable or tuple or list): list of criteria, a criterion being
			a boolean function or a type.

		labels (list): label path.

		quantifiers (str or tuple or list): list of quantifers, a quantifier for now being
			either *'first'*, *'last'* or *'all'*; a quantifier should be defined for each 
			filter;	default is 'last' (admits value ``None``).

	Returns:

		tuple: matching data elements/artefacts.

	Example::

		cells, maps = find_artefacts(analyses, (CellStats, Maps))

	"""
	# filters
	if not isinstance(filters, (tuple, list)):
		filters = (filters,)
	# quantifiers
	quantifier = 'last'
	if quantifiers:
		if not isinstance(quantifiers, (tuple, list)):
			quantifiers = (quantifiers,)
		if len(quantifiers) == len(filters):
			filters = zip(quantifiers, filters)
		elif quantifiers[1:]:
			warnings.warn('wrong number of quantifiers; ignoring them')
		else:
			quantifier = quantifiers[0]
	# labels
	if labels is None:
		labels = []
	elif isinstance(labels, (tuple, list)):
		labels = list(labels) # copy
	else:
		labels = [labels]
	matches, lookup = [], True
	for i, _filter in enumerate(filters):
		if quantifiers:
			quantifier, _fitler = _filter
		if isinstance(_filter, type):
			_type = _filter
			_filter = lambda a: isinstance(a, _type)
		match = []
		while True:
			if lookup:
				try:
					label = labels.pop(0)
				except IndexError:
					_labels = list(analyses.labels)
					if not _labels:
						if match:
							break
						else:
							raise ValueError('no match for {}{} filter'.format(i+1,
								{1: 'st', 2: 'nd', 3: 'rd'}.get(i+1, 'th')))
					elif _labels[1:]:
						if match and i + 1 == len(filters):
							break
						raise ValueError('multiple labels; argument `labels` required')
					else:
						label = _labels[0]
				try:
					analyses = analyses.instances[label]
				except KeyError:
					available = str(list(analyses.labels))[1:-1]
					if available:
						raise KeyError("missing label '{}'; available labels are: {}".format(label, available))
					else:
						raise KeyError("missing label '{}'; no labels available".format(label))
			lookup = True
			if _filter(analyses.artefact):
				match.append(analyses.artefact)
			elif match:
				lookup = False
				break
		if quantifier in ('first', ):
			match = match[0]
		elif quantifier in ('last', None):
			match = match[-1]
		elif quantifier not in ('all', '+'):
			raise ValueError('invalid quantifier: {}'.format(quantifier))
		matches.append(match)
	return tuple(matches)


def coerce_labels(analyses):
	for label in tuple(analyses.labels):
		if isinstance(label, (int, str)):
			coerced = label
		else:
			try: # Py2
				coerced = label.encode('utf-8')
			except AttributeError: # Py3
				try:
					coerced = label.decode('utf-8')
				except AttributeError: # numpy.int64?
					coerced = int(label)
			assert isinstance(coerced, (int, str))
		try:
			comment = analyses.comments[label]
		except KeyError:
			pass
		analysis = analyses.instances.pop(label)
		if isinstance(analysis, Analyses):
			analysis = coerce_labels(analysis)
		analyses.instances[coerced] = analysis
		if comment:
			analyses.comments[coerced] = comment
	return analyses
