
import numpy as np
from collections import namedtuple, OrderedDict

Matrix = namedtuple('Matrix', 'size shape dtype order')

class ArrayChain(object):
	__slots__ = ['members', 'order']

	def __init__(self, order='simple', **kwargs):
		self.order = order
		self.members = OrderedDict()
		for member, example in kwargs.items():
			rd = 'C'
			if isinstance(example, tuple):
				sz = np.prod(np.asarray(example[0]))
				try:
					sh, dt, rd = example
				except:
					sh, dt = example
			else:
				sz, sh, dt = example.size, example.shape, example.dtype
			self.members[member] = Matrix(sz, sh, dt, rd)

	def __contains__(self, member):
		return member in self.seq

	def __getitem__(self, member):
		return self.members[member]

	def __setitem__(self, member, example):
		rd = 'C'
		if isinstance(example, tuple):
			sz = np.prod(np.asarray(example[0]))
			try:
				sh, dt, rd = example
			except:
				sh, dt = example
		else:
			sz, sh, dt = example.size, example.shape, example.dtype
		self.members[member] = Matrix(sz, sh, dt, rd)

	def __delitem__(self, member):
		del self.members[member]

	def _at(self, a, member, matrix):
		k = list(self.members.keys()).index(member)
		if self.order == 'simple':
			i0 = sum([ m.size for m in list(self.members.values())[:k] ])
			return slice(i0, i0 + matrix.size)
		else:
			raise NotImplementedError("only 'simple' order is supported")

	def at(self, a, matrix):
		matrix = self.members[member]
		return self._at(a, member, matrix)

	def get(self, a, member):
		matrix = self.members[member]
		return np.reshape(a[self._at(a, member, matrix)], matrix.shape, order=matrix.order)

	def set(self, a, member, m):
		matrix = self.members[member]
		if matrix.shape == m.shape:
			a[self._at(a, member, matrix)] = m.flatten(matrix.order)
		else:
			raise ValueError('wrong matrix size')

	@property
	def size(self):
		return sum([ m.size for m in self.members.values() ])

	@property
	def shape(self):
		return (self.size,)

