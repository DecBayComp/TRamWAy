
import six
from .storable import *
from collections import deque, OrderedDict
from scipy.sparse import bsr_matrix, coo_matrix, csc_matrix, csr_matrix, \
	dia_matrix, dok_matrix, lil_matrix
import copy
import numpy
import pandas


class GenericStore(StoreBase):
	__slots__ = StoreBase.__slots__
	__slots__.append('verbose')

	def registerStorable(self, storable):
		if not storable.handlers:
			storable = default_storable(storable.python_type, version=storable.version, \
				exposes=storable.exposes, storable_type=storable.storable_type)
		StoreBase.registerStorable(self, storable)

	def strRecord(self, record, container):
		return record

	def formatRecordName(self, objname):
		raise NotImplementedError('abstract method')

	def newContainer(self, objname, obj, container):
		raise NotImplementedError('abstract method')

	def getRecord(self, objname, container):
		raise NotImplementedError('abstract method')

	def getRecordAttr(self, attr, record):
		raise NotImplementedError('abstract method')

	def setRecordAttr(self, attr, val, record):
		raise NotImplementedError('abstract method')

	def isStorable(self, record):
		return self.getRecordAttr('type', record) is not None

	def pokeNative(self, objname, obj, container):
		raise TypeError('record not supported')

	def pokeStorable(self, storable, objname, obj, container):
		#print((objname, storable.storable_type)) # debug
		storable.poke(self, objname, obj, container)
		try:
			record = self.getRecord(objname, container)
			self.setRecordAttr('type', storable.storable_type, record)
			if storable.version is not None:
				self.setRecordAttr('version', from_version(storable.version), record)
		except KeyError:
			# fake storable; silently skip
			pass

	def poke(self, objname, obj, record):
		if self.verbose:
			if self.hasPythonType(obj):
				typetype = 'storable'
			else:	typetype = 'native'
			print('writing `{}` ({} type: {})'.format(objname, typetype, type(obj).__name__))
		if obj is not None:
			objname = self.formatRecordName(objname)
			if self.hasPythonType(obj):
				storable = self.byPythonType(obj).asVersion()
				self.pokeStorable(storable, objname, obj, record)
			else:
				self.pokeNative(objname, obj, record)

	def peekNative(self, record):
		raise TypeError('record not supported')

	def peekStorable(self, storable, record):
		return storable.peek(self, record)

	def peek(self, objname, container):
		record = self.getRecord(self.formatRecordName(objname), container)
		if self.isStorable(record):
			t = self.getRecordAttr('type', record)
			v = self.getRecordAttr('version', record)
			#print((objname, self.byStorableType(t).storable_type)) # debugging
			storable = self.byStorableType(t).asVersion(v)
			return self.peekStorable(storable, record)
		else:
			#print(objname) # debugging
			return self.peekNative(record)



# pokes
def poke(exposes):
	def _poke(store, objname, obj, container):
		try:
			sub_container = store.newContainer(objname, obj, container)
		except:
			raise ValueError('generic poke not supported by store')
		for iobjname in exposes:
			iobj = getattr(obj, iobjname)
			store.poke(iobjname, iobj, sub_container)
	return _poke

def poke_assoc(store, objname, assoc, container):
	try:
		sub_container = store.newContainer(objname, assoc, container)
	except:
		raise ValueError('generic poke not supported by store')
	verbose = store.verbose
	reported_item_counter = 0
	escaped_key_counter = 0
	try:
		for iobjname, iobj in assoc:
			if isinstance(iobjname, str) or isinstance(iobjname, bytes):
				store.poke(iobjname, iobj, sub_container)
			else: # escape key
				store.poke(str(escaped_key_counter), (iobjname, iobj), sub_container)
				store.setRecordAttr('key', 'escaped', sub_container)
				escaped_key_counter += 1
			if store.verbose:
				reported_item_counter += 1
				if reported_item_counter == 9:
					store.verbose = False
					print('...')
	except TypeError as e:
		raise TypeError("wrong type for keys in associative list\n\t{}".format(e.args[0]))
	store.verbose = verbose


# peeks
def default_peek(python_type, exposes):
	with_args = False
	make = python_type
	try:
		make()
	except TypeError:
		make = lambda: python_type.__new__(python_type)
		try:
			make()
		except TypeError:
			make = lambda args: python_type.__new__(python_type, *args)
			with_args = True
	if with_args:
		def peek(store, container):
			state = []
			for attr in exposes: # force order instead of iterating over `container`
				#print((attr, attr in container)) # debugging
				if attr in container:
					state.append(store.peek(attr, container))
				else:
					state.append(None)
			return make(state)
	else:
		def peek(store, container):
			obj = make()
			for attr in exposes: # force order instead of iterating over `container`
				#print((attr, attr in container)) # debugging
				if attr in container:
					val = store.peek(attr, container)
				else:
					val = None
				try:
					setattr(obj, attr, val)
				except AttributeError as e:
					raise AttributeError("can't set attribute '{}' ({})".format(attr, python_type))
			return obj
	return peek

def unsafe_peek(init):
	def peek(store, container):
		return init(*[ store.peek(attr, container) for attr in container ])
	return peek

def peek_with_kwargs(init, vargs=[]):
	def peek(store, container):
		return init(\
			*[ store.peek(attr, container) for attr in vargs ], \
			**dict([ (attr, store.peek(attr, container)) \
				for attr in container if attr not in vargs ]))
	return peek

def peek(init, exposes):
	def _peek(store, container):
		return init(*[ store.peek(objname, container) for objname in exposes ])
	return _peek

def peek_assoc(store, container):
	assoc = []
	try:
		if store.getRecordAttr('key', container) == 'escaped':
			for i in container:
				assoc.append(store.peek(i, container))
		else:
			for i in container:
				assoc.append((store.strRecord(i, container), store.peek(i, container)))
		#print(assoc) # debugging
	except TypeError as e:
		try:
			for i in container:
				pass
			raise e
		except TypeError:
			raise TypeError("container is not iterable; peek is not compatible\n\t{}".format(e.args[0]))
	return assoc


# default storable with __new__ and __slots__
def default_storable(python_type, exposes=None, version=None, storable_type=None):
	if exposes is None:
		try:
			exposes = python_type.__slots__
		except:
			# take __dict__ and sort out the class methods
			raise AttributeError('either define the `exposes` argument or the `__slots__` attribute for type: {!r}'.format(python_type))
	return Storable(python_type, key=storable_type, \
		handlers=StorableHandler(version=version, exposes=exposes, \
		poke=poke(exposes), peek=default_peek(python_type, exposes)))


def kwarg_storable(python_type, exposes=None, version=None, storable_type=None, init=None, vargs=[]):
	if init is None:
		init = python_type
	if exposes is None:
		try:
			exposes = python_type.__slots__
		except:
			# take __dict__ and sort out the class methods
			raise AttributeError('either define the `exposes` argument or the `__slots__` attribute for type: {!r}'.format(python_type))
	return Storable(python_type, key=storable_type, handlers=StorableHandler(version=version, \
		poke=poke(exposes), peek=peek_with_kwargs(init, vargs), exposes=exposes))


# standard sequences
def seq_to_assoc(seq):
	return [ (six.b(str(i)), x) for i, x in enumerate(seq) ]
def assoc_to_list(assoc):
	return [ x for _, x in sorted(assoc, key=lambda a: int(a[0])) ]

def poke_seq(v, n, s, c):
	poke_assoc(v, n, seq_to_assoc(s), c)
def poke_dict(v, n, d, c):
	poke_assoc(v, n, d.items(), c)
def poke_OrderedDict(v, n, d, c):
	poke_dict(v, n, d, c)
def peek_list(s, c):
	return assoc_to_list(peek_assoc(s, c))
def peek_tuple(s, c):
	return tuple(peek_list(s, c))
def peek_dict(s, c):
	return dict(peek_assoc(s, c))
def peek_deque(s, c):
	return deque(peek_list(s, c))
def peek_OrderedDict(s, c):
	return OrderedDict(peek_assoc(s, c))

seq_storables = [Storable(tuple, handlers=StorableHandler(poke=poke_seq, peek=peek_tuple)), \
	Storable(list, handlers=StorableHandler(poke=poke_seq, peek=peek_list)), \
	Storable(dict, handlers=StorableHandler(poke=poke_dict, peek=peek_dict)), \
	Storable(deque, handlers=StorableHandler(poke=poke_seq, peek=peek_deque)), \
	Storable(OrderedDict, handlers=StorableHandler(poke=poke_OrderedDict, peek=peek_OrderedDict))]


# functions (built-in and standard)
def fake_poke(*vargs, **kwargs):
	pass
def fail_peek(unsupported_type):
	helper = "unsupported type '{}'\n; consider using `io.store.Functional` instead"
	helper = helper.format(unsupported_type)
	def peek(*vargs, **kwargs):
		def f(*vargs, **kwargs):
			raise TypeError(helper)
		return f
	return peek

function_storables = [\
	Storable(len.__class__, handlers=StorableHandler(poke=fake_poke, peek=fail_peek)), \
	Storable(poke.__class__, handlers=StorableHandler(poke=fake_poke, peek=fail_peek))]


def poke_native(getstate):
	def poke(service, objname, obj, container):
		service.pokeNative(objname, getstate(obj), container)
	return poke

def peek_native(make):
	def peek(service, container):
		return make(service.peekNative(container))
	return peek


# numpy.dtype
numpy_storables = [\
	Storable(numpy.dtype, handlers=StorableHandler(poke=poke_native(lambda t: t.str), \
		peek=peek_native(numpy.dtype)))]


def handler(init, exposes):
	return StorableHandler(poke=poke(exposes), peek=peek(init, exposes))

# scipy.sparse storable instances
bsr_exposes = ['shape', 'data', 'indices', 'indptr']
def mk_bsr(shape, data, indices, indptr):
	return bsr_matrix((data, indices, indptr), shape=shape)
bsr_handler = handler(mk_bsr, bsr_exposes)

coo_exposes = ['shape', 'data', 'row', 'col']
def mk_coo(shape, data, row, col):
	return bsr_matrix((data, (row, col)), shape=shape)
coo_handler = handler(mk_coo, coo_exposes)

csc_exposes = ['shape', 'data', 'indices', 'indptr']
def mk_csc(shape, data, indices, indptr):
	return csc_matrix((data, indices, indptr), shape=shape)
csc_handler = handler(mk_csc, csc_exposes)

csr_exposes = ['shape', 'data', 'indices', 'indptr']
def mk_csr(shape, data, indices, indptr):
	return csr_matrix((data, indices, indptr), shape=shape)
csr_handler = handler(mk_csr, csr_exposes)

dia_exposes = ['shape', 'data', 'offsets']
def mk_dia(shape, data, offsets):
	return dia_matrix((data, offsets), shape=shape)
dia_handler = handler(mk_dia, dia_exposes)

# previously
def dok_recommend(*vargs):
	raise TypeErrorWithAlternative('dok_matrix', 'coo_matrix')
dok_handler = StorableHandler(poke=dok_recommend, peek=dok_recommend)
# now
def dok_poke(service, matname, mat, container):
	coo_handler.poke(service, matname, mat.tocoo(), container)
def dok_peek(service, container):
	return coo_handler.peek(service, container).todok()
dok_handler = StorableHandler(poke=dok_poke, peek=dok_peek)

# previously
def lil_recommend(*vargs):
	raise TypeErrorWithAlternative('lil_matrix', ('csr_matrix', 'csc_matrix'))
lil_handler = StorableHandler(poke=lil_recommend, peek=lil_recommend)
# now
def lil_poke(service, matname, mat, container):
	csr_handler.poke(service, matname, mat.tocsr(), container)
def lil_peek(service, container):
	return csr_handler.peek(service, container).tolil()
lil_handler = StorableHandler(poke=lil_poke, peek=lil_peek)


sparse_storables = [Storable(bsr_matrix, handlers=bsr_handler), \
	Storable(coo_matrix, handlers=coo_handler), \
	Storable(csc_matrix, handlers=csc_handler), \
	Storable(csr_matrix, handlers=csr_handler), \
	Storable(dia_matrix, handlers=dia_handler), \
	Storable(dok_matrix, handlers=dok_handler), \
	Storable(lil_matrix, handlers=lil_handler)]


def poke_index(service, name, obj, container):
	poke_seq(service, name, obj.tolist(), container)
def peek_index(service, container):
	return pandas.Index(peek_list(service, container))

# as such in Python3; force it in Python2 to be the same
pandas_storables = [Storable(pandas.Index, \
	key='Python.pandas.core.index.Index', \
	handlers=StorableHandler(poke=poke_index, peek=peek_index))]


def namedtuple_storable(namedtuple, *args, **kwargs):
	return default_storable(namedtuple, namedtuple._fields, *args, **kwargs)

