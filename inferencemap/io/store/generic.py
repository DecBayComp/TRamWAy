
import six
from .storable import *
from collections import deque
from scipy.sparse import bsr_matrix, coo_matrix, csc_matrix, csr_matrix, \
	dia_matrix, dok_matrix, lil_matrix


class GenericStore(StoreBase):
	__slots__ = StoreBase.__slots__

	def registerStorable(self, storable):
		if not storable.handlers:
			storable = default_storable(storable.python_type, version=storable.version, \
				exposes=storable.exposes, storable_type=storable.storable_type)
		StoreBase.registerStorable(self, storable)

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
		print((objname, storable.storable_type))
		storable.poke(self, objname, obj, container)
		record = self.getRecord(objname, container)
		self.setRecordAttr('type', storable.storable_type, record)
		if storable.version is not None:
			self.setRecordAttr('version', from_version(storable.version), record)

	def poke(self, objname, obj, record):
		print((objname, type(obj), self.hasPythonType(obj)))
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
	try:
		for iobjname, iobj in assoc:
			store.poke(iobjname, iobj, sub_container)
	except TypeError:
		raise TypeError('wrong type for keys in associative list')


# peeks
def default_peek(python_type, exposes):
	def peek(store, container):
		try:
			obj = python_type()
		except TypeError:
			obj = python_type.__new__(python_type)
		attrs = list(exposes)
		#print(attrs) # debugging
		for attr in attrs: # force order instead of directly iterating over `container`
			#print((attr, attr in container)) # debugging
			if attr in container:
				val = store.peek(attr, container)
				try:
					setattr(obj, attr, val)
				except AttributeError as e:
					raise AttributeError("`setattr` failed with attribute {} in {} instance\nAttributeError: {}".format(attr, python_type, e))
			else:
				setattr(obj, attr, None)
		return obj
	return peek

def unsafe_peek(init):
	def peek(store, container):
		return init(*[ store.peek(attr, container) for attr in container ])
	return peek

def peek_with_kwargs(init, vargs):
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
		for i in container:
			assoc.append((i, store.peek(i, container)))
		#print(assoc) # debugging
	except TypeError as e:
		try:
			for i in container:
				pass
			raise e
		except:
			raise TypeError('container is not iterable; peek is not compatible')
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


# standard sequences
def seq_to_assoc(seq):
	return [ (six.b(str(i)), x) for i, x in enumerate(seq) ]
def assoc_to_list(assoc):
	return [ x for _, x in sorted(assoc, key=lambda a: int(a[0])) ]

def poke_seq(v, n, s, c):
	poke_assoc(v, n, seq_to_assoc(s), c)
def poke_dict(v, n, d, c):
	poke_assoc(v, n, d.items(), c)
def peek_list(s, c):
	return assoc_to_list(peek_assoc(s, c))
def peek_tuple(s, c):
	return tuple(peek_list(s, c))
def peek_dict(s, c):
	return dict(peek_assoc(s, c))
def peek_deque(s, c):
	return deque(peek_list(s, c))

seq_storables = [Storable(tuple, handlers=StorableHandler(poke=poke_seq, peek=peek_tuple)), \
	Storable(list, handlers=StorableHandler(poke=poke_seq, peek=peek_list)), \
	Storable(dict, handlers=StorableHandler(poke=poke_dict, peek=peek_dict)), \
	Storable(deque, handlers=StorableHandler(poke=poke_seq, peek=peek_deque))]


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

def dok_recommend(*vargs):
	raise TypeErrorWithAlternative('coo_matrix', 'dok_matrix')
dok_handler = StorableHandler(poke=dok_recommend, peek=dok_recommend)

def lil_recommend(*vargs):
	raise TypeErrorWithAlternative(('csr_matrix', 'csc_matrix'), 'dok_matrix')
lil_handler = StorableHandler(poke=lil_recommend, peek=lil_recommend)


sparse_storables = [Storable(bsr_matrix, handlers=bsr_handler), \
	Storable(coo_matrix, handlers=coo_handler), \
	Storable(csc_matrix, handlers=csc_handler), \
	Storable(csr_matrix, handlers=csr_handler), \
	Storable(dia_matrix, handlers=dia_handler), \
	Storable(dok_matrix, handlers=dok_handler), \
	Storable(lil_matrix, handlers=lil_handler)]


