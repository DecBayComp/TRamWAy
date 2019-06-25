# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
from collections import namedtuple, OrderedDict
import numpy.ma as ma
from pandas import Series, DataFrame

Matrix = namedtuple('Matrix', 'size shape dtype order')

class ArrayChain(object):
    __slots__ = ('members', 'order')

    def __init__(self, *members, **kwargs):
        order = kwargs.pop('order', 'simple') # Py2 workaround
        if kwargs:
            raise TypeError('expected at most 1 keyword arguments, got {}'.format(len(kwargs)+1))
        self.order = order
        self.members = OrderedDict()
        if members:
            if not members[1:]:
                members = members[0]
            if not isinstance(members[0], tuple):
                members = zip(members[0::2], members[1::2])
        for member, example in members:
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
        return member in self.members

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

    def at(self, a, member):
        matrix = self.members[member]
        return self._at(a, member, matrix)

    def get(self, a, member):
        matrix = self.members[member]
        return a[self._at(a, member, matrix)].reshape(matrix.shape, order=matrix.order)

    def set(self, a, member, m):
        if isinstance(m, (Series, DataFrame)):
            m = m.values
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



class ChainArray(ArrayChain):
    __slots__ = ('combined',)

    def __init__(self, *members, **kwargs):
        ArrayChain.__init__(self, *members, **kwargs)
        self.combined = np.empty(self.shape)
        # adapted copy-paste from ArrayChain.__init__
        if not members[1:]:
            members = members[0]
        kwargs = dict(zip(members[0::2], members[1::2]))
        #
        if any([isinstance(v, ma.MaskedArray) for v in kwargs.values()]):
            self.combined = ma.asarray(self.combined)
        for k in kwargs:
            self[k] = kwargs[k]

    def __getitem__(self, k):
        return self.get(self.combined, k)

    def __setitem__(self, k, v):
        self.set(self.combined, k, v)

    def update(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError('numpy array expected')
        if isinstance(self.combined, ma.MaskedArray):
            self.combined = ma.array(x, mask=self.combined.mask)
        else:
            self.combined = x


__all__ = [
    'Matrix',
    'ArrayChain',
    'ChainArray',
    ]

