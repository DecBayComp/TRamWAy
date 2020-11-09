# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tensorflow.python.keras import saving
from tensorflow import keras
import h5py

class Attr(str):
    __slots__ = ()
    def decode(self, encoding):
        return self
    def __iter__(self):
        if self.startswith('[') and self.endswith(']'):
            for c in self[1:-1].split():
                if c.startswith("b'") and c.endswith("'"):
                    #print(c[2:-1])
                    yield Attr(c[2:-1])
        else:
            for c in str.__iter__(self):
                yield Attr(c)

class AttrsProxy(object):
    __slots__ = ('_attrs',)
    def __init__(self, attrs):
        self._attrs = attrs
    def __contains__(self, attr):
        return attr in self._attrs
    def __getitem__(self, attr):
        assert isinstance(attr, str)
        return Attr(self._attrs[attr])
    def __getattr__(self, attr):
        assert attr not in ('_attrs','__contains__','__getitem__')
        return getattr(self._attrs, attr)

class Proxy(object):
    __slots__ = ('_f',)
    def __init__(self, f):
        self._f = f
    @property
    def attrs(self):
        return AttrsProxy(self._f.attrs)
    def __getitem__(self, attr):
        return self._f[attr]
    def __getattr__(self, attr):
        assert attr not in ('_f','attrs')
        sub = getattr(self._f, attr)
        try:
            sub.attrs
        except AttributeError:
            return sub
        else:
            return Proxy(sub)


__fix_tf_1_14_0_h5py_3_0_0__ = True

class Model(keras.Model):
    def load_weights(self, filepath, by_name=False):
        if __fix_tf_1_14_0_h5py_3_0_0__:
            # originally implemented in tensorflow.python.keras.engine.network
            with h5py.File(filepath, 'r') as f:
                f = Proxy(f)
                if by_name:
                    saving.load_weights_from_hdf5_group_by_name(f, self.layers)
                else:
                    saving.load_weights_from_hdf5_group(f, self.layers)
        else:
            keras.Model.load_weights(self, filepath, by_name)

