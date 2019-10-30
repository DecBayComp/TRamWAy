# -*- coding: utf-8 -*-

# Copyright © 2017-2018, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""This module implements the :class:`~rwa.storable.Storable` class for TRamWAy datatypes."""

#from warnings import filterwarnings
#filterwarnings('ignore', category=FutureWarning, module='h5py', lineno=36) # best in top package __init__
import sys
from rwa import *
from .store import *
from . import store

_rwa_available = True
try:
    hdf5_agnostic_modules += ['tramway.core.analyses', 'tramway.core.scaler', \
        'tramway.tessellation', 'tramway.inference', 'tramway.feature']
except NameError: # rwa is Mock in rtd
    _rwa_available = False
    __all__ = []
else:
    __all__ = ['hdf5_storable', 'hdf5_not_storable', 'lazytype', 'lazyvalue'] # from rwa
__all__ += store.__all__ + ['store'] # from .store


if sys.version_info[0] < 3:
    from .rules import *
    from . import rules
    __all__ += rules.__all__ + ['rules']


# backward compatibility trick for `CellStats` (now `Partition`)
try:
    from tramway.tessellation.base import Partition
except:
    pass
else:
    from ..lazy import Lazy
    partition_exposes = cell_stats_expose = Lazy.__slots__ + Partition.__slots__ + ('_tesselation',)
    __all__.append('cell_stats_expose') # keep it for backward compatibility
    #__all__.append('partition_exposes')
    if _rwa_available:
        hdf5_storable(default_storable(Partition, exposes=partition_exposes), agnostic=True)
        # quick'n dirty fix; [TODO] check this is legal:
        hdf5.hdf5_service.by_storable_type['tramway.tessellation.base.CellStats'] = \
                hdf5.hdf5_service.by_storable_type['tramway.tessellation.base.Partition']

try:
    from tramway.tessellation.kdtree.dichotomy import Dichotomy, ConnectedDichotomy
except:
    pass
else:
    dichotomy_exposes = ['base_edge', 'min_depth', 'max_depth', \
        'origin', 'lower_bound', 'upper_bound', \
        'min_count', 'max_count', 'subset', 'cell']
    __all__.append('dichotomy_exposes')
    if _rwa_available:
        hdf5_storable(default_storable(Dichotomy, dichotomy_exposes), agnostic=True)
    connected_dichotomy_exposes = dichotomy_exposes + ['adjacency']
    __all__.append('connected_dichotomy_exposes')
    if _rwa_available:
        hdf5_storable(default_storable(ConnectedDichotomy, connected_dichotomy_exposes), agnostic=True)


try:
    from tramway.inference.base import Maps
except:
    pass
else:

    def poke_maps(store, objname, self, container, visited=None, _stack=None, legacy=False):
        #print('poke_maps')
        sub_container = store.newContainer(objname, self, container)
        attrs = dict(self.__dict__) # dict
        attrs['maps'] = attrs.pop('_maps')
        #print(list(attrs.keys()))
        if legacy:
            # legacy format
            if callable(self.mode):
                store.poke('mode', '(callable)', sub_container, visited=visited, _stack=_stack)
                store.poke('result', self.maps, sub_container, visited=visited, _stack=_stack)
            else:
                store.poke('mode', self.mode, sub_container, visited=visited, _stack=_stack)
                store.poke(self.mode, self.maps, sub_container, visited=visited, _stack=_stack)
        else:
            #print("poke 'maps'")
            store.poke('maps', self.maps, sub_container, visited=visited, _stack=_stack)
            del attrs['maps']
        deprecated = {}
        for a in ('distributed_translocations','partition_file','tessellation_param','version'):
            deprecated[a] = attrs.pop(a, None)
        for a in attrs:
            if not (a[0] == '_' or attrs[a] is None):
                try:
                    bool(attrs[a])
                except ValueError:
                    store.poke(a, attrs[a], sub_container, visited=visited, _stack=_stack)
                    continue
                if attrs[a] or attrs[a] == 0:
                    #print("poke '{}'".format(a))
                    store.poke(a, attrs[a], sub_container, visited=visited, _stack=_stack)
        for a in deprecated:
            if deprecated[a]:
                warn('`{}` is deprecated'.format(a), DeprecationWarning)
                #print("poke '{}'".format(a))
                store.poke(a, deprecated[a], sub_container, visited=visited, _stack=_stack)

    def peek_maps(store, container, _stack=None):
        #print('peek_maps')
        read = list(Maps.__lazy__) # do not read any lazy attribute;
        # in principle no lazy attribute should be found in `container`
        try:
            mode = store.peek('mode', container, _stack=_stack)
            read.append('mode')
        except KeyError:
            mode = None
        try:
            maps = store.peek('maps', container, _stack=_stack)
            read.append('maps')
        except KeyError:
            # former standalone files
            if mode == '(callable)':
                maps = store.peek('result', container, _stack=_stack)
                read.append('result')
                mode = None
            elif mode is None:
                raise
            else:
                maps = store.peek(mode, container, _stack=_stack)
                read.append(mode)
        maps = Maps(maps, mode=mode)
        for r in container:
            if r not in read:
                setattr(maps, r, store.peek(r, container, _stack=_stack))
        return maps

    __all__ += ['poke_maps', 'peek_maps']

    if _rwa_available:
        hdf5_storable(Storable(Maps, \
            handlers=StorableHandler(poke=poke_maps, peek=peek_maps)), agnostic=True)

from ..analyses import Analyses, base
if _rwa_available:
    _analyses_storable = default_storable(Analyses, exposes=base.Analyses.__slots__)
    _analyses_storable.storable_type = 'tramway.core.analyses.Analyses'
    hdf5_storable(_analyses_storable)

