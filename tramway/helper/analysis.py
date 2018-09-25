# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import os
import itertools
import traceback
import warnings
import tramway.core.analyses as base
from tramway.core.analyses import *
from tramway.core.hdf5 import lazytype, lazyvalue, store
#import tramway.core.hdf5.store
from tramway.tessellation.base import Tessellation, CellStats
from rwa.lazy import LazyPeek, PermissivePeek


def list_rwa(path):
    warnings.warn('`list_rwa` may be removed together with module `helper.analysis`', FutureWarning)
    if not isinstance(path, (tuple, list)):
        path = (path,)
    ext = '.rwa'
    paths = []
    for p in path:
        if os.path.isdir(p):
            paths.append([ os.path.join(p, f) for f in os.listdir(p) if f.endswith(ext) ])
        else:
            if p.endswith(ext):
                ps = [p]
                paths.append(ps)
    paths = tuple(itertools.chain(*paths))
    return paths


def find_analysis(path, labels=None):
    warnings.warn('`find_analysis` may be removed together with module `helper.analysis`', FutureWarning)
    if isinstance(path, (tuple, list, set, frozenset)):
        paths = path
        matches = {}
        for path in paths:
            try:
                matches[path] = find_analysis(path, labels)
            except KeyError:
                pass
        return matches
    else:
        analyses = load_rwa(path)
        if labels:
            analyses = extract_analysis(analyses, labels)
        return analyses


def format_analyses(analyses, prefix='\t', node=None, global_prefix=''):
    if not isinstance(analyses, Analyses) and os.path.isfile(analyses):
        analyses = find_analysis(analyses)
    if node is None:
        try:    node = lazytype
        except: node = type
    return base.format_analyses(analyses, prefix, node, global_prefix)


known_lossy = [Tessellation, CellStats]


def save_rwa(path, analyses, verbose=False, force=False, compress=True, lossy=None):
    if lossy:
        warnings.warn('`lossy` is currently ignored', DeprecationWarning)
        if False:#lossy:
            if isinstance(lossy, (type, tuple, list, frozenset, set)):
                lossy = set(lossy) + known_lossy
            else:
                lossy = known_lossy
            def lossy_compress(data):
                t = lazytype(data)
                if any(issubclass(t, _t) for _t in lossy):
                    if isinstance(data, LazyPeek):
                        data = data.deep()
                    data.freeze()
            try:
                map_analyses(lossy_compress, analyses)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                if verbose:
                    print('lossy compression failed with the following error:')
                    print(traceback.format_exc())
    store.save_rwa(path, analyses, verbose, force, compress)

