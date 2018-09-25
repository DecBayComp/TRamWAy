# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import scipy.sparse.csgraph as csgraph
import itertools


def dilation(node, graph, step=1, boundary=None):
    """
    Expand across a graph from a node, or equivalently from a cell across an adjacency matrix.

    Graph expansions can be further expanded::

        crawled_nodes = dilation(node, graph, step=slice(3))
        more_crawled_nodes = dilation(crawled_nodes, graph, step=slice(8))

    Arguments:

        node (int or dict of sets): row/column index in `graph` (node/cell index)
            or partial `dilation` crawl.

        graph (scipy.sparse.csr_matrix): square adjacency matrix.

        step (int or (int, int)): minimum and maximum (+1) numbers of dilation steps.

        boundary (list or set): nodes to be excluded.

    Returns:

        dict of sets: sets of node indices in a :class:`dict` with steps as keys.
    """
    if boundary and not isinstance(boundary, set):
        boundary = set(boundary)
    if isinstance(step, slice):
        smin, smax = step.start, step.stop
    elif isinstance(step, tuple):
        smin, smax = step
    else:
        smin = step
        smax = smin + 1
    if np.isscalar(node):
        try:
            graph.indices[node]
        except IndexError:
            raise IndexError('node index out of range')
        except:
            raise TypeError('graph is not a csr sparse matrix')
        nodes = {0: {node}}
        s = 0
    elif isinstance(node, dict):
        nodes = node
        if not nodes:
            return nodes
        s = max(nodes.keys())
    if smax is not None and smax <= s + 1:
        # may raise KeyError; this is alright
        nodes = { _s: nodes[_s] for _s in range(smin, smax) }
        if boundary:
            s = smax - 1
            while nodes:
                nodes[s] -= boundary
                if nodes[s]:
                    break
                else:
                    del nodes[s]
                    s -= 1
            for _s in nodes:
                if nodes[_s] & boundary:
                    raise NotImplementedError
        return nodes
    while smax is None or s + 1 < smax:
        ns = set()
        for n in nodes[s]:
            ns |= set(graph.indices[graph.indptr[n]:graph.indptr[n+1]])
        for t in nodes:
            ns -= nodes[t]
        if boundary:
            ns -= boundary
        # stopping criteria:
        if not ns:
            break
        s += 1
        nodes[s] = ns
    if smin:
        if smax is not None and smin == smax - 1:
            try:
                nodes = {smin: nodes[smin]}
            except KeyError:
                nodes = {}
        else:
            nodes = { s: nodes[s] for s in nodes if smin <= s }
    return nodes


class NoSolutionError(ValueError):
    pass


def memoize(f):
    y_name = 'distance'
    _context = None
    known_computation = {}
    def wrapper(x, context, *args, **kwargs):
        if args:
            args = list(args)
            y = args.pop(0)
        else:
            try:
                y = kwargs.pop(y_name)
            except KeyError:
                return f(x, context, **kwargs) # do not cache
        print((known_computation, y_name))
        if _context is not context:
            _context = context
            known_computation = {} # new memoization table
        try:
            ret = known_computation[(x, y)]
        except KeyError:
            ret = known_computation[(x, y)] = f(x, context, y, *args, **kwargs)
        return ret
    return wrapper

#@memoize
def contour(cell, adjacency, distance=1, nodes=None, dilation_adjacency=None,
        fallback=False, debug=False, cells=None):
    # cells was useful to debugging
    if nodes is None:
        if dilation_adjacency is None:
            dilation_adjacency = adjacency
        nodes = dilation(cell, dilation_adjacency, step=slice(distance+1))
    try:
        ns = nodes[distance]
    except KeyError:
        if fallback:
            return ([], False)
        else:
            raise NoSolutionError
    strings = []
    while ns:
        seed = ns.pop()
        string = [seed]
        for _ in range(2):
            _any = seed
            string = string[::-1]
            while True:
                _next = set(adjacency.indices[adjacency.indptr[_any]:adjacency.indptr[_any+1]])
                _next &= ns
                if _next:
                    _any = _next.pop()
                    ns.remove(_any)
                    string.append(_any)
                else:
                    break
        if string[1:]: # discard isolated points
            strings.append(string)
    if not strings[1:]:
        string = strings[0]
        _next = adjacency.indices[adjacency.indptr[string[-1]]:adjacency.indptr[string[-1]+1]]
        if string[0] in _next:
            if fallback:
                return (string, True)
            else:
                return string
    # get the solution at distance-1
    if distance == 1:
        if fallback:
            return ([], False)
        else:
            raise NoSolutionError
    # inner contour
    cont = contour(cell, adjacency, distance=distance-1, nodes=nodes,
        dilation_adjacency=dilation_adjacency, fallback=fallback, debug=debug, cells=cells)
    if fallback:
        if cont[1]:
            cont = cont[0]
        else:
            return cont
    inner = set(cont)
    min_length = len(inner)
    any_inserted = False
    # insert one string at a time into the inner contour
    for graft in strings:
        # first shorten `graft` so that both ends stand near `cont`
        try:
            _graft_bracket = []
            for _i, _di in ((0,1),(-1,-1)):
                while True:
                    graft[_i+_di] # raise IndexError if no such element
                    _head = graft[_i]
                    _neighbours = set(adjacency.indices[adjacency.indptr[_head]:adjacency.indptr[_head+1]]) & inner
                    if _neighbours:
                        _graft_bracket.append(_i)
                        break
                    _i += _di
            _i, _j = _graft_bracket
            if _j < -1:
                graft = graft[_i:_j+1]
            elif 0 < _i:
                graft = graft[_i:]
            if not graft[1:]: # if len(graft) < 2
                raise IndexError
        except (IndexError, ValueError):
            if debug:
                print('skipping graft: no contact point')
            continue
        # find the extreme (bracket) inner nodes
        outer = set(graft)
        _all_neighbours = [ set(adjacency.indices[adjacency.indptr[_n]:adjacency.indptr[_n+1]])
                for _n in graft ]
        _inner_neighbours = set()
        for _a in _all_neighbours:
            _inner_neighbours |= _a
        _inner_neighbours &= inner
        _inner_targets = _bracket_elements(_inner_neighbours, adjacency, (graft[0], graft[-1]),
            cont, len(graft), debug=debug)
        if len(_inner_targets) != 2:
            if debug:
                print('skipping graft: elements are not contiguous')
            continue
        # find the contact nodes, inner and outer
        join = []
        for _target in _inner_targets:
            _outer_neighbours = []
            for _n, _a in zip(graft, _all_neighbours):
                if _target in _a:
                    _outer_neighbours.append((_n, _a))
            _subsets = [outer, inner]
            while _outer_neighbours[1:] and _subsets[1:]:
                # while multiple solutions
                # first favor the lesser number of inner neighbours
                # and then favor the lesser number of outer neighbours
                _subset = _subsets.pop()
                _min_count = len(_subset) + 1
                for _n, _a in _outer_neighbours:
                    _c = len(_a & _subset)
                    if _c < _min_count:
                        _candidates = []
                        _min_count = _c
                    if _c == _min_count:
                        _candidates.append((_n, _a))
                _outer_neighbours = _candidates
            join.append((_target, _outer_neighbours[0][0]))
        # get the indices of the contact nodes
        _inner2, _outer2 = join.pop()
        _inner1, _outer1 = join.pop()
        _remove = _inner_neighbours - set((_inner1, _inner2))
        _i1, _i2 = cont.index(_inner1), cont.index(_inner2)
        _j1, _j2 = graft.index(_outer1), graft.index(_outer2)
        _dj = 1 if _j1 <= _j2 else -1
        # assertions
        if debug:
            _ni, _nj = cont[_i1], graft[_j1]
            assert _nj in adjacency.indices[adjacency.indptr[_ni]:adjacency.indptr[_ni+1]]
            _ni, _nj = cont[_i2], graft[_j2]
            assert _nj in adjacency.indices[adjacency.indptr[_ni]:adjacency.indptr[_ni+1]]
        # solution 1: cut `cont` from _i1 to _i2
        # solution 2: cut `cont` from _i2 to _i1
        graft_length = abs(_j2 - _j1) + 1
        if graft_length < 2:
            if debug:
                print('skipping graft: too short')
            continue
        if _i1 < _i2:
            len1 = graft_length + len(cont) - (_i2 - _i1 - 1)
            len2 = graft_length + (_i2 - _i1 + 1)
            cut1 = set(cont[_i1+1:_i2]) # solution 1: cont[:_i1+1]+graft+cont[_i2:]
            cut2 = set(cont[:_i1] + cont[_i2+1:]) # solution 2: graft + cont[_i1:_i2+1]
        else:
            len1 = graft_length + (_i1 - _i2 + 1)
            len2 = graft_length + len(cont) - (_i1 - _i2 - 1)
            cut1 = set(cont[:_i2] + cont[_i1+1:])
            cut2 = set(cont[_i2+1:_i1])
        count1, count2 = [ len(_cut & _remove) for _cut in (cut1, cut2) ]
        if count1 < count2 and min_length <= len2:
            # solution 2 wins
            _dj = -_dj
            _j1 += _dj
            if _j1 == -1:
                _j1 = None
            graft = graft[_j2:_j1:_dj]
            if _i1 < _i2:
                cont = graft + cont[_i1:_i2+1]
            else:
                cont = cont[:_i2+1] + graft + cont[_i1:]
            assert len(cont) == len2
        elif count2 < count1 and min_length <= len1:
            # solution 1 wins
            _j2 += _dj
            if _j2 == -1:
                _j2 = None
            graft = graft[_j1:_j2:_dj]
            if _i1 < _i2:
                cont = cont[:_i1+1] + graft + cont[_i2:]
            else:
                cont = graft + cont[_i2:_i1+1]
            assert len(cont) == len1
        else:
            if debug:
                print('skipping graft: conflict between solutions')
                print((count1, len1, count2, len2))
                _diagnose((cont, graft, [_inner1, _outer1], [_inner2, _outer2]), cells)
            continue
        any_inserted = True
    if fallback:
        return (cont, any_inserted)
    else:
        if any_inserted:
            return cont
        else:
            raise NoSolutionError


def _bracket_elements(string, adjacency, beacon=None, fullstring=None, outerlen=None, debug=False):
    if 2 < len(string):
        bracket = set()
        for n in string:
            neighbours = set(adjacency.indices[adjacency.indptr[n]:adjacency.indptr[n+1]])
            neighbours &= string
            if len(neighbours) == 1:
                bracket.add(n)
        if beacon and 2 < len(bracket):
            _bracket = set()
            for ns in beacon:
                if not isinstance(ns, (tuple, list, set)):
                    ns = [ns]
                _ns = set(bracket) # copy
                for n in ns:
                    _ns &= set(adjacency.indices[adjacency.indptr[n]:adjacency.indptr[n+1]])
                try:
                    _bracket.add(_ns.pop())
                except KeyError:
                    if debug:
                        print('bracketing: orphan beacon')
                    pass
                if _ns and debug:
                    print('bracketing: choosing between nodes at random')
            bracket = _bracket
    else:
        bracket = set(string) # copy
    return bracket


def _diagnose(paths, cells):
    if cells is None:
        return
    import matplotlib.pyplot as plt
    import tramway.plot.mesh as mesh
    f = plt.figure()
    mesh.plot_delaunay(cells, axes=f.gca())
    colors = 'bgkyr'
    for p, c in zip(paths, colors):
        x = cells.tesselation.cell_centers[p]
        plt.plot(x[:,0], x[:,1], c, linestyle='-', linewidth=3)
    plt.show()

