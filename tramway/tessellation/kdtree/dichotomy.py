# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import *
import numpy as np
from threading import Lock
import itertools


class Dichotomy(object):
    def __init__(self, points=None, min_edge=None, base_edge=None, \
        min_depth=None, max_depth=50, max_level=None, \
        min_count=1, max_count=None, \
        origin=None, center=None, lower_bound=None, upper_bound=None, \
        subset=None, cell=None):
        self.subset_lock = Lock()
        self.cell_lock = Lock()
        self.base_edge = base_edge
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.origin = origin
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.min_count = min_count
        self.max_count = max_count
        try:
            if self.max_count is None:
                self.max_count = points.shape[0]
            if self.lower_bound is None:
                self.lower_bound = points.min(axis=0)
            if self.upper_bound is None:
                self.upper_bound = points.max(axis=0)
        except AttributeError:
            raise AttributeError('missing `points` input argument')
        if center is None:
            center = (self.lower_bound + self.upper_bound) / 2.0
        # define `max_edge`; `max_edge` is level-1 edge, not level 0, hence +1 in `max_depth`
        max_edge = np.max(np.maximum(self.upper_bound - center, center - self.lower_bound))
        if self.origin is None: # `center` cannot be `None`
            self.origin = center - max_edge
        if self.base_edge is None: # `max_edge` is defined
            if min_edge:
                self.max_depth = int(ceil(log(max_edge / min_edge, 2))) + 1
            self.base_edge = max_edge * 2.0 ** (1 - self.max_depth)
        else:
            if min_edge:
                raise AttributeError('`min_edge` ignored') # should be a warning instead
            self.max_depth = int(ceil(log(max_edge / self.base_edge, 2))) + 1
        if self.min_depth is None:
            if max_level is None:
                self.min_depth = 0
            else:
                self.min_depth = self.max_depth - max_level
        dim = self.origin.size
        grid = [(0, 1)] * dim
        grid = np.meshgrid(*grid, indexing='ij')
        self.unit_hypercube = np.stack([ g.flatten() for g in grid ], axis=-1)
        # `min_edge` is defined
        self.reference_length = self.base_edge * 2.0 ** np.arange(self.max_depth, -2, -1)
        if subset is None:
            if points is None:
                self.subset = {}
                self.subset_counter = 0
            else:
                self.subset = {0: (np.arange(points.shape[0]), points)}
                self.subset_counter = 1
        else:
            self.subset = subset
            if subset:
                self.subset_counter = max(subset.keys()) + 1
            else:
                self.subset_counter = 0
        if cell is None:
            self.cell = dict()
            self.cell_counter = 0
        else:
            self.cell = cell
            if cell:
                self.cell_counter = max(cell.keys()) + 1
            else:
                self.cell_counter = 0


    def split(self, points=None):
        if points is None:
            if self.subset_counter == 0:
                raise AttributeError('missing `points` input argument')
        else:
            self.subset = {0: (np.arange(points.shape[0]), points)}
            self.subset_counter = 1
            self.cell = dict()
            self.cell_counter = 0
        self._split(0, self.origin, 0)

    def _split(self, ss_ref, origin, depth):
        indices, points = self.subset[ss_ref]
        ok = depth < self.max_depth and self.min_count < points.shape[0]
        depth += 1
        ok = ok or depth < self.min_depth
        if ok:
            # split and check that every subarea has at least min_count points
            ss_refs = dict()
            lower = dict()
            for i, step in enumerate(self.unit_hypercube): # could be parallelized
                lower[i] = origin + np.float_(step) * self.reference_length[depth] # new origin
                upper = lower[i] + self.reference_length[depth] # opposite hypercube vertex
                self.subset_lock.acquire()
                r = self.subset_counter
                self.subset_counter += 1
                self.subset_lock.release()
                subset = np.all(np.logical_and(lower[i] < points, \
                    points < upper), axis=1)
                self.subset[r] = (indices[subset], points[subset])
                ss_refs[i] = r # dict for future parallelization
            if self.min_depth <= depth:
                counts = [ self.subset[r][0].size for r in ss_refs.values() ]
                ok = any([ self.max_count < c for c in counts ]) or \
                    all([ self.min_count < c for c in counts ])
            if ok:
                del self.subset[ss_ref] # save memory
            else:
                for r in ss_refs.values():
                    del self.subset[r]
        if ok:
            return self.do_split(ss_refs, lower, depth)
        else:
            depth -= 1
            self.cell_lock.acquire()
            i = self.cell_counter
            self.cell_counter += 1
            self.cell_lock.release()
            self.cell[i] = (origin, depth, ss_ref)
            return self.dont_split(i, ss_ref, origin, depth)

    def do_split(self, ss_ref, origin, depth):
        for i in ss_ref:
            self._split(ss_ref[i], origin[i], depth)

    def dont_split(self, cell_ref, ss_ref, origin, depth):
        pass



def _face_hash(v1, n1):
    '''key that identify a same face.'''
    return tuple(v1) + tuple(n1)


class ConnectedDichotomy(Dichotomy):
    def __init__(self, *args, **kwargs):
        adjacency = kwargs.pop('adjacency', None)
        Dichotomy.__init__(self, *args, **kwargs)
        self.adjacency_lock = Lock()
        dim = self.unit_hypercube.shape[1]
        eye = np.eye(dim, dtype=int)
        self.face_normal = np.vstack((eye[::-1], eye))
        self.face_vertex = np.vstack((np.zeros_like(eye), eye))
        def onface(vertex, face_vertex, face_normal):
            return np.sum(face_normal * (face_vertex - vertex), axis=1, keepdims=True) == 0
        self.exterior = np.concatenate([ onface(self.unit_hypercube, v, n) \
            for v, n in zip(self.face_vertex, self.face_normal) ], axis=1)
        if adjacency is None:
            self.adjacency = dict()
            self.edge_counter = 0
        else:
            self.adjacency = adjacency
            self.edge_counter = max(adjacency.keys()) + 1

    def merge_Faces(self, face1, face2, axis):
        #adjacency_old = dict(self.adjacency) # for debugging at the end of the method's body
        dim = self.unit_hypercube.shape[1]
        dims = np.logical_not(axis)
        cell1 = [ (c,) + self.cell[c] for c in face1 ]
        cell2 = [ (c,) + self.cell[c] for c in face2 ]
        #print((cell1, cell2))
        for c1, a1, l1, _ in cell1:
            a1 = a1[dims]
            for c2, a2, l2, _ in cell2:
                a2 = a2[dims]
                if l1 < l2:
                    origin, diagonal = a1, self.reference_length[l1]
                    center = a2 + .5 * self.reference_length[l2]
                else:
                    origin, diagonal = a2, self.reference_length[l2]
                    center = a1 + .5 * self.reference_length[l1]
                # is `center` inside the (origin, diagonal) hypercube?
                rel = center - origin
                if np.all(0 < rel) and np.sqrt(np.dot(rel, rel)) <= diagonal:
                    # connect cell1 with cell2
                    dk = 2 ** ((dim - 1) * abs(l1 - l2))
                    self.adjacency_lock.acquire()
                    k = self.edge_counter
                    self.edge_counter += dk
                    self.adjacency_lock.release()
                    self.adjacency[k] = (c1, c2)
        ## debug
        #self.plot_delaunay()

    def plot_delaunay(self):
        import matplotlib.pyplot
        plt  = matplotlib.pyplot
        x, x0 = [], np.full(2, np.nan, dtype=float)
        for k in self.adjacency:
            c1, c2 = self.adjacency[k]
            x1, l1, _ = self.cell[c1]
            x2, l2, _ = self.cell[c2]
            x1 = x1 + self.reference_length[l1 + 1]
            x2 = x2 + self.reference_length[l2 + 1]
            x.append(x0)
            x.append(x1)
            x.append(x2)
        x = np.stack(x, axis=0)
        plt.figure()
        plt.plot(x[:,0], x[:,1], 'r-')
        plt.axis('equal')
        plt.show()

    def do_split(self, ss_ref, origin, depth):
        interior_cells = dict()
        exterior_cells = dict()
        # split
        for i, step in enumerate(self.unit_hypercube):
            cells = self._split(ss_ref[i], origin[i], depth)
            # factorize interior/exterior
            for c, cs in enumerate(cells):
                if self.exterior[i, c]:
                    Cs = exterior_cells.get(c, [])
                    Cs.append(cs)
                    exterior_cells[c] = Cs
                else:
                    j = _face_hash(step + self.face_vertex[c], \
                        self.face_normal[c])
                    if j in interior_cells:
                        self.merge_Faces(interior_cells[j], cs, \
                            self.face_normal[c])
                    else:
                        interior_cells[j] = cs
        for c in exterior_cells:
            exterior_cells[c] = list(itertools.chain(*exterior_cells[c]))
        dim = self.unit_hypercube.shape[1]
        return [ exterior_cells[c] for c in range(2 * dim) ]

    def dont_split(self, cell_ref, ss_ref, origin, depth):
        dim = self.unit_hypercube.shape[1]
        return [ [cell_ref] for _ in range(2 * dim) ]


