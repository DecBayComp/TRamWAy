# -*- coding: utf-8 -*-

# Copyright © 2017-2018, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import *
from tramway.tessellation.base import *
import numpy as np
import pandas as pd
import copy
import scipy.sparse as sparse


class NestedTessellations(Tessellation):
    """Tessellation of tessellations.

    When nesting, the parent tessellation should have been grown already.

    `tessellation` grows all the nested tessellations.

    In `__init__`, `tessellation` and `cell_index`,
    the `scaler` (if any), `args` (if any) and `kwargs` arguments
    only apply to the nested tessellations.
    """
    __slots__ = ('_parent', '_children', 'child_factory', \
        'parent_index_arguments', 'child_factory_arguments', \
        '_cell_centers')

    __lazy__ = Tessellation.__lazy__ + ('cell_label', 'cell_adjacency', 'adjacency_label', \
        'cell_centers')

    def __init__(self, scaler=None, parent=None, factory=None, parent_index_arguments={}, **kwargs):
        Tessellation.__init__(self, scaler)
        self._parent = parent
        self._children = {}
        self.child_factory = factory
        self.parent_index_arguments = parent_index_arguments
        self.child_factory_arguments = kwargs
        self._cell_centers = None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, tessellation):
        if tessellation != self._parent:
            self.children = {}
            self._parent = tessellation

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, tessellations):
        self.cell_label = None
        self.cell_adjacency = None
        self.adjacency_label = None
        self.cell_centers = None
        self._children = tessellations

    def _parent_index(self, points):
        if isinstance(self.parent, Partition):
            parent = self.parent
        else: # preferred usage
            parent = Partition(tessellation=self.parent)
        if points is not parent.points or \
                (self.parent_index_arguments and parent._cell_index is None):
            parent.points = points
            parent.cell_index = parent.tessellation.cell_index(points, \
                    **self.parent_index_arguments)
        parent_index = format_cell_index(parent.cell_index, 'pair') # ignore association weights
        pt_ids, cell_ids = parent_index
        if isinstance(points, pd.DataFrame):
            def rows(pts, ids):
                return pts.iloc[ids,:]
        else:
            def rows(pts, ids):
                return pts[ids]
        return (pt_ids, cell_ids, rows)

    def tessellate(self, points, *args, **kwargs):
        # initialize `self.scaler`;
        # if we didn't, we should pass copies of `self.scaler` instead, to `self.child_factory`
        any_child = self.child_factory(scaler=self.scaler, **self.child_factory_arguments)
        any_child._preprocess(points)
        #
        pt_ids, cell_ids, rows = self._parent_index(points)
        self.children = {}
        for u in np.unique(cell_ids):
            child = self.child_factory(scaler=self.scaler, **self.child_factory_arguments)
            child_pts = rows(points, pt_ids[cell_ids==u])
            if child_pts.size:
                child.tessellate(child_pts, *args, **kwargs)
                self.children[u] = child

    def cell_index(self, points, *args, **kwargs):
        point_count = points.shape[0]
        #if isinstance(points, pd.DataFrame):
        #       point_count = max(point_count, points.index.max()+1) # NO!
        # point indices are row indices and NOT rows labels
        parent_pt_ids, parent_cell_ids, rows = self._parent_index(points)
        _is_array_ = _is_pair_ = _is_sparse_ = False # (exclusive) type flags
        _first_ = True # initialization flag
        _type_error_ = TypeError('multiple nested partition types; `format` should be enforced')
        pt_cell, pt_ids, cell_ids = [], [], []
        cell_count = 0
        for u in self.children:
            child_cell_count = self.children[u].cell_adjacency.shape[0]
            child_pt_ids = parent_pt_ids[parent_cell_ids==u]
            if child_pt_ids.size: # if cell is empty
                child_partition = self.children[u].cell_index(
                    rows(points, child_pt_ids), *args, **kwargs)
                # TODO: test cases such that not any cell-point association is found;
                # currently, ``continue`` is not an option but
                # _first_ is made False anyway
                if sparse.issparse(child_partition):
                    if _first_:
                        _is_sparse_ = True
                        if args:
                            _format = args.pop(0)
                        elif 'format' in kwargs:
                            _format = kwargs['format']
                        else:
                            _format = None
                        kwargs['format'] = 'coo' # for the next calls to `cell_index`
                        if not sparse.isspmatrix_coo(child_partition):
                            child_partition = child_partition.tocoo()
                    elif not _is_sparse_:
                        raise _type_error_
                    # use `pt_ids` as `rows`, `cell_ids` as `cols` and `pt_cell` as `data`
                    _pt_cell = child_partition.data
                    _pt_ids = child_pt_ids[child_partition.rows]
                    _cell_ids = cell_count + child_partition.cols
                    pt_cell.append(_pt_cell)
                    pt_ids.append(_pt_ids)
                    cell_ids.append(_cell_ids)
                elif isinstance(child_partition, np.ndarray):
                    if cell_count:
                        child_partition[0<=child_partition] += cell_count
                    if _first_:
                        _is_array_ = True
                        pt_cell = np.full(point_count, -1,
                            dtype=child_partition.dtype)
                    else:
                        if not _is_array_:
                            raise _type_error_
                        offset = 0
                    pt_cell[child_pt_ids] = child_partition
                elif isinstance(child_partition, tuple):
                    _pt_ids, _cell_ids = child_partition
                    _pt_ids = child_pt_ids[_pt_ids]
                    _cell_ids += cell_count
                    if _first_:
                        _is_pair_ = True
                    elif not _is_pair_:
                        raise _type_error_
                    pt_ids.append(_pt_ids)
                    cell_ids.append(_cell_ids)
                else:
                    raise ValueError('partition type not supported')
                _first_ = False
            cell_count += child_cell_count
        if isinstance(pt_cell, list) and pt_cell:
            pt_cell = np.concatenate(pt_cell)
        if pt_ids:
            pt_ids = np.concatenate(pt_ids)
        if cell_ids:
            cell_ids = np.concatenate(cell_ids)
        if _is_sparse_:
            partition = sparse.coo_matrix((pt_cell, (pt_ids, cell_ids)),
                shape=(point_count, cell_count))
            if _format is None or _format in ['matrix', 'coo']:
                return partition
            else:
                return dict(
                        csc=partition.tocsc,
                        csr=partition.tocsr,
                        dia=partition.todia,
                        dok=partition.todok,
                        lil=partition.tolil,
                    )[_format]()
        elif _is_array_:
            return pt_cell
        elif _is_pair_:
            return (pt_ids, cell_ids)

    # cell_adjacency property
    @property
    def cell_adjacency(self):
        if self._cell_adjacency is None:
            _all_label_ = True
            _any_label_ = False
            labels = set()
            for child in self.children.values():
                if child.adjacency_label is None:
                    _all_label_ = False
                else:
                    _any_label_ = True
                    labels |= set(list(child.adjacency_label))
            if labels:
                if not _all_label_:
                    na = min(labels) - 1
                    labels.add(na)
                labels = list(labels)
                self._adjacency_label = np.array(labels)
                labels = dict(zip(labels, range(len(labels))))
            row, col, data = [], [], []
            cell_count = 0
            for u in self.children:
                A = self.children[u].cell_adjacency.tocoo()
                A.row += cell_count
                A.col += cell_count
                if self.children[u].adjacency_label is None:
                    if labels:
                        A.data = np.zeros_like(A.data, dtype=int)
                else:
                    A.data = np.array([ labels[e] for e in A.data ])
                row.append(A.row)
                col.append(A.col)
                data.append(A.data)
                cell_count += A.shape[0]
            row = np.concatenate(row)
            col = np.concatenate(col)
            data = np.concatenate(data)
            self._cell_adjacency = sparse.coo_matrix((data, (row, col)),
                shape=(cell_count, cell_count))
        return self.__returnlazy__('cell_adjacency', self._cell_adjacency)

    @cell_adjacency.setter
    def cell_adjacency(self, matrix):
        self.__lazysetter__(matrix)

    # cell_label
    @property
    def cell_label(self):
        if self._cell_label is None:
            _err_ = AttributeError('not all the nested tessellations have cell labels')
            _missing_label_ = False
            labels = []
            for u in self.children:
                child_labels = self.children[u].cell_label
                if child_labels:
                    if _missing_label_:
                        raise _err_
                    labels.append(child_labels)
                else:
                    if labels:
                        raise _err_
                    _missing_label_ = True
            if _missing_label_:
                self._cell_label = False
            else:
                self._cell_label = np.concatenate(labels)
        if self._cell_label is False:
            return None
        else:
            return self.__returnlazy__('cell_label', self._cell_label)

    @cell_label.setter
    def cell_label(self, label):
        self.__lazysetter__(label)

    # adjacency_label
    @property
    def adjacency_label(self):
        if self._adjacency_label is None:
            # this doesn't cost anything if `_cell_adjacency` has already been built
            self.cell_adjacency # makes `_adjacency_labels`
        return self.__returnlazy__('adjacency_label', self._adjacency_label)

    @adjacency_label.setter
    def adjacency_label(self, label):
        self.__lazysetter__(label)

    def freeze(self):
        for child in self.children.values():
            child.freeze()

    @property
    def cell_centers(self):
        if self._cell_centers is None:
            try:
                centers = []
                for child in self.children.values():
                    centers.append(child.cell_centers)
            except AttributeError:
                raise AttributeError("'NestedTessellations' object has no attribute 'cell_centers'")
            else:
                # note that a dimension may not represent the same variable
                # from a child to another
                self._cell_centers = np.vstack(centers)
        return self.__returnlazy__('cell_centers', self._cell_centers)

    @cell_centers.setter
    def cell_centers(self, centers):
        self.__lazyassert__(centers)

    def child_cell_indices(self, u):
        """
        Arguments:

            u (int): child index.

        Returns:

            slice: child's cell indices.
        """
        cell_count = 0
        for v in self.children:
            child = self.children[v]
            child_cell_count = child.cell_adjacency.shape[0]
            if u == v:
                return slice(cell_count, cell_count + child_cell_count)
            cell_count += child_cell_count
        raise IndexError('no such child index: {}'.format(u))



__all__ = ['NestedTessellations']

