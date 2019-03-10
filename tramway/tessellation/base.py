# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import *
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import scipy.sparse as sparse
from scipy.sparse import issparse
import scipy.spatial as spatial
from tramway.core import *
import itertools
import copy
from collections import Counter, namedtuple
import sys


class CellStats(Lazy):
    """Container datatype for a point dataset together with a tessellation.

    A `CellStats` instance conveniently stores the tessellation (:attr:`tessellation`) and the
    partition of the data (:attr:`cell_index`) together with the data itself (:attr:`points`) and
    a few more intermediate results frequently derivated from a data partition.

    :attr:`locations` and :attr:`translocations` are aliases of :attr:`points`.
    No control is performed on whether :attr:`translocations` are actual translocations for example.

    The partition :attr:`cell_index` may be in any of the following formats:

    array
        Cell index of size the number of data points. The element at index ``i`` is the cell
        index of the ``i`` th point or ``-1`` if the ``i`` th point is not assigned to any cell.

    pair of arrays
        Point-cell association in the shape of a sparse representation
        ``(point_index, cell_index)`` such that for all ``i`` the ``point_index[i]`` point is
        in the ``cell_index[i]`` cell.

    sparse matrix (:mod:`scipy.sparse`)
        ``number_of_points * number_of_cells`` matrix with nonzero element wherever
        the corresponding point is in the corresponding cell.


    .. note::

        If the point coordinates are defined as a :class:`~pandas.DataFrame`,
        point indices are row indices and NOT row labels (see also :attr:`~pandas.DataFrame.iloc`).


    See also :meth:`Tessellation.cell_index`.


    Attributes:

        points (array-like):
            the original (trans-)location coordinates, unchanged.

        tessellation (Tessellation):
            The tessellation that defined the partition.

        cell_index (numpy.ndarray or pair of arrays or sparse matrix):
            Point-cell association (or data partition).

        location_count (numpy.ndarray, lazy):
            point count per cell; ``location_count[i]`` is the number of
            points in cell ``i``.

        bounding_box (array-like, lazy):
            ``2 * D`` array with lower values in first row and upper values in second row,
            where ``D`` is the dimension of the point data.

        param (dict):
            Arguments involved in the tessellation and the partition steps, as key-value
            pairs. Such information is maintained in :class:`~tramway.tessellation.CellStats`
            so that it can be stored in *.rwa* files and retrieve for traceability.

    Functional dependencies:

    * setting `tessellation` unsets `cell_index`
    * setting `points` unsets `cell_index` and `bounding_box`
    * setting `cell_index` unsets `location_count`

    """

    __slots__ = ('_points', '_cell_index', '_location_count', '_bounding_box', 'param', '_tessellation')
    __lazy__ = ('cell_index', 'location_count', 'bounding_box')

    def __init__(self, points=None, tessellation=None, cell_index=None, location_count=None, \
        bounding_box=None, param={}, locations=None, translocations=None):
        Lazy.__init__(self)
        exclusivity_violation = ValueError('arguments `points`, `locations` and `translocations` are mutually exclusive')
        if points is None:
            if locations is None:
                if translocations is None:
                    pass
                else:
                    points = translocations
            elif translocations is None:
                points = locations
            else:
                raise exclusivity_violation
        elif locations is None and translocations is None:
            pass
        else:
            raise exclusivity_violation
        self._points = points
        self.cell_index = cell_index
        self._location_count = location_count
        self._bounding_box = bounding_box
        self.param = param
        self._tessellation = tessellation

    @property
    def cell_index(self):
        if self._cell_index is None:
            self._cell_index = self.tessellation.cell_index(self.points)
        return self.__returnlazy__('cell_index', self._cell_index)

    @cell_index.setter
    def cell_index(self, index):
        self.__lazysetter__(index)
        self.location_count = None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, pts):
        self._points = pts
        self.cell_index = None
        self.bounding_box = None

    @property
    def locations(self):
        return self.points

    @locations.setter
    def locations(self, pts):
        self.points = pts

    @property
    def translocations(self):
        return self.points

    @translocations.setter
    def translocations(self, pts):
        self.points = pts

    @property
    def tessellation(self):
        return self._tessellation

    @tessellation.setter
    def tessellation(self, mesh):
        self._tessellation = mesh
        self.cell_index = None

    # property for the backward compatibility trick in :mod:`~tramway.core.hdf5`
    @property
    def _tesselation(self):
        return

    @_tesselation.setter
    def _tesselation(self, mesh):
        if mesh is not None:
            self._tessellation = mesh

    def descriptors(self, *vargs, **kwargs):
        """Proxy method for :meth:`Tessellation.descriptors`."""
        return self.tessellation.descriptors(*vargs, **kwargs)

    @property
    def location_count(self):
        if self._location_count is None:
            try:
                ncells = self.tessellation.cell_adjacency.shape[0]
            except AttributeError: # Delaunay?
                ncells = self.tessellation._cell_centers.shape[0]
            if isinstance(self.cell_index, tuple):
                _point, _cell = self.cell_index
                if np.any(_cell < 0):
                    #import warnings
                    #warnings.warn('point-cell association pair contains invalid assignments')
                    ok = 0 <= _cell
                    _point, _cell = _point[ok], _cell[ok]
                ci = sparse.csc_matrix(
                    (np.ones_like(_point), (_point, _cell)),
                    shape=(self.points.shape[0], ncells))
                self._location_count = np.diff(ci.indptr)
            elif sparse.issparse(self.cell_index):
                self._location_count = np.diff(self.cell_index.tocsc().indptr)
            else:
                valid_cells, _location_count = np.unique(self.cell_index,
                    return_counts=True)
                _location_count = _location_count[0 <= valid_cells]
                valid_cells = valid_cells[0 <= valid_cells]
                self._location_count = np.zeros(ncells, dtype=_location_count.dtype)
                self._location_count[valid_cells] = _location_count
            assert self._location_count.size == ncells
        return self.__returnlazy__('location_count', self._location_count)

    @location_count.setter
    def location_count(self, cc):
        self.__lazysetter__(cc)

    @property
    def bounding_box(self):
        if self._bounding_box is None:
            xmin = self.points.min(axis=0)
            xmax = self.points.max(axis=0)
            if isinstance(self.points, pd.DataFrame):
                self._bounding_box = pd.concat([xmin, xmax], axis=1).T
                self._bounding_box.index = ['min', 'max']
            else:
                self._bounding_box = np.vstack([xmin, xmax]).flatten('F')
        return self.__returnlazy__('bounding_box', self._bounding_box)

    @bounding_box.setter
    def bounding_box(self, bb):
        self.__lazysetter__(bb)

    def freeze(self):
        """
        Proxy method for :meth:`Tessellation.freeze`.
        """
        if self.tessellation is not None:
            self.tessellation.freeze()

    @property
    def number_of_cells(self):
        """
        Proxy property for `Tessellation.number_of_cells`.
        """
        if self.tessellation is not None:
            return self.tessellation.number_of_cells

    def __str__(self):
        def _str(obj, l0=0):
            s = str(obj)
            if l0:
                s = s.replace('\n', '\n' + ' ' * l0)
            return s
        def print_kwargs(_dict, l0=0, l1=None):
            if l1 is None:
                l1 = max(len(k) for k in _dict)
            sep = '\n'
            if l0:
                sep += ' ' * l0
            return sep.join([ '{}:{} {}'.format(k, ' '*(l1-len(k)), _str(v, l0+l1))
                    for k, v in _dict.items() ])
        attrs = {}
        for k in ('tessellation', 'points', 'cell_index', 'location_count'):
            v = getattr(self, '_'+k)
            attrs[k] = None if v is None else type(v)
        attrs['number_of_cells'] = self.number_of_cells
        l = max(len(k) for k in attrs)
        if self.param:
            l = max(l, 1 + max(len(k) for k in self.param))
        l += 1 # for ':'
        l0 = l + 1 # for ' '
        attrs['bounding_box'] = None if self._bounding_box is None \
            else _str(self.bounding_box, l0)
        if self.param:
            l1 = max( max(len(k) for k in _attrs) for _attrs in self.param.values() \
                    if isinstance(_attrs, dict) )
            for k, v in self.param.items():
                if isinstance(v, dict):
                    v = print_kwargs(v, l0, l1)
                else:
                    v = _str(v, l0)
                attrs['@'+k] = v
        try:
            # handle child classes with __dict__ defined
            for k, v in self.__dict__.items():
                if k not in attrs:
                    attrs[k] = _str(v, l0)
        except AttributeError:
            pass
        s = '\n'.join([ '{}:{}{}'.format(k, ' '*(l-len(k)), v) for k, v in attrs.items() ])
        return s



def format_cell_index(K, format=None, select=None, shape=None, copy=False, **kwargs):
    """
    Convert from any valid index format to any other.

    Converting an *array* index to any other format assumes that the point indices are in a
    contiguous range from 0 to the number of elements in the index.

    Arguments:

        K (any): original point-cell association representation.

        format (str): either *array*, *pair*, *matrix*, *coo*, *csr* or *csc*.
            See also :meth:`Tessellation.cell_index`.

        select (callable): called only if ``format == 'array'`` and points are
            associated to multiple cells; `select` takes the point index
            as first argument, the corresponding cell indices (:class:`numpy.ndarray`)
            as second argument and the extra keyword arguments given to
            :func:`format_cell_index`.

        shape (int, int): number of points, number of cells.

        copy (bool): if ``True``, ensures that a copy of `K` is returned if `K`
            is already in the requested format.

    Returns:

        any: point-cell association in the requested format.

    See also :meth:`Tessellation.cell_index` and :func:`nearest_cell`.
    """
    if isinstance(K, np.ndarray) and format not in [None, 'array']:
        I, = np.nonzero(0 <= K)
        K = (I, K[I])
        copy = False # already done
    if format in ['matrix', 'coo', 'csr', 'csc']:
        if issparse(K):
            if format == 'coo':
                K = K.tocoo()
                copy = False # already done
        else:
            K = sparse.coo_matrix((np.ones_like(K[0], dtype=bool), K), shape=shape)
            copy = False # already done
        if format == 'csr':
            K = K.tocsr()
            copy = False # already done
        elif format == 'csc':
            K = K.tocsc()
            copy = False # already done
    elif issparse(K):
        K = K.tocoo()
        K = (K.row, K.col) # drop the values; keep only the indices
        copy = False # already done
    if format == 'array' and isinstance(K, tuple):
        points, cells = K
        K = np.full(shape[0], -1, dtype=int)
        P, I, N = np.unique(points, return_index=True, return_counts=True)
        K[P[N==1]] = cells[I[N==1]] # unambiguous assignments
        P = P[1<N] # ambiguous assignments
        for p in P:
            cs = cells[points == p]
            if 1 < cs.size:
                K[p] = select(p, cs, **kwargs)
            else:
                K[p] = cs
        copy = False # already done
    if copy:
        K = copy.copy(K)
    return K


def nearest_cell(locations, cell_centers):
    """
    Generate a function suitable for use as
    :func:`format_cell_index`'s argument `select`.

    The returned function takes a point index and cell indices as arguments
    and returns the index of the nearest cell.

    Arguments:

        locations (numpy.ndarray): location coordinates.

        cell_centers (numpy.ndarray): cell center coordinates.

    Returns:

        callable: `select` function.
    """
    def f(point, cells):
        x = locations[point]
        y = cell_centers[cells]
        z = y - x
        square_dist = np.sum(z * z, axis=1)
        winner = np.argmin(square_dist)
        return cells[winner]
    return f



def point_adjacency_matrix(cells, symetric=True, cell_labels=None, adjacency_labels=None):
    """
    Adjacency matrix of data points such that a given pair of points is defined as
    adjacent iif they belong to adjacent and distinct cells.

    Arguments:

        cells (CellStats):
            CellStats with both partition and tessellation defined.

        symetric (bool):
            If ``False``, the returned matrix will not be symetric, i.e. wherever i->j is
            defined, j->i is not.

        cell_labels (callable):
            Takes an array of cell labels as input
            (see :attr:`Tessellation.cell_label`)
            and returns a bool array of equal shape.

        adjacency_labels (callable):
            Takes an array of edge labels as input
            (see :attr:`Tessellation.adjacency_label`)
            and returns a bool array of equal shape.

    Returns:

        scipy.sparse.csr_matrix:
            Sparse square matrix with as many rows as data points.

    """
    if not isinstance(cells.cell_index, np.ndarray):
        raise NotImplementedError('cell overlap support has not been implemented here')
    x = cells.descriptors(cells.points, asarray=True)
    ij = np.arange(x.shape[0])
    x2 = np.sum(x * x, axis=1)
    x2.shape = (x2.size, 1)
    I = []
    J = []
    D = []
    n = []
    for i in np.arange(cells.tessellation.cell_adjacency.shape[0]):
        if cell_labels is not None and not cell_labels(cells.tessellation.cell_label[i]):
            continue
        _, js, k = sparse.find(cells.tessellation.cell_adjacency[i])
        if js.size == 0:
            continue
        # the upper triangular part of the adjacency matrix should be defined...
        k = k[i < js]
        js = js[i < js]
        if js.size == 0:
            continue
        if adjacency_labels is not None:
            if cells.tessellation.adjacency_label is not None:
                k = cells.tessellation.adjacency_label
            js = js[adjacency_labels(k)]
            if js.size == 0:
                continue
        if cell_labels is not None:
            js = js[cell_labels(cells.tessellation.cell_label[js])]
            if js.size == 0:
                continue
        ii = ij[cells.cell_index == i]
        xi = x[cells.cell_index == i]
        x2i = x2[cells.cell_index == i]
        for j in js:
            xj = x[cells.cell_index == j]
            x2j = x2[cells.cell_index == j]
            d2 = x2i + x2j.T - 2 * np.dot(xi, xj.T)
            jj = ij[cells.cell_index == j]
            i2, j2 = np.meshgrid(ii, jj, indexing='ij')
            I.append(i2.flatten())
            J.append(j2.flatten())
            D.append(d2.flatten())
            if symetric:
                I.append(j2.flatten())
                J.append(i2.flatten())
                D.append(d2.flatten())
    I = np.concatenate(I)
    J = np.concatenate(J)
    D = np.sqrt(np.concatenate(D))
    n = cells.points.shape[0]
    return sparse.csr_matrix((D, (I, J)), shape=(n, n))



class Tessellation(Lazy):
    """Abstract class for tessellations.

    The methods to be implemented are :meth:`tessellate` and :meth:`cell_index`.

    Attributes:
        scaler (tramway.core.scaler.Scaler): scaler.

        cell_adjacency (sparse matrix):
            square adjacency matrix for cells.
            If :attr:`_adjacency_label` is defined, :attr:`_cell_adjacency` should be
            sparse and the explicit elements should be indices in :attr:`_adjacency_label`.

        cell_label (numpy.ndarray):
            cell labels with as many elements as cells.

        adjacency_label (numpy.ndarray):
            inter-cell edge labels with as many elements as there are edges.
    """
    __slots__ = ('scaler', '_cell_adjacency', '_cell_label', '_adjacency_label')

    def __init__(self, scaler=None):
        """
        Arguments:
            scaler (tramway.core.scaler.Scaler): scaler.
        """
        Lazy.__init__(self)
        if scaler is None:
            self.scaler = Scaler()
        else:
            self.scaler = scaler
        self._cell_adjacency = None
        self._cell_label = None
        self._adjacency_label = None

    def _preprocess(self, points):
        """
        Identify euclidean variables (usually called *x*, *y*, *z*) and scale the coordinates.

        See also:
            :mod:`tramway.core.scaler`.
        """
        if self.scaler.euclidean is None:
            # initialize
            if isstructured(points):
                self.scaler.euclidean = ['x', 'y']
                if not ('x' in points and 'y' in points): # enforce presence of 'x' and 'y'
                    raise AttributeError('missing ''x'' or ''y'' in input dataframe.')
                if 'z' in points:
                    self.scaler.euclidean.append('z')
            else:   self.scaler.euclidean = np.arange(0, points.shape[1])
        return self.scaler.scale_point(points)

    def tessellate(self, points, **kwargs):
        """
        Grow the tessellation.

        Arguments:
            points (pandas.DataFrame): point coordinates.

        Admits keyword arguments.
        """
        raise NotImplementedError

    def cell_index(self, points, format=None, select=None, **kwargs):
        """
        Partition.

        The returned value depends on the `format` input argument:

        * *array*: returns a vector ``v`` such that ``v[i]`` is cell index for
            point index ``i`` or ``-1``.

        * *pair*: returns a pair of ``I``-sized arrays ``(p, c)`` where, for each
            point-cell association ``i`` in ``range(I)``, ``p[i]`` is a point index
            and ``c[i]`` is a corresponding cell index.

        * *matrix* or *coo* or *csr* or *csc*:
            returns a :mod:`~scipy.sparse` matrix with points as rows and
            cells as columns; non-zeros are all ``True`` or float weights.

        By default with `format` undefined, any implementation may favor any format.

        Note that *array* may not be an acceptable format and :meth:`cell_index` may
        not comply with ``format='index'`` unless `select` is defined.
        When a location or a translocation is associated to several cells, `select`
        chooses a single cell among them.

        The default implementation calls :func:`format_cell_index` on the result of an
        abstract `_cell_index` method that any :class:`Tessellation` implementation can
        implement instead of :meth:`cell_index`.

        See also :func:`format_cell_index`.

        Arguments:
            points (pandas.DataFrame): point (location) coordinates.

            format (str): preferred representation of the point-cell association
                (or partition).

            select (callable): takes the point index, an array of cell indices and the
                tessellation as arguments, and returns a cell index or ``-1`` for no cell.

        """
        point_count = len(points)
        #if isinstance(points, pd.DataFrame):
        #       point_count = max(point_count, points.index.max()+1) # NO!
        # point indices are row indices and NOT rows labels
        return format_cell_index(self._cell_index(points, **kwargs), format=format, select=select,
            shape=(point_count, self.cell_adjacency.shape[0]))

    # cell_label property
    @property
    def cell_label(self):
        """Cell labels, :class:`numpy.ndarray` with as many elements as there are cells."""
        return self._cell_label

    @cell_label.setter
    def cell_label(self, label):
        self._cell_label = label

    # cell_adjacency property
    @property
    def cell_adjacency(self):
        """Square cell adjacency matrix. If :attr:`adjacency_label` is defined,
        :attr:`cell_adjacency` is sparse and the explicit elements are indices in
        :attr:`adjacency_label`."""
        return self._cell_adjacency

    @cell_adjacency.setter
    def cell_adjacency(self, matrix):
        self._cell_adjacency = matrix

    # adjacency_label property
    @property
    def adjacency_label(self):
        """Inter-cell edge labels, :class:`numpy.ndarray` with as many elements as edges."""
        return self._adjacency_label

    @adjacency_label.setter
    def adjacency_label(self, label):
        self._adjacency_label = label

    def simplified_adjacency(self, adjacency=None, label=None, format='coo'):
        """
        Simplified copy of :attr:`cell_adjacency` as a :class:`scipy.sparse.spmatrix` sparse
        matrix with no explicit zeros.

        Non-zero values indicate adjacency and all these values are strictly positive.

        In addition, cells with negative (or null) labels are also disconnected from their
        neighbours.

        Labels are `cell_label` by default. Alternative labels can be provided as `label`.

        To prevent label-based disconnection, set `label` to ``False``.

        Multiple arrays of labels can also be supplied as a tuple.
        Note that explicit labels always supersede `cell_label` and the later should be
        explicitely listed in the tuple so that it is applied in combination with other
        label arrays.

        Arguments:

            adjacency (scipy.sparse.spmatrix): adjacency matrix (`cell_adjacency` is used
                if `adjacency` is ``None``).

            label (bool or array-like): cell labels.

            format (str): any of *'coo'*, *'csr'* and *'csc'*.

        Returns:

            scipy.sparse.spmatrix: simplified adjacency matrix.

        """
        if adjacency is None:
            adjacency = self.cell_adjacency
        if label is False:
            pass
        elif label is True: # `cell_label` is required (cannot be None)
            label = (self.cell_label, )
        elif label is None: # `cell_label` can be None
            if self.cell_label is not None:
                label = (self.cell_label, )
        elif not isinstance(label, tuple):
            label = (label, )
        adjacency = adjacency.tocoo()
        if self.adjacency_label is None:
            ok = 0 < adjacency.data
        else:
            ok = 0 < self.adjacency_label[adjacency.data]
        row, col = adjacency.row[ok], adjacency.col[ok]
        if label:
            edges_not_ok = np.zeros(row.size, dtype=bool)
            for cell_label in label:
                cells_not_ok = cell_label <= 0
                edges_not_ok[cells_not_ok[row]] = True
                edges_not_ok[cells_not_ok[col]] = True
            edges_ok = np.logical_not(edges_not_ok)
            row, col = row[edges_ok], col[edges_ok]
        data = np.ones(row.size, dtype=bool)
        matrix = dict(coo=sparse.coo_matrix, csc=sparse.csc_matrix, csr=sparse.csr_matrix)
        return matrix[format]((data, (row, col)), shape=adjacency.shape)

    def descriptors(self, points, asarray=False):
        """Keep the data columns that were involved in growing the tessellation.

        Arguments:
            points (pandas.DataFrame): point coordinates.
            asarray (bool): returns a :class:`numpy.ndarray`.

        Returns:
            array-like: selected coordinates; the data may not be copied.

        See also:
            :meth:`tramway.core.scaler.Scaler.scaled`.
        """
        return self.scaler.scaled(points, asarray)
        try:
            return self.scaler.scaled(points, asarray)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            if asarray:
                return np.asarray(points)
            else:
                return points

    @property
    def number_of_cells(self):
        """
        Number of cells.

        Read-only property.
        """
        return self.cell_adjacency.shape[0]

    def neighbours(self, i):
        """
        Indices of neighbour cells.

        Arguments:
            i (int): cell index.

        Returns:
            numpy.ndarray: indices of the neighbour cells of cell *i*.
        """
        A = self.cell_adjacency.tocsr()
        return A.indices[A.indptr[i]:A.indptr[i+1]]

    def contour(self, cell, distance=1, fallback=False, adjacency=None, **kwargs):
        """
        Select a close path around a cell.

        This method may be moved out of `Tessellation` in the near future.
        """
        import tramway.feature.adjacency as feature
        if adjacency is None:
            adjacency = self.simplified_adjacency().tocsr()
        return feature.contour(cell, adjacency, distance, None, fallback=fallback, **kwargs)

    def freeze(self):
        """
        Delete the data required to and only to incrementally update the tessellation.

        This may save large amounts of memory but differs from
        :meth:`~tramway.core.lazy.Lazy.unload` in that the subsequent data loss may not
        be undone.
        """
        pass



class Delaunay(Tessellation):
    """
    Delaunay graph.

    A cell is represented by a centroid and an edge of the graph represents a neighour relationship
    between two cells.

    :class:`Delaunay` implements the nearest neighour feature and support for cell overlap.

    Attributes:
        cell_centers (numpy.ndarray): coordinates of the cell centers.
    """
    __slots__ = ('_cell_centers',)

    def __init__(self, scaler=None):
        Tessellation.__init__(self, scaler)
        self._cell_centers = None

    def tessellate(self, points):
        self._cell_centers = np.asarray(self._preprocess(points))

    def cell_index(self, points, format=None, select=None, knn=None, radius=None,
        min_location_count=None, metric='euclidean', filter=None,
        filter_descriptors_only=False, **kwargs):
        """
        See :meth:`Tessellation.cell_index`.

        A single array representation of the point-cell association may not be possible with
        `knn` defined, because a point can be associated to multiple cells. If such
        a case happens the default output format will be *pair*.

        In addition to the values allowed by :meth:`Tessellation.cell_index`, `format` admits
        value *force array* that acts like ``format='array', select=nearest_cell(...)``.
        The implementation however is more straight-forward and simply ignores
        the minimum number of nearest neighbours if provided.

        Arguments:
            points: see :meth:`Tessellation.cell_index`.
            format: see :meth:`Tessellation.cell_index`; additionally admits *force array*.
            select: see :meth:`Tessellation.cell_index`.
            knn (int or tuple or callable):
                If `int`: minimum number of points per cell (or of nearest neighours to the cell
                center). Cells may overlap and the returned cell index may be a sparse
                point-cell association.
                If `tuple` (pair of ints): minimum and maximum number of points per cell
                respectively.
                If `callable`: takes a cell index and returns the minimum and maximum
                number of points.
            radius (float or tuple or callable)
                If `float`: distance from the cell center; smaller cells may include
                locations from neighbour cells and larger cells may include only part of
                their associated locations.
                If `tuple` (pair of floats): minimum and maximum radius of a cell
                respectively; any of these values can be None.
                If `callable`: takes a cell index and returns the minimum and maximum
                radius.
            min_location_count (int):
                minimum number of points for a cell to be included in the labeling.
                This argument applies before `knn`. The points in these cells, if not
                associated with another cell, are labeled ``-1``. The other cell labels
                do not change.
            metric (str): any metric name understandable by :func:`~scipy.spatial.distance.cdist`.
            filter (callable): takes the calling instance, a cell index and the corresponding
                subset of points; returns ``True`` if the corresponding cell should be
                included in the labeling.
            filter_descriptors_only (bool): whether `filter` should get points as
                descriptors only.

        Returns:
            see :meth:`Tessellation.cell_index`.

        """
        if self._cell_centers.size == 0:
            return format_cell_index(np.full(len(points), -1, dtype=int), format=format)
        if callable(knn):
            min_nn = max_nn = True
        elif isinstance(knn, tuple):
            min_nn, max_nn = knn
            if min_nn is not None and max_nn is not None and max_nn < min_nn:
                raise ValueError('min_nearest_neighbours > max_nearest_neighbours')
        else:
            min_nn, max_nn = knn, None
        if callable(radius):
            min_r = max_r = True
        elif isinstance(radius, tuple):
            min_r, max_r = radius
            if min_r is not None and max_r is not None and max_r < min_r:
                raise ValueError('min_radius > max_radius')
        else:
            min_r = max_r = radius
            if radius and not(min_nn or max_nn or min_location_count or filter):
                return cell_index_by_radius(self, points, radius,
                        format=format, select=select, metric=metric, **kwargs)
        points = self.scaler.scale_point(points, inplace=False)
        X = self.descriptors(points, asarray=True)
        Y = self._cell_centers
        try:
            D = cdist(X, Y, metric, **kwargs)
        except MemoryError as e:
            memory_error = e # make it available outside the except block
            # slice X to process less rows at a time
            if metric != 'euclidean':
                raise #NotImplementedError
            K = np.zeros(X.shape[0], dtype=int)
            X2 = np.sum(X * X, axis=1, keepdims=True).astype(np.float32)
            Y2 = np.sum(Y * Y, axis=1, keepdims=True).astype(np.float32)
            X, Y = X.astype(np.float32), Y.astype(np.float32)
            n = 0
            while True:
                n += 1
                block = int(ceil(X.shape[0] * 2**(-n)))
                try:
                    np.empty((block, Y.shape[0]), dtype=X.dtype)
                except MemoryError:
                    pass # continue
                else:
                    break
            n += 2 # safer
            block = int(ceil(X.shape[0] * 2**(-n)))
            for i in range(0, X.shape[0], block):
                j = min(i+block, X2.size)
                Di = np.dot(np.float32(-2.)* X[i:j], Y.T)
                Di += X2[i:j]
                Di += Y2.T
                K[i:j] = np.argmin(Di, axis=1)
            D = None
        else:
            K = None
        #
        ncells = self._cell_centers.shape[0]
        if format == 'force array':
            min_nn = min_r = None
            format = 'array' # for later call to :func:`format_cell_index`
        if max_nn or min_nn or min_location_count or filter is not None or min_r or max_r:
            if K is None:
                # D should be defined
                K = np.argmin(D, axis=1) # cell indices
            nonempty, positive_count = np.unique(K, return_counts=True)
            if filter is not None:
                for c in nonempty:
                    cell = K == c
                    if filter_descriptors_only:
                        x = X[cell]
                    else:
                        x = points[cell]
                    if not filter(self, c, x):
                        K[cell] = -1
            # min_location_count:
            # set K[i] = -1 for all point i in cells that are too small
            if min_location_count:
                excluded_cells = positive_count < min_location_count
                if np.any(excluded_cells):
                    for c in nonempty[excluded_cells]:
                        K[K == c] = -1
                    # remove the excluded cells from nonempty and positive_count
                    ok = np.ones(nonempty.size, dtype=bool)
                    ok[excluded_cells] = False
                    nonempty, positive_count = nonempty[ok], positive_count[ok]
            # max_nn:
            # set K[i] = -1 for all point i in cells that are too large
            if max_nn:
                if callable(knn):
                    for i, c in enumerate(nonempty):
                        _, _max = knn(c)
                        if _max is None or positive_count[i] <= _max:
                            continue
                        if D is None:
                            raise memory_error
                        cell = K == c
                        I = np.argsort(D[cell, c])
                        cell, = cell.nonzero()
                        excess = cell[I[_max:]]
                        K[excess] = -1
                else:
                    large, = (max_nn < positive_count).nonzero()
                    if large.size:
                        if D is None:
                            raise memory_error
                        for c in nonempty[large]:
                            cell = K == c
                            I = np.argsort(D[cell, c])
                            cell, = cell.nonzero()
                            excess = cell[I[max_nn:]]
                            K[excess] = -1
            # max radius:
            if max_r:
                excluded_cells = []
                for i, c in enumerate(nonempty):
                    if callable(radius):
                        _, max_r = radius(c)
                        if max_r is None:
                            continue
                    cell = K == c
                    if D is None:
                        d = X[cell] - Y[[c]]
                        d = np.sqrt(np.sum(d * d, axis=1))
                    else:
                        d = D[cell, c]
                    cell, = cell.nonzero()
                    discard = max_r < d
                    K[cell[discard]] = -1
                    if np.all(discard):
                        excluded_cells.append(i)
                if excluded_cells:
                    # remove the excluded cells from nonempty and positive_count
                    ok = np.ones(nonempty.size, dtype=bool)
                    ok[excluded_cells] = False
                    nonempty, positive_count = nonempty[ok], positive_count[ok]
            # min_nn:
            # switch to vector-pair representation if any cell is too small
            if min_nn:
                if callable(knn):
                    any_small = False
                    I, n = [], []
                    for i, c in enumerate(nonempty):
                        _min, _ = knn(c)
                        if _min is None or _min <= positive_count[i]:
                            Ic, = (K == c).nonzero()
                        else:
                            any_small = True
                            if D is None:
                                raise memory_error
                            Ic = np.argsort(D[:,c])[:_min]
                        I.append(Ic)
                        n.append(len(Ic))
                    if any_small:
                        I = np.concatenate(I)
                        J = np.repeat(nonempty, n)
                        K = (I, J)
                else:
                    if min_location_count:
                        small = np.zeros(ncells, dtype=bool)
                    else:
                        small = np.ones(ncells, dtype=bool)
                    small[nonempty] = positive_count < min_nn
                    if np.any(small):
                        if D is None:
                            raise memory_error
                        # small and missing cells
                        if D.shape[0] <= min_nn:
                            assert np.all(small)
                            # the total number of points is lower than
                            # the desired minimum number of points per
                            # cell
                            n = D.shape[0]
                            I = np.repeat(np.arange(n), ncells)
                            J = np.tile(np.arange(ncells), n)
                            K = (I, J)
                        else:
                            I = np.argsort(D[:,small], axis=0)[:min_nn].flatten()
                            small, = small.nonzero()
                            J = np.tile(small, min_nn) # cell indices
                            assert I.size == J.size
                            # large-enough cells
                            #if min_location_count:
                            #    small = count < min_nn
                            point_in_small_cells = np.any(
                                small[:,np.newaxis] == K[np.newaxis,:], axis=0)
                            Ic = np.logical_not(point_in_small_cells)
                            Jc = K[Ic]
                            Ic, = Ic.nonzero()
                            Ic = Ic[0 <= Jc]
                            Jc = Jc[0 <= Jc]
                            #
                            K = (np.concatenate((I, Ic)), np.concatenate((J, Jc)))
            # min radius:
            # switch to vector-pair representation
            if min_r:
                if isinstance(K, tuple):
                    _I, _J = K
                I, n = [], []
                for c in nonempty:
                    if callable(knn):
                        min_r, _ = knn(c)
                        if min_r is None:
                            if isinstance(K, tuple):
                                Ic = I[J == c]
                            else:
                                Ic, = (K == c).nonzero()
                            I.append(Ic)
                            continue
                    pending_cells = set([c])
                    visited_cells = set()
                    included_points = np.zeros(X.shape[0], dtype=bool)
                    while pending_cells:
                        _pending_cells = set()
                        for _c in pending_cells:
                            visited_cells.add(_c)
                            if isinstance(K, tuple):
                                cell = _I[_J == _c]
                            else:
                                cell = K == _c
                            if D is None:
                                d = X[cell] - Y[[c]]
                                d = np.sqrt(np.sum(d * d, axis=1))
                            else:
                                d = D[cell, c]
                            _in = d <= min_r
                            if np.any(_in):
                                if np.all(_in):
                                    included_points[cell] = True
                                else:
                                    cell, = cell.nonzero()
                                    included_points[cell[_in]] = True
                                _pending_cells |= set(self.neighbours(_c).tolist())
                        pending_cells = _pending_cells - visited_cells
                    Ic, = included_points.nonzero()
                    I.append(Ic)
                    n.append(len(Ic))
                I = np.concatenate(I)
                J = np.repeat(nonempty, n)
                K = (I, J)

        elif K is None:
            K = np.argmin(D, axis=1) # cell indices
        point_count = points.shape[0]
        #if isinstance(points, pd.DataFrame):
        #       point_count = max(point_count, points.index.max()+1) # NO!
        # point indices are row indices and NOT row labels
        return format_cell_index(K, format=format, select=select,
            shape=(point_count, ncells))

    # cell_centers property
    @property
    def cell_centers(self):
        """Unscaled coordinates of the cell centers (numpy.ndarray)."""
        if isinstance(self.scaler.factor, pd.Series):
            return np.asarray(self.scaler.unscale_point(pd.DataFrame(self._cell_centers, \
                columns=self.scaler.factor.index)))
        else:
            return self.scaler.unscale_point(self._cell_centers)

    @cell_centers.setter
    def cell_centers(self, centers):
        self._cell_centers = self.scaler.scale_point(centers)


class Voronoi(Delaunay):
    """
    Voronoi graph.

    :class:`Voronoi` explicitly represents the cell boundaries, as a Voronoi graph, on top of the
    Delaunay graph that connects the cell centers.
    It implements the construction of this additional graph using :class:`scipy.spatial.Voronoi`.
    This default implementation is lazy. If vertices and ridges are available, they are stored in
    private attributes :attr:`_vertices`, :attr:`_vertex_adjacency` and :attr:`_cell_vertices`.
    Otherwise, when `vertices`, `vertex_adjacency` or `cell_vertices` properties are called, the
    attributes are transparently made available calling the :meth:`_postprocess` private method.
    Memory space can thus be freed again, setting `vertices`, `vertex_adjacency` and `cell_vertices`
    to ``None``.
    Note however that subclasses may override these on-time calculation mechanics.

    Attributes:

        vertices (numpy.ndarray):
            coordinates of the Voronoi vertices.

        vertex_adjacency (scipy.sparse):
            adjacency matrix for Voronoi vertices.

        cell_vertices (dict of array-like):
            mapping of cell indices to their associated vertices as indices in
            :attr:`vertices`.

        cell_volume (numpy.ndarray):
            cell volume (or surface area in 2D); only 2D is supported for now.
            Important note: if time is treated just as an extra dimension like
            in :class:`~tramway.tessellation.time.TimeLattice`, then cell volume
            may actually be spatial volume multiplied by segment duration.

    """
    __slots__ = ('_vertices', '_vertex_adjacency', '_cell_vertices', '_cell_volume')

    __lazy__ = Delaunay.__lazy__ + \
        ('vertices', 'cell_adjacency', 'cell_vertices', 'vertex_adjacency', 'cell_volume')

    def __init__(self, scaler=None):
        Delaunay.__init__(self, scaler)
        self._vertices = None
        self._vertex_adjacency = None
        self._cell_vertices = None
        self._cell_volume = None

    # vertices property
    @property
    def vertices(self):
        """Unscaled coordinates of the Voronoi vertices (numpy.ndarray)."""
        if self._cell_centers is not None and self._vertices is None:
            self._postprocess()
        return self.scaler.unscale_point(self._vertices, inplace=False)

    @vertices.setter
    def vertices(self, vertices):
        if vertices is not None:
            vertices = self.scaler.scale_point(vertices)
        self.__lazysetter__(vertices)

    # cell_adjacency property
    @property
    def cell_adjacency(self):
        if self._cell_centers is not None and self._cell_adjacency is None:
            self._postprocess()
        return self.__returnlazy__('cell_adjacency', self._cell_adjacency)

    # whenever you redefine a getter you have to redefine the corresponding setter
    @cell_adjacency.setter # copy/paste
    def cell_adjacency(self, matrix):
        self.__lazysetter__(matrix)

    # cell_vertices property
    @property
    def cell_vertices(self):
        if self._cell_centers is not None and self._cell_vertices is None:
            self._postprocess()
        return self.__returnlazy__('cell_vertices', self._cell_vertices)

    @cell_vertices.setter
    def cell_vertices(self, vertex_indices):
        self.__lazysetter__(vertex_indices)

    # vertex_adjacency property
    @property
    def vertex_adjacency(self):
        if self._cell_centers is not None and self._vertex_adjacency is None:
            self._postprocess()
        return self.__returnlazy__('vertex_adjacency', self._vertex_adjacency)

    @vertex_adjacency.setter
    def vertex_adjacency(self, matrix):
        self.__lazysetter__(matrix)

    def _postprocess(self, adjacency_label=False):
        """Compute the Voronoi.

        This private method may be called anytime by :attr:`vertices`, :attr:`vertex_adjacency`
        or :attr:`cell_vertices`.
        """
        if self._cell_centers is None:
            raise NameError('`cell_centers` not defined; tessellation has not been grown yet')
        points = np.asarray(self._cell_centers)
        if False:#points.shape[1] == 2:
            voronoi = boxed_voronoi_2d(points)
        else:
            voronoi = spatial.Voronoi(points)
        self._vertices = voronoi.vertices
        self._cell_vertices = { i: np.array([ v for v in voronoi.regions[r] if 0 <= v ]) \
                for i, r in enumerate(voronoi.point_region) if 0 <= r }
        n_centers = self._cell_centers.shape[0]
        # decompose the ridges as valid pairs of vertices and build an adjacency matrix
        ps = []
        for r in voronoi.ridge_vertices:
            pairs = np.c_[r, np.roll(r, 1)]
            pairs = pairs[np.all(0 <= pairs, axis=1)]
            ps.append(pairs)
        ij = np.concatenate(ps)
        n_vertices = self._vertices.shape[0]
        self._vertex_adjacency = sparse.coo_matrix((np.ones(ij.size, dtype=bool),
                (ij.ravel('F'), np.fliplr(ij).ravel('F'))),
            shape=(n_vertices, n_vertices))
        #
        if self._cell_adjacency is None:
            ridge_points = voronoi.ridge_points
            ridge_points = ridge_points[np.all(0 <= ridge_points, axis=1)]
            n_ridges = ridge_points.shape[0]
            if adjacency_label:
                self._adjacency_label = np.ones(n_ridges, dtype=int)
            if self._adjacency_label is None:
                edges = np.ones(n_ridges*2, dtype=bool)
            else:
                edges = np.tile(np.arange(0, n_ridges, dtype=int), 2)
            self._cell_adjacency = sparse.csr_matrix((edges, ( \
                ridge_points.flatten('F'), \
                np.fliplr(ridge_points).flatten('F'))), \
                shape=(n_centers, n_centers))
        return voronoi

    @property
    def cell_volume(self):
        if self._cell_volume is None:
            adjacency = self.vertex_adjacency.tocsr()
            cell_volume = np.full(len(self._cell_centers), np.NaN)
            for i, u in enumerate(self._cell_centers):
                js = _js = self.cell_vertices[i] # vertex indices

                if u.size != 2:
                    # use Qhull to estimate the volume
                    pts = self._vertices[js]
                    if pts.shape[1] < pts.shape[0]: # if enough points
                        hull = spatial.ConvexHull(pts)
                        cell_volume[i] = hull.volume
                    continue

                js = set(js.tolist())
                simplices = []
                while js:
                    j = js.pop()
                    # j's neighbours
                    ks = adjacency.indices[adjacency.indptr[j]:adjacency.indptr[j+1]].tolist()
                    for k in ks:
                        if k in js:
                            simplices.append((j, k))

                if len(simplices) != self.cell_vertices[i].size:
                    # missing vertices are at infinite distance;
                    # take instead the convex hull of the local vertices plus
                    # the center of the cell (ideally all the points in the cell)
                    pts = np.r_[self._vertices[_js], u[np.newaxis,:]]
                    if pts.shape[1] < pts.shape[0]: # if enough points
                        hull = spatial.ConvexHull(pts)
                        cell_volume[i] = hull.volume
                    continue

                assert bool(simplices) # `simplices` is not empty
                cell_volume[i] = 0.
                for j, k in simplices:
                    v = self._vertices[j] if isinstance(j, int) else j
                    w = self._vertices[k] if isinstance(k, int) else k
                    cell_volume[i] += .5 * abs(\
                            (v[0] - u[0]) * (w[1] - u[1]) - \
                            (w[0] - u[0]) * (v[1] - u[1]) \
                        )
            self._cell_volume = self.scaler.unscale_surface_area(cell_volume)
        return self.__returnlazy__('cell_volume', self._cell_volume)

    @cell_volume.setter
    def cell_volume(self, area):
        self.__setlazy__('cell_volume', area)



def dict_to_sparse(cell_vertex, shape=None):
    """
    Convert cell-vertex association :class:`dict` to :mod:`~scipy.sparse` matrices.
    """
    if not sparse.issparse(cell_vertex):
        if shape:
            n_cells = shape[0]
            args = [shape]
        else:
            n_cells = max(cell_vertex.keys())
            args = []
        indices = [ cell_vertex.get(c, []) for c in range(n_cells) ]
        indptr = np.r_[0, np.cumsum([ len(list(vs)) for vs in indices ])]
        indices = np.asarray(list(itertools.chain(*indices)))
        cell_vertex = sparse.csr_matrix((np.ones(indices.size, dtype=bool), indices, indptr),
            *args)
    return cell_vertex

def sparse_to_dict(cell_vertex):
    """
    Convert cell-vertex associations :mod:`~scipy.sparse` matrices to :class:`dict`.
    """
    if sparse.issparse(cell_vertex):
        matrix = cell_vertex.tocsr()
        cell_vertex = { i: matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]] \
                for i in range(matrix.shape[0]) }
    return cell_vertex


_Voronoi = namedtuple('BoxedVoronoi', (
        'points',
        'vertices',
        'ridge_points',
        'ridge_vertices',
        'regions',
        'point_region',
    ))

def boxed_voronoi_2d(points, bounding_box=None):
    voronoi = spatial.Voronoi(points)
    if bounding_box is None:
        #x_min = np.minimum(np.min(points, axis=0), np.min(voronoi.vertices, axis=0))
        #x_max = np.maximum(np.max(points, axis=0), np.max(voronoi.vertices, axis=0))
        x_min, x_max = np.min(points, axis=0), np.max(points, axis=0)
        dx = x_max - x_min
        x_min -= 1e-4 * dx
        x_max += 1e-4 * dx
        bounding_box = (
            x_min,
            np.array([x_max[0], x_min[1]]),
            x_max,
            np.array([x_min[0], x_max[1]]),
            )
    t = (bounding_box[0] + bounding_box[2]) * .5
    lstsq_kwargs = dict(rcond=None)
    region_point = np.full(len(voronoi.regions), -1, dtype=int)
    region_point[voronoi.point_region] = np.arange(voronoi.point_region.size)
    extra_vertices = []
    vertex_index = n_vertices = voronoi.vertices.shape[0]
    extra_ridges = []
    n_ridges = voronoi.ridge_points.shape[0]
    _ridge_vertices, _ridge_points, _regions = \
        voronoi.ridge_vertices, voronoi.ridge_points, [[]]
    for c, u in enumerate(points):
        r = voronoi.point_region[c]
        region = voronoi.regions[r]
        #assert region_point[r] == c
        _region = []
        for k, h in enumerate(region):
            if 0 <= h:
                _region.append(h)
                continue

            # find the two "adjacent" vertices
            i, j = region[k-1], region[(k+1)%len(region)]
            assert 0<=i
            assert 0<=j
            # find the corresponding neighbour cells
            m, n = set(), set() # mutable
            for d, hs in enumerate(voronoi.regions):
                if not hs or d == r:
                    continue
                for e, cs in ((i,m), (j,n)):
                    try:
                        l = hs.index(e)
                    except ValueError:
                        continue
                    if (hs[l-1]==-1) or (hs[(l+1)%len(hs)]==-1):
                        cs.add(region_point[d])
            p, q = m.pop(), n.pop()
            assert not m
            assert not n
            # pick a distant point on the perpendicular bissector
            # of the neighbour centers, at the intersection with
            # the bounding box
            prev_edge = None
            for h, d in ((i,p), (j,q)):
                gs = []
                try:
                    gs = _ridge_vertices[n_ridges+extra_ridges.index([d,c])]
                except ValueError:
                    try:
                        gs = _ridge_vertices[n_ridges+extra_ridges.index([c,d])]
                    except ValueError:
                        pass
                if gs:
                    g, = [ g for g in gs if g != h ]
                    _region.append(g)
                    continue
                v = voronoi.vertices[h]
                w = points[d]
                if np.any(v<bounding_box[0]) or np.any(bounding_box[2]<v):
                    # vertex v stands outside the bounding box
                    # TODO: check for corners as potential intermediate vertices
                    continue
                n = np.array([u[1]-w[1], w[0]-u[0]])
                #y = (u + w) * .5
                y = v
                # determine direction: n or -n?
                if 0 < np.dot(n, t-y):
                    n = -n
                # intersection of [y,n) and [a,ab]
                v_next = None
                for l, a in enumerate(bounding_box):
                    b = bounding_box[(l+1)%len(bounding_box)]
                    M, p = np.c_[n, a-b], a-y
                    q = np.linalg.lstsq(M, p, **lstsq_kwargs)
                    q = q[0]
                    if 0<=q[0] and 0<=q[1] and q[1]<=1:
                        # intersection found
                        v_next = y + q[0] * n
                        #if 0<q[0]:
                        break
                assert v_next is not None
                if prev_edge is None or prev_edge == l:
                    h_prev = h
                else:
                    # add the corner which the previous
                    # and current edges intersect at
                    e_max = len(bounding_box)-1
                    if (0<l and prev_edge==l-1) or \
                        (l==0 and prev_edge==e_max):
                        v_prev = a
                    elif (l<e_max and prev_edge==l+1) or \
                        (l==e_max and prev_edge==0):
                        v_prev = b
                    else:
                        # two corners?
                        # better connect the intersection points and let the corners out
                        h_prev = h
                        v_prev = None
                    if v_prev is not None:
                        h_prev = vertex_index
                        vertex_index += 1
                        extra_vertices.append(v_prev)
                        extra_ridges.append([-1,c])
                        _ridge_vertices.append([h,h_prev])
                        _region.append(h_prev)
                prev_edge = l
                #
                h_next = vertex_index
                vertex_index += 1
                # insert the new vertex
                extra_vertices.append(v_next)
                k, = ((_ridge_points[:,0]==c) & (_ridge_points[:,1]==d)).nonzero()
                if k.size==0:
                    k, = ((_ridge_points[:,0]==d) & (_ridge_points[:,1]==c)).nonzero()
                k = k[0].tolist()
                _ridge_vertices = _ridge_vertices[:k] \
                    + [[h_prev,h_next]] \
                    + _ridge_vertices[k+1:]
                _region.append(h_next)
                assert len(extra_vertices) == vertex_index - n_vertices

        assert all( r in _region for r in region if 0 < r )
        _regions.append(_region)
    _points = voronoi.points
    _vertices = np.vstack([voronoi.vertices]+extra_vertices)
    _ridge_points = np.vstack([_ridge_points]+extra_ridges)
    _point_region = np.arange(1, len(_regions))#voronoi.point_region
    return _Voronoi(_points, _vertices, _ridge_points, _ridge_vertices, _regions, _point_region)


def cell_index_by_radius(tessellation, points, radius, format=None, select=None, metric='euclidean',
        **kwargs):
    """
    See :meth:`Delaunay.cell_index`.

    Specialized routine to assign locations to cells which center is no further than `radius`.
    """
    #if metric != 'euclidean':
    #    raise NotImplementedError('%s metric not supported', metric)
    r2 = radius * radius
    points = tessellation.scaler.scale_point(points, inplace=False)
    X = tessellation.descriptors(points, asarray=True)
    Y = tessellation._cell_centers
    ncells = Y.shape[0]
    shape = (X.shape[0], ncells)
    try:
        D = cdist(X, Y, metric, **kwargs)
    except MemoryError:
        # slice X to process less rows at a time
        if metric != 'euclidean':
            raise #NotImplementedError
        X2 = np.sum(X * X, axis=1, keepdims=True).astype(np.float32)
        Y2 = np.sum(Y * Y, axis=1, keepdims=True).astype(np.float32)
        X, Y = X.astype(np.float32), Y.astype(np.float32)
        n = 0
        while True:
            n += 1
            block = int(ceil(X.shape[0] * 2**(-n)))
            try:
                np.empty((block, Y.shape[0]), dtype=X.dtype)
            except MemoryError:
                pass # continue
            else:
                break
        n += 2 # safer
        block = int(ceil(X.shape[0] * 2**(-n)))
        P, C = [], []
        for i in range(0, X.shape[0], block):
            j = min(i+block, X2.size)
            Di = np.dot(np.float32(-2.) * X[i:j], Y.T)
            Di += X2[i:j]
            Di += Y2.T
            Pi, Ci = (Di <= r2).nonzero()
            P.append(i+Pi)
            C.append(Ci)
        associations = (np.concatenate(P), np.concatenate(C))
    else:
        associations = (D <= r2).nonzero()
    return format_cell_index(associations, format=format, select=select, shape=shape)


__all__ = ['CellStats', 'point_adjacency_matrix', 'Tessellation', 'Delaunay', 'Voronoi', \
    'format_cell_index', 'nearest_cell', 'dict_to_sparse', 'sparse_to_dict', \
    '_Voronoi', 'boxed_voronoi_2d', 'cell_index_by_radius']


