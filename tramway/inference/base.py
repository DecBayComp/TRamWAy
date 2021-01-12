# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import *
from tramway.core.exceptions import *
from tramway.tessellation import format_cell_index, nearest_cell
import tramway.tessellation as tessellation
from .gradient import grad1, delta0
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.spatial.qhull
from copy import copy
from collections import OrderedDict
from multiprocessing import Pool, Lock
import os # for os.name
import six
from functools import partial
from warnings import warn
import traceback



class Local(Lazy):
    """
    Spatially local subset of elements (e.g. translocations). Abstract class.

    Attributes:

        index (int):
            This cell's index as referenced in :class:`FiniteElements`.

        data (collection of terminal elements or :class:`Local`):
            Elements, either terminal or not.

        center (array-like):
            Cell center coordinates.

        span (array-like):
            Difference vectors from this cell's center to adjacent centers.

        boundary/hull (scipy.spatial.qhull.ConvexHull):
            Convex hull of the contained locations.

    """
    __slots__ = ('index', 'data', 'center', 'span', '_boundary', '_volume')
    __lazy__ = Lazy.__lazy__ + ('volume',)

    def __init__(self, index, data, center=None, span=None, boundary=None):
        Lazy.__init__(self)
        self.index = index
        self.data = data
        self.center = center
        self.span = span
        self._boundary = boundary
        self._volume = None

    @property
    def boundary(self):
        """
        *any*, property

        Cell boundary/hull.
        """
        return self._boundary

    @boundary.setter
    def boundary(self, b):
        self.volume = None
        self._boundary = b

    @property
    def hull(self):
        """
        `scipy.spatial.qhull.ConvexHull` (or *any* if not convex), property

        Alias for `boundary`.
        """
        return self.boundary

    @hull.setter
    def hull(self, h):
        self.boundary = h

    @property
    def volume(self):
        """
        `float`, property

        Surface area (2D) or volume (3D+) of the convex hull.

        Note that the hull should be convex so that the volume can be
        transparently extracted.
        In other cases, `volume` can be manually set just like a normal
        attribute.
        """
        if self._volume is None and self.hull is not None:
            if self.dim == 2:
                area = 0
                u = self.center
                for i in self.hull.simplices:
                    v, w = self.hull.points[i]
                    area += .5 * abs(\
                            (v[0] - u[0]) * (w[1] - u[1]) - \
                            (w[0] - u[0]) * (v[1] - u[1]) \
                        )
                self._volume = area
            else:
                self._volume = self.hull.volume
        return self.__returnlazy__('volume', self._volume)

    @volume.setter
    def volume(self, v):
        self.__setlazy__('volume', v)

    @property
    def dim(self):
        """
        `int`, property

        Dimension of the terminal elements.
        """
        raise NotImplementedError('abstract method')

    @dim.setter
    def dim(self, d):
        self.__assertlazy__('dim', d, related_attribute='data')

    @property
    def tcount(self):
        """
        `int`, property

        Total number of terminal elements (e.g. translocations).
        """
        raise NotImplementedError('abstract method')

    @tcount.setter
    def tcount(self, c):
        self.__assertlazy__('tcount', c, related_attribute='data')

    def get_localization_error(self, _kwargs=None, _default_value=None, _localization_error_is_sigma=False, **kwargs):
        """
        Return the localization error as :math:`sigma^2`.

        Arguments:

            _kwargs (dict):
                mutable keyword arguments;
                '*localization_error*', '*sigma*' and '*sigma2*' are popped out.

            _default_value (float):
                default localization error (subject to `_localization_error_is_sigma`).

            _localization_error_is_sigma (bool):
                argument '*localization_error*' in `_kwargs` or `kwargs` is considered as
                :math:`sigma`.

        Returns:

            float: :math:`sigma^2`.

        """
        if _localization_error_is_sigma:
            _default_sigma2 = _default_value * _default_value
        else:
            _default_sigma2 = _default_value

        if _kwargs:
            args = {}
            for arg in ('localization_error', 'sigma', 'sigma2'):
                try:
                    val = kwargs[arg]
                except KeyError:
                    try:
                        val = _kwargs.pop(arg)
                    except KeyError:
                        val = None
                args[arg] = val
        elif kwargs:
            args = kwargs
        else:
            return _default_sigma2

        localization_error = args.get('localization_error', None)
        sigma = args.get('sigma', None)
        sigma2 = args.get('sigma2', None)

        if localization_error is None:
            _sigma2 = None
        elif _localization_error_is_sigma:
            _sigma2 = localization_error * localization_error
        else:
            _sigma2 = localization_error

        if sigma2 is None:
            if sigma is None:
                if localization_error is None:
                    sigma2 = _default_sigma2
                else:
                    if _localization_error_is_sigma:
                        warn('`localization_error` may become sigma square in the coming 0.4 release, instead of sigma; please use `sigma` or `sigma2` to disambiguate', PendingDeprecationWarning)
                    sigma2 = _sigma2
            else:
                sigma2 = sigma * sigma
                if not (localization_error is None or np.isclose(_sigma2, sigma2)):
                    raise MultipleArgumentError('localization_error', 'sigma')
        else:
            if not (sigma is None or np.isclose(sigma*sigma, sigma2)):
                if localization_error is None:
                    raise MultipleArgumentError('sigma', 'sigma2')
                else:
                    raise MultipleArgumentError('localization_error', 'sigma', 'sigma2')
            if not (_sigma2 is None or np.isclose(_sigma2, sigma2)):
                raise MultipleArgumentError('localization_error', 'sigma2')

        return sigma2



class Distributed(Local):
    """
    Attributes:

        central (array of bools):
            margin cells are not central.

    """
    __slots__ = ('_reverse', '_adjacency', 'central', '_degree', '_ccount', '_tcount')
    __lazy__  = Local.__lazy__ + ('reverse', 'degree', 'ccount', 'tcount')

    def __init__(self, cells, adjacency, index=None, center=None, span=None, central=None, \
        boundary=None):
        Local.__init__(self, index, OrderedDict(), center, span, boundary)
        self.cells = cells # let's `cells` setter perform the necessary checks
        self.adjacency = adjacency
        self.central = central

    @property
    def cells(self):
        """
        `list` or `OrderedDict`, rw property for :attr:`data`

        Collection of :class:`Local`. Indices may not match with the global
        :attr:`~Local.index` attribute of the elements, but match with attributes
        :attr:`central`, :attr:`adjacency` and :attr:`degree`.
        """
        return self.data

    @cells.setter
    def cells(self, cells):
        celltype = type(self.cells)
        assert(celltype is dict or celltype is OrderedDict)
        if not isinstance(cells, celltype):
            if isinstance(cells, dict):
                cells = celltype(sorted(cells.items(), key=lambda t: t[0]))
            elif isinstance(cells, list):
                cells = celltype(sorted(enumerate(cells), key=lambda t: t[0]))
            else:
                raise TypeError('`cells` argument is not a dictionnary (`dict` or `OrderedDict`)')
        if not all([ isinstance(cell, Local) for cell in cells.values() ]):
            raise TypeError('`cells` argument is not a dictionnary of `Local`')
        #try:
        #       if self.ccount == len(cells): #.keys().reversed().next(): # max
        #           self.adjacency = None
        #           self.central = None
        #except:
        #       pass
        self.reverse = None
        self.ccount = None
        self.data = cells

    @property
    def indices(self):
        return np.array([ cell.index for cell in self.cells.values() ])

    @property
    def reverse(self):
        """
        `dict of ints`, ro lazy property

        Get "local" indices from global ones.
        """
        if self._reverse is None:
            self._reverse = {cell.index: i for i, cell in self.cells.items()}
        return self._reverse

    @reverse.setter
    def reverse(self, r): # ro
        self.__assertlazy__('reverse', r, related_attribute='cells')

    @property
    def space_cols(self):
        """
        `list of ints` or `strings`, ro property

        Column indices for coordinates of the terminal points.
        """
        return self.any_cell().space_cols

    @property
    def dim(self):
        """
        `int`, ro property

        Dimension of the terminal points.
        """
        if self.center is None:
            return self.any_cell().dim
        else:
            return self.center.size

    @property
    def tcount(self):
        """
        `int`, ro property

        Total number of terminal points. Duplicates are ignored.
        """
        if self._tcount is None:
            if self.central is None:
                self._tcount = sum([ cell.tcount \
                    for i, cell in self.cells.items() if self.central[i] ])
            else:
                self._tcount = sum([ cell.tcount for cell in self.cells.values() ])
        return self._tcount

    @tcount.setter
    def tcount(self, c):
        # write access allowed for performance issues, but `c` should equal self.tcount
        self.__setlazy__('tcount', c)

    @property
    def ccount(self):
        """
        `int`, ro property

        Total number of terminal cells. Duplicates are ignored.
        """
        #return self.adjacency.shape[0] # or len(self.cells)
        if self._ccount is None:
            self._ccount = sum([ cell.ccount if isinstance(cell, FiniteElements) else 1 \
                for cell in self.cells.values() ])
        return self._ccount # not mutable (no need for __returnlazy__)

    @ccount.setter
    def ccount(self, c): # rw for performance issues, but `c` should equal self.ccount
        self.__setlazy__('ccount', c)

    @property
    def adjacency(self):
        """
        `scipy.sparse.csr_matrix`, rw property

        Cell adjacency matrix. Row and column indices are to be mapped with `indices`.
        """
        return self._adjacency

    @adjacency.setter
    def adjacency(self, a):
        if a is not None:
            a = a.tocsr()
        self._adjacency = a
        self._degree = None # `degree` is ro, hence set `_degree` instead

    @property
    def degree(self):
        """
        `array of ints`, ro lazy property

        Number of adjacent cells.
        """
        if self._degree is None:
            self._degree = np.diff(self._adjacency.indptr)
        return self._degree

    @degree.setter
    def degree(self, d): # ro
        self.__lazyassert__(d, 'adjacency')

    def grad(self, i, X, index_map=None, **kwargs):
        """
        Local spatial gradient.

        Sub-classes are free to return multi-component gradients as matrices provided that
        they exhibit as many columns as there are dimensions in the (trans-)location data.
        :meth:`grad_sum` is then responsible for summing all the elements of such matrices.

        Arguments:

            i (int):
                cell index at which the gradient is evaluated.

            X (numpy.ndarray):
                vector of a scalar measurement at every cell.

            index_map (numpy.ndarray):
                index map that converts cell indices to indices in X.

        Returns:

            numpy.ndarray:
                gradient vector with as many elements as spatial dimensions.

        See also :func:`~tramway.inference.gradient.grad1` and documentation section :ref:`gradient`.
        """
        return grad1(self, i, X, index_map, **kwargs)

    def grad_sum(self, i, grad, index_map=None):
        """
        Mixing operator for the gradient at a given cell.

        Arguments:

            i (int):
                cell index.

            grad (numpy.ndarray):
                local gradient.

            index_map (numpy.ndarray):
                index mapping, useful to convert cell indices to positional indices in
                an optimization array for example.

        Returns:

            float:
                weighted sum of the elements of `grad`.

        """
        cell = self.cells[i]
        if cell.volume:
            return cell.volume * np.sum(grad)
        else:
            return np.sum(grad)

    def local_variation(self, i, X, index_map=None, **kwargs):
        """
        Local spatial variation, gradient-like, aimed at penalizing spatial variations.

        See also :func:`~tramway.inference.gradient.delta0`.

        As of version *0.3.8*: new; called for spatial regularization in `stochastic_dv`.

        As of version *0.4*: default implementation becomes :func:`~tramway.inference.gradient.delta0`.

        May become the new default for spatial regularization.

        Arguments:

            i (int):
                cell index at which the gradient is evaluated.

            X (numpy.ndarray):
                vector of a scalar measurement at every cell.

            index_map (numpy.ndarray):
                index map that converts cell indices to indices in X.

        Returns:

            numpy.ndarray:
                delta vector with as many elements as there are spatial dimensions.

        """
        return delta0(self, i, X, index_map, **kwargs)

    def flatten(self):
        def concat(arrays):
            if isinstance(arrays[0], tuple):
                raise NotImplementedError
            elif isinstance(arrays[0], pd.DataFrame):
                return pd.concat(arrays, axis=0)
            else:
                return np.stack(arrays, axis=0)

        new = copy(self)
        new.cells = {i: Cell(i, concat([cell.data for cell in dist.cells.values()]), \
                dist.center, dist.span, hull=dist.hull) \
            if isinstance(dist, FiniteElements) else dist \
            for i, dist in self.cells.items() }

        return new

    def group(self, ngroups=None, max_cell_count=None, cell_centers=None, \
        adjacency_margin=2, connected=False):
        """
        Make groups of cells.

        This builds up an extra hierarchical level.
        For example if `self` is a `FiniteElements` of `Cell`, then this returns a `FiniteElements`
        of `FiniteElements` (the groups) of `Cell`.

        Several grouping strategies are proposed.

        Arguments:

            ngroups (int):
                number of groups.

            max_cell_count (int):
                maximum number of cells per group.

            cell_centers (array-like):
                spatial centers of the groups.
                Cells are sorted by nearest neighbours.
                If not provided, :meth:`group` will use
                a k-means approach to positionning the centers.

            adjacency_margin (int):
                groups are dilated to the adjacent cells `adjacency_margin` times.
                Defaults to 2.

            connected (bool):
                separates the connected components from one another.
                Conflicts with the other arguments.

        Returns:

            FiniteElements:
                a new object of the same type as `self` that contains other such
                objects as cells.

        """
        ncells = self.adjacency.shape[0]
        new = copy(self)

        strategy_0 = ngroups or (max_cell_count and max_cell_count < ncells) or cell_centers is not None
        if not strategy_0 and not connected:
            #raise KeyError('`group` expects more input arguments')
            return new

        elif strategy_0:

            if connected:
                raise ValueError('`connected` is not supported in combination with other arguments')

            if cell_centers is None and max_cell_count == 1 and not adjacency_margin:
                distr = type(new)
                for i in new:
                    cell = new[i]
                    new[i] = distr({i: cell}, sparse.eye(1, dtype=bool, format='csr'),
                            center=cell.center)
                return new

            any_cell = self.any_cell()

            points = np.zeros((ncells, self.dim), dtype=any_cell.center.dtype)
            ok = np.zeros(points.shape[0], dtype=bool)
            for i in self.cells:
                if self.cells[i]: # non-empty
                    points[i] = self.cells[i].center
                    ok[i] = True

            if cell_centers is None:
                if max_cell_count == 1:
                    assert False
                    grid = tessellation.Voronoi()
                    grid.cell_centers = points
                else:
                    avg_probability = 1.0
                    if ngroups:
                        avg_probability = min(1.0 / float(ngroups), avg_probability)
                    if max_cell_count:
                        avg_probability = min(float(max_cell_count) / \
                            float(points.shape[0]), avg_probability)
                    import tramway.tessellation.kmeans as kmeans
                    grid = kmeans.KMeansMesh(avg_probability=avg_probability)
                    try:
                        grid.tessellate(points[ok])
                    except scipy.spatial.qhull.QhullError:
                        print('Qhull errors are handled; full map optimization')
                        return new
                    except ValueError:
                        print(points)
                        raise

            else:
                grid = tessellation.Voronoi()
                grid.cell_centers = cell_centers

            I = np.full(ok.size, -1, dtype=int)
            I[ok] = grid.cell_index(points[ok], min_location_count=1)
            #if not np.all(ok):
            #       print(ok.nonzero()[0])
            new.adjacency = grid.simplified_adjacency(format='csr') # macro-cell adjacency matrix
            J = np.unique(I)
            J = J[0 <= J]
            assert 0 < J.size
            new.data = type(self.cells)()

            for j in J: # for each macro-cell
                K = I == j # find corresponding cells
                assert np.any(K)

                if 0 < adjacency_margin:
                    L = np.copy(K)
                    for k in range(adjacency_margin):
                        # add adjacent cells for future gradient calculations
                        K[self.adjacency[K,:].indices] = True
                    L = L[K]

                A = self.adjacency[K,:].tocsc()[:,K].tocsr() # point adjacency matrix
                C = grid.cell_centers[j]
                D = OrderedDict([ (i, self.cells[k]) \
                    for i, k in enumerate(K.nonzero()[0]) if k in self.cells ])

                for i in D:
                    adj = A[i].indices
                    if 0 < D[i].tcount and adj.size:
                        span = np.stack([ D[k].center for k in adj ], axis=0)
                    else:
                        span = np.empty((0, D[i].center.size), \
                            dtype=D[i].center.dtype)
                    if span.shape[0] < D[i].span.shape[0]:
                        D[i] = copy(D[i])
                        D[i].span = span - D[i].center

                R = grid.cell_centers[new.adjacency[j].indices] - C

                new.cells[j] = type(self)(D, A, index=j, center=C, span=R)

                if 0 < adjacency_margin:
                    new.cells[j].central = L
                assert bool(new.cells[j])

        else:
            new.cells = type(self.cells)()
            j = 0
            available_cells = self.cells
            while available_cells:

                any_cell = next(iter(available_cells))
                neighbours = set(self.neighbours(any_cell).tolist())
                visited_cells = set([any_cell])
                while neighbours:
                    visited_cells |= neighbours
                    new_neighbours = set()
                    for cell in neighbours:
                        new_neighbours |= set(self.neighbours(cell).tolist())
                    neighbours = new_neighbours - visited_cells
                available_cells = { k: cell for k, cell in available_cells.items()
                    if k not in visited_cells } # update for next iteration

                K = list(visited_cells)
                A = self.adjacency[K,:].tocsc()[:,K].tocsr() # point adjacency matrix
                D = OrderedDict([ (i, self.cells[k]) \
                    for i, k in enumerate(K) if k in self.cells ])

                new.cells[j] = type(self)(D, A, index=j)

                j += 1

            new.adjacency = sparse.csr_matrix((np.array([], dtype=bool), (np.array([], dtype=int), np.array([], dtype=int))), shape=(j, j))


        new.ccount = self.ccount
        # _tcount is not supposed to change

        return new

    def run(self, function, *args, **kwargs):
        """
        Apply a function that takes a group (:class:`FiniteElements`) of terminal cells
        as input argument, plus args and kwargs, and must return a `pandas.DataFrame` array
        with the indices referring to cells.

        `function` is called for each group of terminal cells, adjacency margins
        are removed if any, and the resulting `DataFrame` are merged into a single
        `DataFrame`.

        `function` may instead be applied to `self` in the case `self.cells` has been
        overloaded to exhibit the output features as attributes.

        Arguments:

            function (callable):
                the function to be called on each terminal :class:`FiniteElements`.
                Its first argument is the :class:`FiniteElements` object.
                It should return maps as a :class:`~pandas.DataFrame` and optionally
                posteriors as :class:`~pandas.Series` or :class:`~pandas.DataFrame`.

            args (list):
                positional arguments for `function` after the first one.

            kwargs (dict):
                keyword arguments for `function`;
                the following arguments are popped out of `kwargs`:

            returns (list):
                attributes to be collected from all the individual cells as return values;
                if defined, the values returned by `function` are ignored.

            worker_count (int):
                number of simultaneously working processing units,
                if the `self.cells` .

            profile (bool or str or tuple):
                profile each child job if any;
                if `str`, dump the output stats into *.prof* files;
                if `tuple`, print a report with :func:`~pstats.Stats.print_stats` and
                tuple elements as input arguments.

            function_worker_count (int):
                if `worker_count` is a reserved keyword argument for `function`,
                `function_worker_count` can be specified and is passed to `function`
                with name/keyword `worker_count`;
                in this case, `worker_count` must be a multiple of `function_worker_count`.

        Returns:

            pandas.DataFrame:
                single merged array of maps.
                If `function` returns two output arguments, :meth:`run` also
                returns a second merged array of posterior probabilities(?).

        """
        # clear the caches
        self.clear_caches()

        returns = kwargs.pop('returns', None)

        parallel = all(isinstance(cell, FiniteElements) for cell in self.cells.values())
        if parallel:
            worker_count = kwargs.pop('worker_count', None)
            profile = kwargs.pop('profile', False)

        for arg in ('returns', 'worker_count', 'profile'):
            try:
                val = kwargs.pop('function_'+arg)
            except KeyError:
                pass
            else:
                kwargs[arg] = val
                if arg == 'worker_count' and \
                        isinstance(worker_count, int) and \
                        isinstance(val, int) and 0<val:
                    worker_count /= val

        if parallel:
            # parallel for-loop over the subsets of cells
            # if Windows, make the computation sequential
            if os.name == 'nt':
                if worker_count is None:
                    worker_count = 0
                else:
                    warn('multiprocessing may break on Windows', RuntimeWarning)
            # if `worker_count` is `None`, `Pool` will use `multiprocessing.cpu_count()`
            # if `worker_count == 0`, make it single-processing
            if worker_count == 0:
                pool = None
            else:
                pool = Pool(worker_count)
            fargs = (function, args, kwargs)
            if profile:
                fargs = (profile, fargs)
                cells = [ (i, self.cells[i]) for i in self.cells ]#if bool(self.cells[i]) ]
            else:
                cells = [ self.cells[i] for i in self.cells ]#if bool(self.cells[i]) ]
            if six.PY3 or pool is None:
                if profile:
                    _run = __profile_run__
                else:
                    _run = __run__
                if pool is None:
                    ys = [ _run(fargs, c) for c in cells ]
                else:
                    ys = pool.map(partial(_run, fargs), cells)
            elif six.PY2:
                import itertools
                if profile:
                    _run = __profile_run_star__
                else:
                    _run = __run_star__
                ys = pool.map(_run,
                    itertools.izip(itertools.repeat(fargs), cells))
            if returns is None:
                ys = [ y for y in ys if y is not None ]
                if ys:
                    if ys[1:]:
                        if isinstance(ys[0], tuple):
                            ys = zip(*ys)
                            result = tuple([
                                pd.concat(_ys, axis=0).sort_index()
                                for _ys in ys
                                if _ys and isinstance(_ys[0], pd.DataFrame) ])
                            if not result[1:]:
                                result, = result
                        else:
                            result = pd.concat(ys, axis=0).sort_index()
                    else:
                        result = ys[0]
                else:
                    result = None

        else:
            # direct function application
            result = function(self, *args, **kwargs)

        if returns:
            index = { v: [] for v in returns }
            result = { v: [] for v in returns }
            for i in self.cells:
                cell = self.cells[i]
                if cell:
                    for v in returns:
                        try:
                            x = getattr(cell, v)
                        except AttributeError:
                            x = None
                        if x is not None:
                            index[v].append(i)
                            result[v].append(x)
            for v in result:
                x = np.vstack(result[v])
                result[v] = pd.DataFrame(x, index=index[v], \
                        columns=[v] if x.shape[1] == 1 \
                        else [ '{} {:d}'.format(v, i) for i in range(x.shape[1]) ])
            _result = result
            result = _result[returns[0]]
            if returns[1:]:
                result = result.join([ _result[v] for v in returns[1:] ])

        return result

    # `dict` interface
    def __len__(self):
        return self.adjacency.shape[0]

    def __nonzero__(self):
        return bool(self.cells)

    def __iter__(self):
        return iter(self.cells)

    def __getitem__(self, i):
        return self.cells[i]

    def __setitem__(self, i, cell):
        self.cells[i] = cell

    def __delitem__(self, i):
        self.cells.__delitem__(i)

    def keys(self):
        try:
            return self.cells.keys()
        except AttributeError:
            return range(len(self.cells))

    def values(self):
        try:
            return self.cells.values()
        except AttributeError:
            return self.cells

    def items(self):
        try:
            return self.cells.items()
        except AttributeError:
            return enumerate(self.cells)

    def any_cell(self):
        return self.cells[next(iter(self.cells))]

    def neighbours(self, i):
        """
        Indices of neighbour cells.

        Argument:

            i (int): cell index.

        Returns:

            numpy.ndarray: indices of the neighbour cells of cell *i*.

        """
        return self.adjacency.indices[self.adjacency.indptr[i]:self.adjacency.indptr[i+1]]

    def clear_caches(self):
        try:
            first = True
            for c in self.values():
                c.clear_cache()
                first = False
        except AttributeError as e:
            if first:
                try:
                    first = True
                    for c in self.values():
                        c.clear_caches()
                        first = False
                    return
                except:
                    if first:
                        raise e
            if not first:
                raise


def __run__(func, cell):
    function, args, kwargs = func
    try:
        x = cell.run(function, *args, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print(traceback.format_exc())
        raise
    if x is None:
        return None
    else:
        is_tuple = isinstance(x, tuple)
        if is_tuple:
            _x = x[1:]
            x = x[0]
        i = cell.indices
        if cell.central is not None:
            try:
                x = x.iloc[cell.central[x.index]]
            except IndexError as e:
                if cell.central.size < x.index.max():
                    raise IndexError('dataframe indices do no match with group-relative cell indices (maybe are they global ones)')
                else:
                    print(x.shape)
                    print((cell.central.shape, cell.central.max()))
                    print(x.index.max())
                    raise e
            i = i[x.index]
            if x.shape[0] != i.shape[0]:
                raise IndexError('not as many indices as values')
        x.index = i
        if is_tuple:
            return (x,)+_x
        else:
            return x

def __run_star__(args):
    return __run__(*args)


def __profile_run__(func, args):
    import cProfile, pstats
    proptions, func = func
    try:
        process, cells = args
    except TypeError:
        process = None
    profile = cProfile.Profile()
    profile.enable()
    result = __run__(func, cells)
    profile.disable()
    if process is not None and isinstance(proptions, str):
        profile.dump_stats('{}{}.prof'.format(filename, process))
    else:
        if proptions in (True, False):
            proptions = (.1, )
        elif not isinstance(proptions, (tuple, list)):
            proptions = (proptions, )
        stats = pstats.Stats(profile).sort_stats('cumulative')
        stats.print_stats(*proptions)
    return result

def __profile_run_star__(args):
    return __profile_run__(*args)


FiniteElements = Distributed



class Cell(Local):
    """
    Spatially constrained subset of (trans-)locations with associated intermediate calculations.

    Attributes:

        cache (any):
            Depending on the inference approach and objective, caching an intermediate
            result may avoid repeating many times a same computation. Usage of this cache
            is totally free and comes without support for concurrency.

    """
    __slots__ = ('_time_col', '_space_cols', 'cache', 'fuzzy')
    __lazy__  = Local.__lazy__ + ('time_col', 'space_cols')

    def __init__(self, index, data, center=None, span=None, boundary=None):
        """
        Arguments:

            index (int):
                this cell's index as granted in :class:`FiniteElements` 's `cells` dict.

            center (array-like):
                cell center coordinates.

            span (array-like):
                difference vectors from this cell's center to adjacent centers.

        """
        if not (isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame)):
            raise TypeError('unsupported (trans-)location type `{}`'.format(type(data)))
        Local.__init__(self, index, data, center, span, boundary)
        #self._tcount = data.shape[0]
        self._time_col = None
        self._space_cols = None
        self.cache = None
        #self.data = (self.space_data, self.time_data)
        self.fuzzy = None

    @property
    def time_col(self):
        """
        `int` or `string`, lazy

        Column index for time.
        """
        if self._time_col is None:
            if isstructured(self.data):
                self._time_col = 't'
            else:
                self._time_col = 0
        return self._time_col

    @time_col.setter
    def time_col(self, col):
        if isinstance(col, (tuple, list)):
            if not col or col[1:]:
                raise ValueError('a single time column is supported')
            col = col[0]
        # space_cols is left unchanged
        self.__lazysetter__(col)

    @property
    def space_cols(self):
        """
        `list of ints` or `strings`, lazy

        Column indices for coordinates.
        """
        if self._space_cols is None:
            if isstructured(self.data):
                self._space_cols = columns(self.data)
                if isinstance(self._space_cols, pd.Index):
                    self._space_cols = self._space_cols.drop(self.time_col)
                else:
                    self._space_cols.remove(self.time_col)
            else:
                if self.time_col == 0:
                    self._space_cols = np.arange(1, self.data.shape[1])
                else:
                    self._space_cols = np.ones(self.data.shape[1], \
                        dtype=bool)
                    self._space_cols[self.time_col] = False
                    self._space_cols, = self._space_cols.nonzero()
        return self._space_cols

    @space_cols.setter
    def space_cols(self, cols):
        # time_col is left unchanged
        self.__lazysetter__(cols)

    @property
    def dim(self):
        return len(self.space_cols)

    @dim.setter
    def dim(self, d): # ro
        self.__lazyassert__(d, 'data')

    @property
    def tcount(self):
        """
        `int`

        Number of (trans-)locations.
        """
        return self.space_data.shape[0]

    @tcount.setter
    def tcount(self, c): # ro
        self.__lazyassert__(c, 'data')

    @property
    def time_data(self):
        if not isinstance(self.data, tuple):
            self.data = (self._extract_space(), self._extract_time())
        return self.data[1]

    @time_data.setter
    def time_data(self, t):
        self.data = (self.space_data, t)

    def _extract_time(self):
        if isstructured(self.data):
            return np.asarray(self.data[self.time_col])
        else:
            return np.asarray(self.data[:,self.time_col])

    @property
    def space_data(self):
        if not isinstance(self.data, tuple):
            self.data = (self._extract_space(), self._extract_time())
        return self.data[0]

    @space_data.setter
    def space_data(self, xy):
        self.data = (xy, self.time_data)

    def _extract_space(self):
        if isstructured(self.data):
            return np.asarray(self.data[self.space_cols])
        else:
            return np.asarray(self.data[:,self.space_cols])

    def __len__(self):
        return self.tcount

    def __nonzero__(self):
        if isinstance(self.data, tuple):
            return 0 < int(self.data[0].size)
        else:
            return 0 < int(self.data.size)

    def clear_cache(self):
        self.cache = None


FiniteElement = Cell



class Locations(Cell):
    """

    """
    __slots__ = ()

    @property
    def locations(self):
        """
        `array-like`, property

        Locations as a matrix of coordinates and times with as many
        columns as dimensions; this is an alias for :attr:`~Local.data`.
        """
        return self.data

    @locations.setter
    def locations(self, tr):
        self.data = tr

    @property
    def t(self):
        """
        `array-like`, ro property

        Location timestamps.
        """
        return self.time_data

    @t.setter
    def t(self, t):
        self.time_data = t

    @property
    def r(self):
        """
        `array-like`, ro property

        Location coordinates; `xy` is an alias
        """
        return self.space_data

    @r.setter
    def r(self, r):
        self.space_data = r

    @property
    def xy(self):
        return self.space_data

    @xy.setter
    def xy(self, xy):
        self.space_data = xy


class Translocations(Cell):
    """
    Attributes:

        origins (array-like, ro property):
            Initial locations (both spatial coordinates and times).

        destinations (array-like, ro property):
            Final locations (both spatial coordinates and times).

    """
    __slots__ = ('origins', 'destinations')

    def __init__(self, index, translocations, center=None, span=None, boundary=None):
        Cell.__init__(self, index, translocations, center, span, boundary)
        self.origins = None
        self.destinations = None

    @property
    def translocations(self):
        """
        `array-like`, property

        Translocations as a matrix of variations of coordinate and time with as many
        columns as dimensions; this is an alias for :attr:`~Local.data`.
        """
        return self.data

    @translocations.setter
    def translocations(self, tr):
        self.data = tr

    @property
    def dt(self):
        """
        `array-like`, ro property

        Translocation durations.
        """
        return self.time_data

    @dt.setter
    def dt(self, dt):
        self.time_data = dt

    @property
    def dr(self):
        """
        `array-like`, ro property

        Translocation displacements in space; `dxy` is an alias.
        """
        return self.space_data

    @dr.setter
    def dr(self, dr):
        self.space_data = dr

    @property
    def dxy(self):
        return self.space_data

    @dxy.setter
    def dxy(self, dxy):
        self.space_data = dxy

    @property
    def t(self):
        """
        `array-like`, ro property

        Initial timestamps.
        """
        if isstructured(self.origins):
            return np.asarray(self.origins[self.time_col])
        else:
            return np.asarray(self.origins[:,self.time_col])

    @property
    def r(self):
        if isstructured(self.origins):
            return np.asarray(self.origins[self.space_cols])
        else:
            return np.asarray(self.origins[:,self.space_cols])


class TrackedMolecules(Translocations):
    """
    Attributes:

        n (numpy.ndarray):
            Trajectory indices.

    """
    __slots__ = 'n',


def identify_columns(points, trajectory_col=True):
    """
    Identify columns by type.

    Arguments:
        points (array-like):
            location coordinates and trajectory index.

        trajectory_col (bool or int or str):
            trajectory column index or name,
            or ``False`` or ``None`` if there is no trajectory information is to be
            found or extracted,
            or ``True`` to automatically identify such a column.

    Returns:
        tuple: (*coord_cols*, *trajectory_col*, *get_var*, *get_point*)

    *coord_cols* is an array of column indices/names other than the trajectory column.
    If the data contains typical translocation column names such as 'dx', 'dy', 'dt',
    *coord_cols* is a tuple with, as first element, an array of location column names,
    and as second element an array of translocation column names.

    *trajectory_col* is the column index/name that contains tracking information.

    *get_var* is a callable that takes an array like `points` and column identifiers and
    returns the corresponding sub-matrix.

    *get_point* is a callable that takes an array like `points` and row indices and
    returns the corresponding sub-matrix.

    """
    _has_trajectory = trajectory_col not in [False, None]
    _traj_undefined = trajectory_col is True
    _has_delta_columns = False

    if isstructured(points):
        coord_cols = columns(points)
        if _has_trajectory:
            if _traj_undefined:
                trajectory_col = 'n'
            try:
                if isinstance(coord_cols, pd.Index):
                    coord_cols = coord_cols.drop(trajectory_col)
                else:
                    coord_cols.remove(trajectory_col)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                if _traj_undefined:
                    _has_trajectory = False
                else:
                    raise
        # assume no column name is empty
        delta_cols = [ col for col in coord_cols if col[0] == 'd' and col[1:] in coord_cols ]
        if delta_cols:
            coord_cols = [ col for col in coord_cols if col not in delta_cols ]
            coord_cols = (coord_cols, delta_cols)
    else:
        if isinstance(points, (tuple, list)):
            points = np.asarray(points)
        if not _has_trajectory:
            coord_cols = np.arange(points.shape[1])
        elif _traj_undefined:
            trajectory_col = 0
            coord_cols = np.arange(1, points.shape[1])
        else:
            coord_cols = np.r_[np.arange(trajectory_col), \
                np.arange(trajectory_col+1, points.shape[1])]
    if not _has_trajectory:
        trajectory_col = None

    if isinstance(points, pd.DataFrame):
        def get_point(a, i):
            return a[i]
        def get_var(a, j):
            return a[j]
    elif isstructured(points):
        def get_point(a, i):
            return a[i,:]
        def get_var(a, j):
            return a[j]
    else:
        def get_point(a, i):
            return a[i]
        def get_var(a, j):
            return a[:,j]

    return coord_cols, trajectory_col, get_var, get_point


def get_locations(points, index=None, coord_cols=None, get_var=None, get_point=None):
    """
    Make helpers for manipulating the point data.

    Arguments:

        points (array-like):
            location coordinates and trajectory index.

        index (ndarray or pair of ndarrrays or sparse matrix):
            point-cell association (see :class:`~tramway.tessellation.base.CellStats`
            documentation or :meth:`~tramway.tessellation.base.Tessellation.cell_index`).

    Returns:
        tuple: (*locations*, *location_cell*, *get_point*)

    *locations* are point coordinates (same format as `points` with the trajectory column
    removed.

    *location_cell* is either an array of cell indices (same size as *locations*) or
    a callable that takes a cell index and returns a boolean array with Trues
    for locations associated with the specified cell and Falses elsewhere.

    *get_point* is a callable that takes an array like `points` or `locations` and row indices
    and returns the corresponding rows in the same format.

    """
    if coord_cols is None:
        coord_cols, _, get_var, get_point = identify_columns(points)
    locations = get_var(points, coord_cols)
    #location_cell = format_cell_index(index, 'array', nearest_cell, (locations.shape[0],))
    if index is None or isinstance(index, np.ndarray):

        location_cell = index

    elif isinstance(index, tuple):

        _point, _cell = index
        loc_count = locations.shape[0]

        def __associated__(cell):
            """
            Location-cell association.

            Arguments:

                cell (int):
                    cell index.

            Returns:

                bool or ndarray:
                    True for locations associated to cell `cell`,
                    False otherwise.
            """
            _in = np.zeros(loc_count, dtype=bool)
            _in[_point[_cell == cell]] = True
            return _in

        location_cell = __associated__

    else:#if sparse.issparse(index):
        assert sparse.issparse(index)

        try:
            index = index.tocsc(copy=True)
        except TypeError: # not already a CSC matrix; copy is implicit
            index = index.tocsc()

        def __association__(cell):
            """
            Location-cell association.

            Arguments:

                cell (int):
                    cell index.

            Returns:

                any:
                    scalar value or ndarray which elements evaluate to
                    True for locations associated to cell `cell`,
                    False otherwise.
            """
            return index.getcol(cell).todense()

        location_cell = __association__

    return locations, location_cell, get_point


def get_translocations(points, index=None, coord_cols=None, trajectory_col=True,
        get_var=None, get_point=None):
    """
    Identify translocations as initial and final points.

    Arguments:
        points (array-like):
            location coordinates and trajectory index.

        index (ndarray or pair of ndarrays or sparse matrix):
            point-cell association (see :class:`~tramway.tessellation.base.CellStats`
            documentation or :meth:`~tramway.tessellation.base.Tessellation.cell_index`).

    Returns:
        tuple: (*initial_point*, *final_point*, *initial_cell*, *final_cell*, *get_point*)

    *initial_point* and *final_point* are point coordinates (same format as `points` with the
    trajectory index column removed) respectively of the initial and final displacement locations.

    *initial_cell* and *final_cell* are either arrays of cell indices (same size as *initial_point*)
    or callables that take a cell index and return a boolean array with Trues
    for displacements/translocations associated with the specified cell and Falses elsewhere.

    *get_point* is a callable that takes an array like `points` and row indices and returns
    the corresponding rows in the same format.

    """
    if coord_cols is None:
        coord_cols, trajectory_col, get_var, get_point = identify_columns(points, trajectory_col)
    if isinstance(coord_cols, tuple):
        coord_cols, delta_cols = coord_cols
    else:
        delta_cols = []

    if delta_cols:
        # translocation "coordinates"
        deltas = get_var(points, delta_cols)
        # location coordinates
        points = get_var(points, coord_cols)
        # fake `final`
        initial = final = ~np.any(np.isnan(np.asarray(deltas)), axis=1)#np.ones(points.shape[0], dtype=bool)
        # note: the destination cell will be undefined
    else:
        if trajectory_col is None:
            raise ValueError('cannot find trajectory indices')

        # trajectory index
        n = np.asarray(get_var(points, trajectory_col))
        # point coordinates
        points = get_var(points, coord_cols)

        initial = np.diff(n, axis=0) == 0
        final = np.r_[False, initial]
        initial = np.r_[initial, False]

    if index is None:

        initial_cell = final_cell = None

    elif isinstance(index, np.ndarray):

        initial_cell = index[initial]
        final_cell = index[final]

    elif isinstance(index, tuple):

        _point, _cell = index
        if _point.size == 0:
            raise ValueError('no data points found')
        _unique = np.unique(_point)
        loc_count = max(initial.size, _unique[-1]+1) # should be enough
        transloc_count = np.sum(initial)

        def __f__(termination):
            _loc_ok = np.zeros(loc_count, dtype=bool)
            _loc_ok[_unique] = True
            _transloc_ok = _loc_ok[termination]
            _transloc = np.arange(transloc_count)
            _transloc[~_transloc_ok] = -1
            _loc = np.full(loc_count, -1, dtype=int)
            _loc[termination] = _transloc
            _loc = _loc[_point]
            if not np.any(_transloc_ok):
                raise ValueError('no translocations available')
            def __associated__(cell):
                """
                Translocation-cell association.

                Arguments:

                    cell (int):
                        cell index.

                Returns:

                    bool or numpy.ndarray:
                        True for translocations associated to cell `cell`,
                        False otherwise.
                """
                _in = np.zeros(transloc_count, dtype=bool)
                _ok = _loc[_cell == cell]
                _in[_ok[0<=_ok]] = True
                return _in
            return __associated__

        initial_cell = __f__(initial)
        final_cell = __f__(final)
        # debug
        #_c = 0
        #pts = points[['x', 'y']].values
        #_pts = pts[_point[_cell==_c]]
        #print((_pts.min(axis=0), _pts.max(axis=0)))
        #_pts = pts[initial][initial_cell(_c)]
        #print((_pts.min(axis=0), _pts.max(axis=0)))

    else:#if sparse.issparse(index):
        assert sparse.issparse(index)

        try:
            index = index.tocsc(copy=True)
        except TypeError: # not already a CSC matrix; copy is implicit
            index = index.tocsc()
        loc_count = index.shape[0]
        transloc_count = np.sum(initial)

        def __f__(termination):
            _transloc = np.full(loc_count, -1, dtype=int)
            _transloc[termination] = np.arange(transloc_count)
            if np.all(_transloc==-1):
                raise ValueError('no translocations available')
            def __association__(cell):
                """
                Translocation-cell association.

                Arguments:

                    cell (int):
                        cell index.

                Returns:

                    any:
                        scalar value or ndarray which elements evaluate to
                        True for translocations associated to cell `cell`,
                        False otherwise.
                """
                _in = np.zeros(transloc_count, dtype=bool)
                _ok = _transloc[index.indices[index.indptr[cell]:index.indptr[cell+1]]]
                _in[_ok[0<=_ok]] = True
                return _in
            return __association__

        initial_cell = __f__(initial)
        final_cell = __f__(final)

    initial_point = get_point(points, initial)
    assert not np.any(np.isnan(np.asarray(initial_point)))

    if delta_cols:
        # if deltas are available, the destination cell is undefined
        if callable(final_cell):
            final_cell = lambda cell: []
        elif final_cell is not None:
            final_cell[...] = -1

        deltas = get_point(deltas, initial)
        cols = deltas.columns
        deltas.columns = [ col[1:].lstrip() for col in cols ]
        final_point = initial_point + deltas
        if np.any(np.isnan(np.asarray(final_point))):
            for col in initial_point.columns:
                if col not in deltas.columns:
                    if col[0] == 'd':
                        raise ValueError("no column corresponding to delta '{}'".format(col))
                    else:
                        raise ValueError("no delta column corresponding to '{}'".format(col))
            assert False # final_point contains NaNs
    else:
        final_point = get_point(points, final)
        assert not np.any(np.isnan(np.asarray(final_point)))
    assert initial_point.shape == final_point.shape

    return initial_point, final_point, initial_cell, final_cell, get_point


def distributed(cells, new_cell=None, new_group=FiniteElements, fuzzy=None,
        new_cell_kwargs={}, new_group_kwargs={}, fuzzy_kwargs={},
        new=None, include_empty_cells=False, verbose=False):
    """
    Build a `FiniteElements`-like object from a :class:`~tramway.tessellation.base.CellStats` object.

    Cells with no (trans-)locations are discarded in addition to those with null or negative label.

    Arguments:

        cells (CellStats): tessellation and partition; must contain (trans-)location data.

        new_cell (callable): cell constructor; default is :class:`Locations` or
            :class:`Translocations` depending on the data in `cells`.

        new_group (callable): constructor for groups of cell; default is :class:`FiniteElements`.

        fuzzy (callable): (trans-)location-cell weighting function; the default for
            translocations considers the initial cell, no weight.

        new_cell_kwargs (dict): keyword arguments for `new_cell`.

        new_group_kwargs (dict): keyword arguments for `new_group`.

        fuzzy_kwargs (dict): keyword arguments for `fuzzy`.

        new (callable): legacy argument; use `new_group` instead.

        include_empty_cells (bool): do not discard cells with no (trans-)locations.

    `fuzzy` takes a :class:`~tramway.tessellation.base.Tessellation` object, a cell index (`int`),
    location coordinates (array-like), cell indices (array-like or callable) and the *get_point*
    function returned by `get_locations` or `get_translocations`.

    If the data are translocations, `fuzzy` takes a pair of arrays as location coordinates (initial
    and final locations respectively) and a pair of arrays or callables as cell indices (initial
    and final cells respectively).

    `fuzzy` returns an array of booleans or values of any scalar type that can be evaluated logically.
    The returned array contains as many elements as input (trans-)locations.

    """
    if new is not None:
        # `new` is for backward compatibility
        warn('`new` is deprecated in favor of `new_group`', DeprecationWarning)
        new_group = new
    if not isinstance(cells, tessellation.CellStats):
        raise TypeError('`cells` is not a `CellStats`')
    if cells.points.size == 0:
        raise ValueError('no data points found')
    if isinstance(cells.cell_index, tuple):
        if len(cells.cell_index[0]) == 0:
            raise ValueError('not any point assigned')

    # format (trans-)locations
    coord_cols, trajectory_col, get_var, get_point = identify_columns(cells.points)
    has_precomputed_deltas = isinstance(coord_cols, tuple)
    if has_precomputed_deltas:
        _nan = np.any(np.isnan(np.asarray(get_var(cells.points, coord_cols[0]))))
    else:
        _nan = np.any(np.isnan(np.asarray(cells.points)))
    if _nan:
        raise ValueError('NaN in location data')
    precomputed = ()
    if new_cell is None:
        are_tracked_molecules = trajectory_col is not None and isinstance(cells.points, pd.DataFrame)
        are_translocations = trajectory_col is not None or has_precomputed_deltas
    else:
        if issubclass(new_cell, Locations):
            are_translocations = are_tracked_molecules = False
        elif issubclass(new_cell, Translocations):
            are_translocations = True
            are_tracked_molecules = issubclass(new_cell, TrackedMolecules)
        else:
            raise TypeError('`new_cell` is neither `Locations` nor `Translocations`')
    if are_translocations:
        precomputed = (coord_cols, trajectory_col, get_var, get_point)
    else:
        precomputed = (coord_cols, get_var, get_point)
    if are_translocations:
        initial_point, final_point, initial_cell, final_cell, get_point = \
            get_translocations(cells.points, cells.cell_index, *precomputed)
        if new_cell is None:
            if are_tracked_molecules:
                new_cell = TrackedMolecules
            else:
                new_cell = Translocations
        fuzzy_args = ((initial_point, final_point), (initial_cell, final_cell), get_point)
    else:
        locations, location_index, get_point = \
            get_locations(cells.points, cells.cell_index, *precomputed)
        fuzzy_args = (locations, location_index, get_point)
        if new_cell is None:
            new_cell = Locations

    # assign/weight (trans-)locations to cells
    if fuzzy is None:
        if are_translocations:
            def f(tessellation, cell, translocations, translocation_cell, get_point):
                initial_point, final_point = translocations
                initial_cell, final_cell = translocation_cell
                if callable(initial_cell):
                    return initial_cell(cell)
                else:
                    return initial_cell == cell
        else:
            def f(tesselation, cell, locations, location_cell, get_point):
                if callable(location_cell):
                    return location_cell(cell)
                else:
                    return location_cell == cell
        fuzzy = f

    # time and space columns in (trans-)locations array
    def _bool(a):
        try:
            return bool(a.size)
        except AttributeError:
            return bool(a)
    if isstructured(cells.points):
        time_col = 't'
    else:
        time_col = cells.points.shape[1] - 1
    if cells.tessellation.scaler is not None and _bool(cells.tessellation.scaler.columns):
        space_cols = cells.tessellation.scaler.columns
    elif isstructured(cells.points):
        space_cols = columns(cells.points)
        if 'n' in space_cols:
            not_space = ['n', time_col]
        else:
            not_space = [time_col]
        if isinstance(space_cols, pd.Index):
            space_cols = space_cols.drop(not_space)
        else:
            space_cols = [ c for c in space_cols if c not in not_space ]
    else:
        if time_col == cells.points.shape[1] - 1:
            space_cols = np.arange(time_col)
        else:
            space_cols = np.ones(cells.points.shape[1], dtype=bool)
            space_cols[time_col] = False
            space_cols, = space_cols.nonzero()
    # remove delta columns
    if has_precomputed_deltas:
        _, delta_cols = coord_cols
        space_cols = [ col for col in space_cols if col not in delta_cols ]

    # pre-select cells
    if include_empty_cells:
        if cells.tessellation.cell_label is None:
            J = np.ones(cells.location_count.shape, dtype=bool)
        else:
            J = 0 < cells.tessellation.cell_label
    else:
        if cells.tessellation.cell_label is None:
            J = 0 < cells.location_count
        else:
            J = np.logical_and(0 < cells.location_count, 0 < cells.tessellation.cell_label)

    # select (with the fuzzy filter) and pre-build cells
    _fuzzy, data, hull = {}, {}, {}
    if are_translocations:
        extra = {}
    for j, ok in enumerate(J): # for each cell
        if not ok:
            continue

        # find (trans-)locations for cell j
        i = fuzzy(cells.tessellation, j, *fuzzy_args, **fuzzy_kwargs)
        if i.dtype in (bool, np.bool, np.bool8, np.bool_):
            _fuzzy[j] = None
        else:
            _fuzzy[j] = i[i != 0]
            i = i != 0

        if are_translocations:
            _origin = get_point(initial_point, i)
            _destination = get_point(final_point, i)
            if has_precomputed_deltas:
                assert np.all(_origin.index == _destination.index)
                __origin = _origin
            else:
                __origin = _origin.copy() # make copy
                __origin.index += 1
                try:
                    _ok = np.array([i in _destination.index for i in __origin.index])
                    _ok &= np.array([i in __origin.index for i in _destination.index])
                except TypeError:
                    J[j] = False
                    continue
                _origin = get_point(_origin, _ok)
                __origin = get_point(__origin, _ok)
                _destination = get_point(_destination, _ok)
            _points = _origin # for convex hull; ideally not only origins
            points = _destination - __origin # translocations
        else:
            points = _points = get_point(locations, i) # locations

        assert not np.any(np.isnan(np.asarray(points)))

        # convex hull
        _points = np.asarray(get_var(_points, space_cols))
        try:
            hull[j] = cells.tessellation.cell_volume[j]
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            #try:
            #    hull[j] = scipy.spatial.qhull.ConvexHull(_points)
            #except (KeyboardInterrupt, SystemExit):
            #    raise
            #except Exception as e:
            raise
            warn(str(e), RuntimeWarning)

        if not include_empty_cells and points.size == 0:
            J[j] = False
        else:
            if are_tracked_molecules:
                _trajectory_index = cells.points['n'][_origin.index].values
                extra[j] = (_origin, _destination, _trajectory_index)
            elif are_translocations:
                extra[j] = (_origin, _destination)
            data[j] = points

    # simplify the adjacency matrix
    try:
        _adjacency = cells.tessellation.diagonal_adjacency
    except AttributeError:
        _adjacency = None
    _adjacency = cells.tessellation.simplified_adjacency(adjacency=_adjacency, label=J, format='csr')
    ## reweight each row i as 1/n_i where n_i is the degree of cell i
    #n = np.diff(_adjacency.indptr)
    #_adjacency.data = np.repeat(1.0 / np.maximum(1, n), n)

    # build every cells
    J, = np.nonzero(J)
    _cells = OrderedDict()
    for j in J: # for each cell
        try:
            center = cells.tessellation.cell_centers[j]
        except AttributeError:
            center = span = None
        else:
            adj = _adjacency[j].indices
            span = cells.tessellation.cell_centers[adj] - center

        # make cell object
        args = ()
        _volume = None
        try:
            _hull = hull[j]
            _hull.simplices
        except AttributeError:
            _volume = _hull
        except KeyError:
            pass
        else:
            args = (_hull,)
        _cells[j] = new_cell(j, data[j], center, span, *args, **new_cell_kwargs)
        _cells[j].time_col = time_col
        _cells[j].space_cols = space_cols
        if _volume:
            _cells[j].volume = _volume
        if are_translocations:
            try:
                _cells[j].origins = extra[j][0]
            except AttributeError: # `_cells` does not have the `origins` attribute
                pass
            try:
                _cells[j].destinations = extra[j][1]
            except AttributeError: # `_cells` does not have the `destinations` attribute
                pass
            try:
                _cells[j].n = extra[j][2]
            except AttributeError: # `_cells` does not have the `n` attribute
                pass
            except IndexError: # extra[j] has two elements
                pass
        try:
            _cells[j].fuzzy = _fuzzy[j]
        except AttributeError: # `_cells` does not have the `fuzzy` attribute
            pass

    # perform a few last checks
    if verbose:
        if hull:
            for j in J:
                if j not in hull:
                    print('cannot compute convex hull for cell {}'.format(j))
        else:
            print('cannot compute convex hulls')

    #print(sum([ c.tcount == 0 for c in _cells.values() ]))
    self = new_group(_cells, _adjacency, **new_group_kwargs)
    self.tcount = cells.points.shape[0]
    #self.dim = cells.points.shape[1]
    self.ccount = self.adjacency.shape[0]

    return self



class DistributeMerge(FiniteElements):

    __slots__ = ('merged', )

    def __init__(self, cells, adjacency, min_location_count=None):
        if min_location_count is None:
            raise ValueError('`min_location_count` is not defined')
        FiniteElements.__init__(self, cells, adjacency)
        count = np.array([ len(cells[i]) for i in cells ])
        index, = (count < min_location_count).nonzero()
        ordered_index = np.argsort(count[index])
        ordered_index = index[ordered_index]
        index = set(index)
        merged = dict()
        for i in ordered_index[::-1]:
            js = [ j for j in self.neighbours(i).tolist() if j not in index ]
            if not js:
                warn('no large-enough neighbours for cell {:d}'.format(i), RuntimeWarning)
                continue
            dr = np.vstack([ cells[j].center for j in js ]) - cells[i].center#[np.newaxis,:]
            j = js[np.argmin(np.sum(dr * dr, axis=1))]
            j = merged.get(j, j)
            # merge cells i and j
            # attributes: center(?),time_data,space_data,span,boundary,fuzzy,origins,destinations
            cells[j].data = (np.vstack((cells[j].space_data, cells[i].space_data)),
                    np.r_[cells[j].time_data, cells[i].time_data])
            if cells[j].span is not None:
                try:
                    cells[j].span = np.vstack((cells[j].span, cells[i].span))
                except (AttributeError, TypeError):
                    pass
            if cells[j].boundary is not None:
                # cannot merge properly
                try:
                    cells[j].boundary = np.vstack((cells[j].boundary, cells[i].boundary))
                except (AttributeError, TypeError):
                    pass
            try:
                cells[j].fuzzy = np.r_[cells[j].fuzzy, cells[i].fuzzy]
            except (AttributeError, ValueError):
                pass
            try:
                cells[j].origins = np.vstack((cells[j].origins, cells[i].origins))
                cells[j].destinations = np.vstack((cells[j].destinations, cells[i].destinations))
            except AttributeError:
                pass
            #
            merged[i] = j
            index.remove(i)
        # clear and register the merged cells
        discarded = set(ordered_index)
        self.cells = type(cells)([(i, cells[i]) for i in cells if i not in discarded])
        self.merged = dict()
        for i, j in merged.items():
            try:
                self.merged[j].add(i)
            except KeyError:
                self.merged[j] = set([i])
        # modify the adjacency matrix
        adjacency = sparse.lil_matrix(adjacency.shape, dtype=adjacency.dtype)
        for i in range(adjacency.shape[0]):
            if i in discarded:
                continue
            js = set(self.neighbours(i).tolist())
            try:
                ks = self.merged[i]
            except KeyError:
                pass
            else:
                for k in ks:
                    js |= set(self.neighbours(k).tolist())
            js -= discarded
            for j in js:
                adjacency[i,j] = True
                adjacency[j,i] = True
        self.adjacency = adjacency.tocsr()
        assert set(self.cells.keys()) == set((self.adjacency.indptr[1:] - self.adjacency.indptr[:-1]).nonzero()[0].tolist())
        assert set(self.cells.keys()) == set(self.adjacency.indices.tolist())


class Maps(Lazy):
    """
    Basic container for maps, posteriors and the associated input parameters used to generate
    the maps.

    Attributes `distributed_translocations`, `partition_file`, `tessellation_program`, `version`
    are deprecated and will be removed.

    Functional dependencies:

    * setting `maps` unsets `_features`

    """
    __lazy__ = Lazy.__lazy__ + ('_features',)

    def __init__(self, maps, mode=None, posteriors=None):
        Lazy.__init__(self)
        self.maps = maps
        self.mode = mode
        self.min_diffusivity = None
        self.localization_error = None
        self.diffusivity_prior = None
        self.potential_prior = None
        self.jeffreys_prior = None
        self.extra_args = None
        self.distributed_translocations = None # legacy attribute
        self.partition_file = None # legacy attribute
        self.tessellation_param = None # legacy attribute
        self.version = None # legacy attribute
        self.runtime = None
        self.posteriors = posteriors

    @property
    def maps(self):
        return self._maps

    @maps.setter
    def maps(self, m):
        if m is None:
            self._features = None
        else:
            if not isinstance(m, pd.DataFrame):
                raise TypeError('DataFrame expected')
            _err = TypeError('not any column defined')
            if isinstance(m.columns, (tuple, list)):
                if not m.columns:
                    raise _err
            elif m.columns.size == 0:
                raise _err
            self._features = splitcoord(m.columns)
        self._maps = m

    @property
    def features(self):
        """
        `list` or ``None``

        List of different feature names without the coordinate suffix.

        For example, if `maps` has columns *diffusivity*, *force x* and *force y*, `features`
        will return *diffusivity* and *force*.
        """
        if self._features is None:
            return None
        else:
            return list(self._features.keys())

    @property
    def variables(self):
        """
        Alias for `features`.
        """
        warn('`variables` is deprecated; use `features` instead', PendingDeprecationWarning)
        return self.features

    @property
    def _variables(self):
        """
        Alias for `_features`.
        """
        warn('`_variables` is deprecated; use `_features` instead', PendingDeprecationWarning)
        return self._features

    def __nonzero__(self):
        return self.maps is not None

    def __len__(self):
        return 0 if self._features is None else len(self._features)

    def __contains__(self, feature_name):
        return feature_name in self._features

    def __getitem__(self, feature_name):
        try:
            if isinstance(feature_name, str):
                return self.maps[dict(self._features)[feature_name]]
            else:
                import itertools
                return self.maps[list(itertools.chain(*[
                            dict(self._features)[ftr] for ftr in feature_name
                        ]))]
        except (TypeError, KeyError):
            raise KeyError("no such mapped feature: '{}'".format(feature_name))

    def __setitem__(self, feature_name, val):
        if isinstance(val, np.ndarray):
            if val.size == self.maps.shape[0]:
                self.maps[feature_name] = val
                self._features = None
            else:
                raise ValueError('wrong array size')
        else:
            raise NotImplementedError('only numpy ndarrays are supported')

    def sub(self, ix, reindex=False):
        """
        Sub-map.

        Operates like `loc` on a `DataFrame`.

        Arguments:

            ix (slice or array-like):
                cell indices or boolean array.

            reindex (bool):
                make index range from 0 without missing integer.

        Returns:

            tramway.inference.base.Maps: sub-map.

        """
        sub_map = copy(self)
        if isinstance(ix, slice):
            ix = np.arange(ix.start, ix.stop, ix.step)
        sub_map.maps = self.maps.loc[ix]
        if reindex:
            sub_map.maps.index = np.arange(sub_map.maps.index.size)
        return sub_map

    def __str__(self):
        attrs = { k: v for k, v in self.__dict__.items() if not (k[0] == '_' or v is None) }
        v = self.features
        if v is not None:
            attrs['features'] = v
        attrs['maps'] = type(self.maps)
        l = max(len(k) for k in attrs)
        s = '\n'.join([ '{}:{} {}'.format(k, ' '*(l-len(k)), str(v)) for k, v in attrs.items() ])
        return s

    def defattr(self, attr, val):
        count = 0
        while True:
            try:
                getattr(self, attr)
            except AttributeError:
                break
            else:
                if count:
                    attr = attr[:-1] + str(count)
                else:
                    if attr[-1] in '0123456789' or '_' in attr:
                        attr = attr + '_0'
                    else:
                        attr = attr + '0'
                count += 1
        setattr(self, attr, val)



class OptimizationWarning(RuntimeWarning):
    pass


class DiffusivityWarning(RuntimeWarning):
    def __init__(self, diffusivity, lower_bound=None):
        self.diffusivity = diffusivity
        self.lower_bound = lower_bound

    def __repr__(self):
        if self.lower_bound is None:
            return 'DiffusivityWarning({})'.format(self.diffusivity)
        else:
            return 'DiffusivityWarning({}, {})'.format(self.diffusivity, self.lower_bound)

    def __str__(self):
        if self.lower_bound is None:
            return 'wrong diffusivity value: {}'.format(self.diffusivity)
        else:
            return 'diffusivity too low: {} < {}'.format(self.diffusivity, self.lower_bound)


def smooth_infer_init(cells, min_diffusivity=None, jeffreys_prior=None, **kwargs):
    """
    Initialize the diffusivity array for translocations.

    This function is a collection of checks and initial computations suitable for *infer* functions
    in inference plugins.

    Arguments:

        cells (FiniteElements): first input argument of the *infer* function.

        min_diffusivity (float): minimum allowed diffusivity value.

        jeffreys_prior (bool): activate Jeffreys' prior.

    Returns:

    * *index* (:class:`numpy.ndarray`) -- cell indices corresponding to arrays *n*, *dt_mean* and *D_initial*.
    * *reverse_index* (:class:`numpy.ndarray`) -- element indices in arrays *index*, *n*, *dt_mean*, *D_initial* for each cell index.
    * *n* (:class:`numpy.ndarray`) -- translocation count for each cell.
    * *dt_mean* (:class:`numpy.ndarray`) -- mean translocation duration for each cell.
    * *D_initial* (:class:`numpy.ndarray`) -- initial diffusivity array.
    * *min_diffusivity* (:class:`float` or ``None``) -- minimum diffusivity value; the returned value depends on `jeffreys_prior`.
    * *D_bounds* (:class:`list`) -- list of (lower bound, upper bound) couples; suitable as *bounds* input argument for the :func:`scipy.optimize.minimize` function.
    * *border* (:class:`numpy.ndarray`) -- MxD boolean matrix with M the number of cells and D the number of space dimensions

    """
    # initial values and sanity checks
    index, n, dt_max, dt_mean, D_initial, border = [], [], [], [], [], []
    reverse_index = np.full(cells.adjacency.shape[0], -1, dtype=int)

    j = 0
    for i in cells:
        cell = cells[i]

        #assert i == cell.index # NO!
        if not bool(cell):
            raise ValueError('empty cells')

        # sanity checks
        if cell.dr.shape[1] == 0:
            raise ValueError('translocation array has no column')
        if cell.dt.shape[1:]:
            raise ValueError('time deltas are structured in multiple dimensions')
        if np.any(np.isnan(cell.dt)):
            raise ValueError('time delta is nan')
        # ensure that translocations are properly oriented in time
        if not np.all(0 < cell.dt):
            warn('translocation dts are not all positive', RuntimeWarning)
            cell.dr[cell.dt < 0] *= -1.
            cell.dt[cell.dt < 0] *= -1.

        # check cell i has neighbours
        try:
            adjacent = cells.adjacency.indices[cells.adjacency.indptr[i]:cells.adjacency.indptr[i+1]]
            adjacent = [ c for c in adjacent if cells[c] ]
            if not adjacent:
                continue
        except ValueError:
            continue

        # initialize the local diffusivity parameter
        dt_max_i = np.max(cell.dt)
        dt_mean_i = np.mean(cell.dt)
        D_initial_i = np.mean(cell.dr * cell.dr) / (2. * dt_mean_i)
        #

        index.append(i)
        reverse_index[i] = j
        j += 1
        n.append(float(len(cell)))
        dt_max.append(dt_max_i)
        dt_mean.append(dt_mean_i)
        D_initial.append(D_initial_i)

        # border
        try:
            if cell.center is None:
                warn('missing cell center', RuntimeWarning)
                border.append(np.zeros(cell.dim, dtype=np.bool_))
            else:
                adjacent = np.vstack([ cells[c].center for c in adjacent if cells[c] ])
                border.append(np.logical_or(
                    np.max(adjacent, axis=0) <= cell.center,
                    cell.center <= np.min(adjacent, axis=0)
                    )) # to be improved
        except ValueError:
            border.append(None)

    n, dt_mean, D_initial = np.array(n), np.array(dt_mean), np.array(D_initial)

    if min_diffusivity is None:
        noise_dt = kwargs['sigma2']
        D_bounds = [( (1e-16 -noise_dt)/dt_max_i, None ) for dt_max_i in dt_max ]
        min_diffusivity = 0
    else:
        if min_diffusivity is False:
            min_diffusivity = None
        D_bounds = [(min_diffusivity, None)] * D_initial.size

    try:
        border = np.vstack(border)
    except:
        border = None

    return index, reverse_index, n, dt_mean, D_initial, min_diffusivity, D_bounds, border


__all__ = ['Local', 'Distributed', 'Cell', 'Locations', 'Translocations', 'Maps',
    'FiniteElement', 'FiniteElements',
    'identify_columns', 'get_locations', 'get_translocations', 'distributed',
    'TrackedMolecules', 'DistributeMerge',
    'DiffusivityWarning', 'OptimizationWarning', 'smooth_infer_init']

