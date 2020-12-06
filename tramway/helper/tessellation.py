# -*- coding: utf-8 -*-

# Copyright © 2017-2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from ..core import *
from ..core.hdf5 import *
from ..core.analyses import abc
from ..tessellation import *
from .base import *
from warnings import warn
import six
import traceback
import itertools
# no module-wide matplotlib import for head-less usage of `tessellate`
# in the case matplotlib's backend is interactive


class Tessellate(Helper):
    def __init__(self):
        Helper.__init__(self)
        self._label_is_output = True
        self.plugins = plugins
        self.tessellation_kwargs = {}
        self.partition_kwargs = {}
        self.time_window_kwargs = {}
        self.reassignment_kwargs = {}

    @property
    def _partition_kwargs(self):
        if self.reassignment_kwargs:
           # reassigning is based on true partitions (Voronoi),
           # therefore over- and under-sampling arguments should be silenced
           over_sampling_args = ('knn', 'radius', 'time_knn')
           return { kw: arg for kw, arg in self.partition_kwargs.items() \
                        if kw not in over_sampling_args }
        else:
           return self.partition_kwargs

    @_partition_kwargs.setter
    def _partition_kwargs(self, kwargs):
        self.partition_kwargs = kwargs

    def prepare_data(self, input_data, labels=None, types=None, metadata=True, \
            verbose=None, scaling=False, time_scale=None, **kwargs):

        if (isinstance(input_data, six.string_types) and os.path.isdir(input_data)) or \
                (self.are_multiple_files(input_data) and input_data and all(os.path.exists(_d) for _d in input_data)):
            input_data, self.input_file = load_xyt(input_data, return_paths=True, **kwargs)
            self.metadata['datafile'] = self.input_file if self.input_file[1:] else self.input_file[0]
        elif isinstance(input_data, six.string_types):
            if not os.path.exists(input_data):
                raise OSError('file not found: {}'.format(input_data))
            try:
                input_data, self.input_file = load_xyt(input_data, return_paths=True, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except (UnicodeDecodeError, pd.errors.ParserError): # rwa file
                pass
            else:
                assert self.input_file and not self.input_file[1:]
                self.metadata['datafile']= self.input_file[0]

        if labels is None:
            labels = self.input_label
        nesting = labels is not None
        if types is None and nesting:
            types = Partition

        data = Helper.prepare_data(self, input_data, labels, types, metadata, verbose, **kwargs)

        if nesting:
            assert isinstance(data, tuple) and not data[1:]
            data, = data
            if isinstance(data, Partition):
                self.input_partition = data
                self.xyt_data = data.points
            else:
                if isinstance(data, pd.DataFrame):
                    msg = '(trans-)location data'
                else:
                    msg = 'the following datatype: {}'.format(type(data))
                raise TypeError('nesting tessellations does not apply to {}'.format(msg))
            if self.input_label is None:
                self.input_label = self.find(data)
        elif isinstance(data, abc.Analyses):
            self.xyt_data = data.data
        else:
            self.xyt_data = data

        if scaling:
            if scaling is True:
                scaling = 'whiten'
            self.scaler = dict(whiten=whiten, unit=unitrange)[scaling]()
        else:
            self.scaler = None

        self.colnames = ['x', 'y']
        if 'z' in self.xyt_data:
            self.colnames.append('z')
        if time_scale:
            self.colnames.append('t')
            self.scaler.factor = [('t', time_scale)]

        return data

    def output_file(self, output_file=None, suffix=None, extension=None):
        none_defined = output_file is None and suffix is None and extension is None
        output_file = Helper.output_file(self, output_file, suffix, extension)
        if none_defined and self.input_file:
            output_file = os.path.splitext(output_file)[0] + '.rwa'
        return output_file

    def plugin(self, method, plugins=None, verbose=None):
        Helper.plugin(self, method, plugins, verbose)
        self.constructor = self.setup['make']
        if isinstance(self.constructor, str):
            self.constructor = getattr(self.module, self.setup['make'])

    def standard_parameters(self, \
            distance=None, ref_distance=None, \
            rel_min_distance=None, rel_avg_distance=None, rel_max_distance=None, \
            min_location_count=None, avg_location_count=None, max_location_count=None, \
            knn=None, radius=None, time_knn=None, verbose=None):
        if verbose is None:
            verbose = self.verbose
        transloc_length = min_distance = avg_distance = max_distance = None
        if ref_distance is None and distance is not None:
            # `distance` is only for compatibility with the tramway commandline
            ref_distance = distance
        nesting = self.input_label is not None
        if nesting and self.input_partition.param is not None:
            prm = self.input_partition.param.get('tessellation', self.input_partition.param)
            if not ref_distance:
                ref_distance = prm.get('ref_distance', None)
            min_distance = prm.get('min_distance', min_distance)
            avg_distance = prm.get('avg_distance', avg_distance)
            max_distance = prm.get('max_distance', max_distance)
        # former default values for `rel_min_distance` and `rel_avg_distance`
        if rel_min_distance is None and min_distance is None:
            rel_min_distance = .8
        if rel_avg_distance is None and avg_distance is None:
            rel_avg_distance = 2.

        if ref_distance is None:
            if 'n' not in self.xyt_data.columns:
                raise ValueError('please specify ref_distance')
            transloc_xy = np.asarray(translocations(self.xyt_data))
            if transloc_xy.shape[0] == 0:
                raise ValueError('no translocation found')
            transloc_length = np.nanmean(np.sqrt(np.sum(transloc_xy * transloc_xy, axis=1)))
            if verbose:
                print('average translocation distance: {}'.format(transloc_length))
            ref_distance = transloc_length
        if rel_min_distance is not None:
            min_distance = rel_min_distance * ref_distance
        if rel_avg_distance is not None:
            avg_distance = rel_avg_distance * ref_distance
        if rel_max_distance is not None:
            # applies only to KDTreeMesh and Kohonen
            max_distance = rel_max_distance * ref_distance

        if min_location_count is None: # former default value: 20
            if knn is None and radius is None:
                min_location_count = 20
        n_pts = float(self.xyt_data.shape[0])
        if min_location_count:
            min_probability = float(min_location_count) / n_pts
        else:
            min_probability = None
        if avg_location_count is None:
            if min_location_count is None:
                avg_location_count = 80 # former default value
            else:
                avg_location_count = 4 * min_location_count
        if avg_location_count:
            avg_probability = float(avg_location_count) / n_pts
        else:
            avg_probability = None
        if max_location_count:
            max_probability = float(max_location_count) / n_pts
        else:
            max_probability = None

        params = dict( \
            ref_distance=ref_distance, \
            min_distance=min_distance, \
            avg_distance=avg_distance, \
            max_distance=max_distance, \
            min_probability=min_probability, \
            avg_probability=avg_probability, \
            max_probability=max_probability, \
            min_location_count=min_location_count, \
            avg_location_count=avg_location_count, \
            max_location_count=max_location_count, \
            )

        return params


    def parse_args(self, params, knn=None, radius=None,
            rel_max_size=None, rel_max_volume=None, \
            time_window_duration=None, time_window_shift=None, \
            enable_time_regularization=False, time_window_options=None, \
            min_n=None, time_knn=None, kwargs={}):
        for ignored in ['max_level']:
            try:
                if self.tessellation_kwargs[ignored] is None:
                    del self.tessellation_kwargs[ignored]
            except KeyError:
                pass
        params = dict(params)
        params.update(self.tessellation_kwargs)
        for key in self.setup.get('make_arguments', {}):
            try:
                param = params[key]
            except KeyError:
                pass
            else:
                self.tessellation_kwargs[key] = param

        self.time_window_kwargs = {}
        if time_window_duration:
            self.time_window_kwargs['duration'] = time_window_duration
        if time_window_shift:
            self.time_window_kwargs['shift'] = time_window_shift
        if time_window_options:
            time_window_options = dict(time_window_options) # copy to prevent side effects
            enable_time_regularization = time_window_options.pop(
                        'regularisable',
                        time_window_options.pop(
                            'regularizable',
                            enable_time_regularization))
            self.time_window_kwargs.update(time_window_options)
        if enable_time_regularization:
            time_dimension = self.time_window_kwargs.get('time_dimension', None)
            if time_dimension is False:
                warn('`enable_time_regularization={}` and `time_dimension={}` are not compatible'.format(enable_time_regularization, time_dimension), RuntimeWarning)
            elif time_dimension is None:
                self.time_window_kwargs['time_dimension'] = True
        if self.time_window_kwargs:
            self.time_window_kwargs['duration'] # required; fails with KeyError if missing

        ref_distance = params.get('ref_distance', None)
        _filter_f = self.partition_kwargs.get('filter', None)
        try:
            max_size = kwargs['rel_max_size']
        except KeyError:
            _filter_fg = _filter_f
        else:
            max_size *= ref_distance
            def _filter_g(voronoi, cell, points):
                return voronoi.scaler.unscale_distance(np.sqrt(np.min(np.sum(( \
                        voronoi._center[cell] - \
                        voronoi._vertices[voronoi.cell_vertices[cell]] \
                    )*2, axis=1)))) <= max_size
            _filter_fg = _filter_g if _filter_f is None \
                else lambda *a: _filter_g(*a) and _filter_f(*a)
        try:
            max_volume = kwargs['rel_max_volume']
        except KeyError:
            _filter_fgh = _filter_fg
        else:
            max_volume *= ref_distance
            def _filter_h(voronoi, cell, points):
                return voronoi.cell_volume[cell] <= max_volume
            _filter_fgh = _filter_h if _filter_fg is None \
                else lambda *a: _filter_h(*a) and _filter_fg(*a)
        try:
            max_actual_size = self.partition_kwargs.pop('rel_max_size')
        except KeyError:
            _filter_fghi = _filter_fgh
        else:
            max_actual_size *= ref_distance
            _descr = not self.partition_kwargs.get('filter_descriptors_only', True)
            def _filter_i(voronoi, cell, x):
                if _descr:
                    x = voronoi.descriptors(x, asarray=True)
                x2 = np.sum(x * x, axis=1, keepdims=True)
                d2 = x2 + x2.T - 2. * np.dot(x, x.T)
                d = np.sqrt(np.max(d2.ravel()))
                return d <= max_actual_size
            _filter_fghi = _filter_i if _filter_fgh is None \
                else lambda *a: _filter_i(*a) and _filter_fgh(*a)
        if _filter_fghi is not None:
            self.partition_kwargs['filter'] = _filter_fghi
            if 'filter_descriptors_only' not in self.partition_kwargs:
                self.partition_kwargs['filter_descriptors_only'] = True
        if not (knn is None and radius is None and time_knn is None):
            if knn is not None:
                self.partition_kwargs['knn'] = knn
            if radius is not None:
                if 'radius' in self.partition_kwargs:
                    warn('overwriting `radius`', RuntimeWarning)
                self.partition_kwargs['radius'] = radius
            if time_knn is not None:
                self.partition_kwargs['time_knn'] = time_knn
            if 'min_location_count' not in self.partition_kwargs:
                min_location_count = params['min_location_count']
                self.partition_kwargs['min_location_count'] = min_location_count
            if 'metric' not in self.partition_kwargs:
                self.partition_kwargs['metric'] = 'euclidean'
        if min_n is not None:
            self.partition_kwargs['min_location_count'] = min_n

        _options = kwargs.pop('reassignment_options', {})
        if _options:
            self.reassignment_kwargs.update(_options)
        for arg in dict(kwargs):
            if arg.startswith('reassignment_'):
                self.reassignment_kwargs[arg[13:]] = kwargs.pop(arg)
            if arg.startswith('reassign_'):
                self.reassignment_kwargs[arg[9:]] = kwargs.pop(arg)

    def tessellate(self, comment=None, verbose=None):
        if comment is None:
            comment = self.comment
            if self.comment is not None:
                self.comment = None
        if verbose is None:
            verbose = self.verbose

        nesting = self.input_label is not None

        # initialize a Tessellation object
        if nesting:
            if self.time_window_kwargs:
                raise NotImplementedError('spatial tessellation combined with time windowing is not supported yet by tessellation nesting')
            tess = NestedTessellations(self.scaler, self.input_partition, factory=self.constructor,
                **self.tessellation_kwargs)
        else:
            if self.time_window_kwargs:# and not self.reassignment_kwargs:
                # note: bin reassignment applies only to the spatial component;
                #       time windowing is carried out after reassigning bins
                import tramway.tessellation.window as window
                tess = window.SlidingWindow(**self.time_window_kwargs)
                tess.spatial_mesh = self.constructor(self.scaler, **self.tessellation_kwargs)
            else:
                tess = self.constructor(self.scaler, **self.tessellation_kwargs)

        # grow the tessellation
        if nesting:
            data = self.xyt_data
        else:
            data = self.xyt_data[self.colnames]
        tessellate_kwargs = self.tessellation_kwargs
        tessellate_hidden_kwargs = {}
        if verbose is not None:
            tessellate_hidden_kwargs['verbose'] = verbose
        tess.tessellate(data, **tessellate_kwargs, **tessellate_hidden_kwargs)

        # partition the dataset into the cells of the tessellation
        try:
            cell_index = tess.cell_index(self.xyt_data, **self._partition_kwargs)
        except MemoryError:
            if verbose:
                print(traceback.format_exc())
            warn('memory error: cannot assign points to cells', RuntimeWarning)
            cell_index = None

        self.cells = cells = Partition(self.xyt_data, tess, cell_index)

        if self.reassignment_kwargs:
            while True:
                ncells = cells.number_of_cells
                cells = self.reassign()
                if cells.number_of_cells == ncells:
                    break
                #assert self.cells.cell_index.max() < cells.number_of_cells
            if cells.number_of_cells < 10:
                warn('coarse tessellation: {} cells only'.format(cells.number_of_cells), RuntimeWarning)
            # post-reassignment step to introduce overlap and time windowing if requested
            if self.time_window_kwargs:
                cells = self.cells
                tess = cells.tessellation
            if self._partition_kwargs is not self.partition_kwargs:
                cell_index = tess.cell_index(self.xyt_data, **self.partition_kwargs)
                if isinstance(cell_index, tuple) and len(cell_index[0]) == 0:
                    print('time_window_kwargs', self.time_window_kwargs)
                    print('partition_kwargs', self.partition_kwargs)
                    print('cell_index', cell_index)
                    print('data.shape', self.xyt_data.shape, 'number_of_cells', tess.number_of_cells)
                    raise ValueError('not any point assigned')
                #self.cells = cells = Partition(self.xyt_data, tess, cell_index)
                self.cells.cell_index = cell_index

        # store some parameters together with the partition
        method = self.name
        cells.param['method'] = method
        if self.time_window_kwargs:
            cells.param['time_window'] = self.time_window_kwargs
        if self.tessellation_kwargs:
            cells.param['tessellation'] = self.tessellation_kwargs
        if self.partition_kwargs:
            cells.param['partition'] = self.partition_kwargs
        if self.reassignment_kwargs:
            cells.param['reassignment'] = self.reassignment_kwargs

        # insert the resulting analysis in the analysis tree
        if self.analyses is not None:
            self.insert_analysis(cells, comment=comment)

        return cells

    def reassign(self):
        """called by :met:`tessellate`. Should not be called directly."""
        cells = self.cells

        #partition_kwargs = dict(cells.param.get('partition',{}))
        #partition_kwargs.update(self.partition_kwargs)
        partition_kwargs = self._partition_kwargs

        if self.time_window_kwargs:
            import copy
            tess = cells.tessellation.spatial_mesh
            point_index,cell_index = cells.cell_index
            cell_index = cell_index % tess.number_of_cells
            def noway(*args, **kwargs):
                raise RuntimeError('cell overlap is not supported in combination with point reassignment')
            cell_index = format_cell_index((point_index,cell_index),
                    format='array', select=noway, shape=(len(cells.points),))
            cells = Partition(cells.points, tess, cell_index)

        update_centroids = self.reassignment_kwargs.get('update_centroids', False)

        reassignment_count_threshold = self.reassignment_kwargs['count_threshold']
        reassignment_priority = self.reassignment_kwargs.get('priority_by', 'count')

        label = True
        while True:
            if update_centroids:
                if cells._cell_index is None:
                    warn('updating the centroids after reassigning points may fail because of memory error', RuntimeWarning)
                prev_cell_indices = cells.cell_index
                if not isinstance(prev_cell_indices, np.ndarray):
                    raise RuntimeError('cell overlap is not supported in combination with point reassignment')

            cells, deleted_cells, label = delete_low_count_cells(cells,
                    reassignment_count_threshold, reassignment_priority, label, self._partition_kwargs)

            if deleted_cells.size == 0:
                break

            dim = cells.tessellation.cell_centers.shape[1]
            min_ncells = dim + 2
            if cells.number_of_cells < min_ncells:
                raise RuntimeError('too few remaining cells ({}<{})'.format(cells.number_of_cells, min_ncells))

            if update_centroids:
                cells = update_cell_centers(cells, update_centroids, partition_kwargs)
            if self.time_window_kwargs:
                self.cells.tessellation.spatial_mesh = tess = cells.tessellation
                self.cells.cell_index = point_index,cell_index = \
                        self.cells.tessellation.cell_index(cells.points, **partition_kwargs)
                cell_index = cell_index % tess.number_of_cells
                cells.cell_index = cell_index = \
                        format_cell_index((point_index,cell_index),
                                format='array', select=noway, shape=(len(cells.points),))

        if not self.time_window_kwargs:
            self.cells = cells
        return cells


def tessellate1(xyt_data, method='gwr', output_file=None, verbose=False, \
        scaling=False, time_scale=None, \
        knn=None, radius=None, distance=None, ref_distance=None, \
        rel_min_distance=None, rel_avg_distance=None, rel_max_distance=None, \
        min_location_count=None, avg_location_count=None, max_location_count=None, \
        rel_max_size=None, rel_max_volume=None, \
        time_window_duration=None, time_window_shift=None, time_window_options=None, \
        enable_time_regularization=False, time_knn=None, \
        label=None, output_label=None, comment=None, input_label=None, inplace=False, \
        overwrite=None, return_analyses=False, \
        load_options=None, tessellation_options=None, partition_options=None, save_options=None, \
        force=None, \
        **kwargs):
    """
    Tessellation from points series and partitioning.

    This helper routine is a high-level interface to the various tessellation techniques
    implemented in TRamWAy.

    In addition to `knn`, `radius`, *filter* and *metric*, arguments with prefix *strict_* in
    their name apply to the partitioning step only, while the others apply to the tessellation step.

    *rel_max_size* and *rel_max_volume* are notable exceptions in that they currently apply to
    the partitioning step whereas they should conceptually apply to the tessellation step instead.
    This may change in a future version.

    Reassignment options can be provided using the `reassignment_options` dictionnary or keyworded
    arguments which names begin with '*reassign_*' or '*reassignment_*'.

    Arguments:
        xyt_data (str or pandas.DataFrame):
            Path to a *.trxyt* or *.rwa* file or raw data in the shape of
            :class:`pandas.DataFrame`.


        method (str):
            Tessellation method or plugin name.
            See for example
            :class:`~tramway.tessellation.random.RandomMesh` ('*random*'),
            :class:`~tramway.tessellation.grid.RegularMesh` ('*grid*'),
            :class:`~tramway.tessellation.hexagon.HexagonalMesh` ('*hexagon*'),
            :class:`~tramway.tessellation.kdtree.KDTreeMesh` ('*kdtree*'),
            :class:`~tramway.tessellation.kmeans.KMeansMesh` ('*kmeans*') and
            :class:`~tramway.tessellation.gwr.GasMesh` ('*gas*' or '*gwr*').

        output_file (str):
            Path to a *.rwa* file. The resulting tessellation and data partition will be
            stored in this file. If `xyt_data` is a path to a file and `output_file` is not
            defined, then `output_file` will be adapted from `xyt_data` with extension
            *.rwa* and possibly overwrite the input file.

        verbose (bool or int): Verbose output.

        scaling (bool or str):
            Normalization of the data.
            Any of '*unitrange*', '*whiten*' or other methods defined in
            :mod:`tramway.core.scaler`.

        time_scale (bool or float):
            If this argument is defined and intepretable as ``True``, the time axis is
            scaled by this factor and used as a space variable for the tessellation (2D+T or
            3D+T, for example).
            This is equivalent to manually scaling the ``t`` column and passing
            ``scaling=True``.

        knn (int or pair of ints):
            After growing the tessellation, a minimum and maximum numbers of nearest
            neighbours of each cell center can be used instead of the entire cell
            population. Let us denote ``min_nn, max_nn = knn``. Any of ``min_nn`` and
            ``max_nn`` can be ``None``.
            If a single `int` is supplied instead of a pair, then `knn` becomes ``min_nn``.
            ``min_nn`` enables cell overlap and any point may be associated with several
            cells.
            See also :meth:`~tramway.tessellation.base.Delaunay.cell_index`.

        radius (float):
            After growing the tessellation as a set of centroids, a cell will consist of
            the locations within this distance from the centroid.
            See also :meth:`~tramway.tessellation.base.Delaunay.cell_index`.

        distance/ref_distance (float):
            Supposed to be the average translocation distance. Can be modified so that the
            cells are smaller or larger.

        rel_min_distance (float):
            Multiplies with `ref_distance` to define the minimum inter-cell distance.

        rel_avg_distance (float):
            Multiplies with `ref_distance` to define an upper on the average inter-cell
            distance.

        rel_max_distance (float):
            Multiplies with `ref_distance` to define the maximum inter-cell distance.

        min_location_count (int):
            Minimum number of points per cell. Depending on the method, can be strictly
            enforced or regarded as a recommendation.

        avg_location_count (int):
            Average number of points per cell. For non-plugin method, per default, it is
            set to four times `min_location_count`.

        max_location_count (int):
            Maximum number of points per cell. This is used by *kdtree* and *gwr*.

        rel_max_size (float):
            Maximum cell diameter as a number of `ref_distance`. Diameter (or size) is
            estimated as twice the distance between the center of cell and the nearest
            vertex. Cells of excess size are ignored so as the associated locations.

        rel_max_volume (float):
            Maximum cell volume (or surface area in 2D) as a number of `ref_distance`.
            Cells of excess volume are ignored so as the associated locations.

        strict_min_location_count/min_n (int):
            Minimum number of points per cell in the eventual partition. Cells with
            insufficient points are ignored so as the associated locations.

        strict_rel_max_size (float):
            Maximum cell diameter as a number of `ref_distance`. Diameter (or size) is
            estimated as the maximum distance between any pair of locations in the cell.
            Cells of excess size are ignored so as the associated locations.

        time_window_duration (float):
            Window duration in seconds (or frames with
            ``time_window_options=dict(frames=True)``).
            This time windowing combines with any other spatial tessellation method.
            To use the :mod:`~tramway.tessellation.window` plugin only, use ``method=window``
            and its *duration* and *shift* arguments instead.
            See also the :mod:`~tramway.tessellation.window` plugin.

        time_window_shift (float):
            Window shift in seconds (or frames with
            ``time_window_options=dict(frames=True)``).
            Default is no overlap, i.e. ``time_window_shift=time_window_duration``.
            See also the :mod:`~tramway.tessellation.window` plugin.

        time_window_options (dict):
            Extra arguments for time windowing.
            See also the :mod:`~tramway.tessellation.window` plugin.

        enable_time_regularization (bool):
            Equivalent to ``time_window_options['time_dimension'] = enable_time_regularization``.

        time_knn (int or pair of ints):
            ``min_nn`` minimum number of nearest "neighbours" of the time segment center,
            or ``(min_nn, max_nn)`` minimum and maximum numbers of nearest neighbours in time.
            See also :meth:`~tramway.tessellation.time.TimeLattice.cell_index`.

        input_label (str):
            Label for the input tessellation for nesting tessellations.

        label/output_label (int or str):
            Label for the resulting analysis instance.

        inplace (bool):
            If True, `label`/`output_label`/`input_label` are exclusive, they all define
            a same analysis and the resulting analysis replaces the input analysis.

        comment (str):
            Description message for the resulting analysis.

        overwrite (bool): if an implicit output file already exists, overwrite it.

        return_analyses (bool):
            Return a :class:`~tramway.core.analyses.base.Analyses` object instead of
            the default :class:`~tramway.tessellation.base.Partition` output.

        load_options (dict):
            Pass extra keyword arguments to :func:`~tramway.core.xyt.load_xyt` if called.

        tessellation_options (dict):
            Pass explicit keyword arguments to the *__init__* function of the
            tessellation class as well as to the
            :meth:`~tramway.tessellation.base.Tessellation.tessellate` method, and ignore
            the extra input arguments.

        partition_options (dict):
            Pass explicit keyword arguments to the
            :meth:`~tramway.tessellation.base.Tessellation.cell_index` method and ignore
            the extra input arguments.

        reassignment_options (dict):
            allowed keys are:
            *count_threshold* (int): minimum number of points for a cell not to be deleted;
            *update_centroids* (bool or int): if evaluates to ``True``, cell centers are
            updated as the centers of mass for the assigned points, and then the new cells
            are checked again for *count_threshold* and so on; the number of iterations is
            not limited by default, and can be set passing an integer value instead of ``True``.

        save_options (dict):
            Pass extra keyword arguments to :func:`~tramway.core.xyt.save_rwa` if called.

    Returns:
        tramway.tessellation.base.Partition: A partition of the data with its
            :attr:`~tramway.tessellation.base.Partition.tessellation` attribute set.


    Apart from the parameters defined above, extra input arguments are admitted and may be passed
    to the initializer of the selected tessellation method as well as to the
    :meth:`~tramway.tessellation.base.Tessellation.tessellate` and
    :meth:`~tramway.tessellation.base.Tessellation.cell_index` methods.

    See the individual documentation of these methods for more information.

    """
    helper = Tessellate()
    helper.verbose = verbose
    helper.labels(label=label, input_label=input_label, output_label=output_label, inplace=inplace)
    if load_options is None:
        load_options = {}
    if helper.are_multiple_files(xyt_data) and len(xyt_data) == 1:
        xyt_data = next(iter(xyt_data))
    helper.prepare_data(xyt_data, scaling=scaling, time_scale=time_scale,
            metadata=not kwargs.pop('disable_metadata',None), **load_options)
    helper.plugin(method)

    # distinguish between tessellation and partition arguments
    if tessellation_options is None and partition_options is None:
        helper.tessellation_kwargs = dict(kwargs)
        helper.partition_kwargs = {}
        for _kw in kwargs:
            if _kw.startswith('strict_'):
                helper.partition_kwargs[_kw[7:]] = helper.tessellation_kwargs.pop(_kw)
        for _kw in ('filter', 'filter_descriptors_only', 'metric'):
            try:
                _arg = helper.tessellation_kwargs.pop(_kw)
            except KeyError:
                pass
            else:
                helper.partition_kwargs[_kw] = _arg
        for _kw in ('rel_max_size', 'rel_max_volume'):
            try:
                del helper.tessellation_kwargs[_kw]
            except KeyError:
                pass
    elif tessellation_options is None:
        helper.tessellation_kwargs = kwargs
        helper.partition_kwargs = partition_options
    else:
        helper.tessellation_kwargs = tessellation_options
        helper.partition_kwargs = kwargs

    params = helper.standard_parameters(distance=distance, ref_distance=ref_distance, \
        rel_min_distance=rel_min_distance, rel_avg_distance=rel_avg_distance, \
        rel_max_distance=rel_max_distance, \
        min_location_count=min_location_count, avg_location_count=avg_location_count, \
        max_location_count=max_location_count, knn=knn, radius=radius, time_knn=time_knn)
    helper.parse_args(params, knn=knn, radius=radius, time_knn=time_knn, \
        rel_max_size=rel_max_size, rel_max_volume=rel_max_volume, \
        time_window_duration=time_window_duration, time_window_shift=time_window_shift, \
        time_window_options=time_window_options, enable_time_regularization=enable_time_regularization, \
        kwargs=kwargs)

    cells = helper.tessellate(comment=comment)
    cells.param.update(kwargs)

    if overwrite is None and force is not None:
        warn('`force` is deprecated; please use `overwrite` instead', PendingDeprecationWarning)
        overwrite = force
    if save_options is None:
        save_options = {}
    helper.save_analyses(output_file, force=overwrite, **save_options)

    if return_analyses:
        return helper.analyses
    else:
        return cells



fig_formats = ['png', 'pdf', 'ps', 'eps', 'svg', 'html']

def tessellate0(xyt_data, method='gwr', output_file=None, verbose=False, \
    scaling=False, time_scale=None, \
    knn=None, radius=None, distance=None, ref_distance=None, \
    rel_min_distance=None, rel_avg_distance=None, rel_max_distance=None, \
    min_location_count=None, avg_location_count=None, max_location_count=None, \
    rel_max_size=None, rel_max_volume=None, \
    time_window_duration=None, time_window_shift=None, time_window_options=None, \
    label=None, output_label=None, comment=None, input_label=None, inplace=False, \
    force=None, return_analyses=False, \
    load_options=None, tessellation_options=None, partition_options=None, save_options=None, \
    **kwargs):
    """
    Tessellation from points series and partitioning.

    This helper routine is a high-level interface to the various tessellation techniques
    implemented in TRamWAy.

    In addition to `knn`, `radius`, *filter* and *metric*, arguments with prefix *strict_* in
    their name apply to the partitioning step only, while the others apply to the tessellation step.

    *rel_max_size* and *rel_max_volume* are notable exceptions in that they currently apply to
    the partitioning step whereas they should conceptually apply to the tessellation step instead.
    This may change in a future version.

    Arguments:
        xyt_data (str or pandas.DataFrame):
            Path to a *.trxyt* or *.rwa* file or raw data in the shape of
            :class:`pandas.DataFrame`.


        method (str):
            Tessellation method or plugin name.
            See for example
            :class:`~tramway.tessellation.random.RandomMesh` ('*random*'),
            :class:`~tramway.tessellation.grid.RegularMesh` ('*grid*'),
            :class:`~tramway.tessellation.hexagon.HexagonalMesh` ('*hexagon*'),
            :class:`~tramway.tessellation.kdtree.KDTreeMesh` ('*kdtree*'),
            :class:`~tramway.tessellation.kmeans.KMeansMesh` ('*kmeans*') and
            :class:`~tramway.tessellation.gwr.GasMesh` ('*gas*' or '*gwr*').

        output_file (str):
            Path to a *.rwa* file. The resulting tessellation and data partition will be
            stored in this file. If `xyt_data` is a path to a file and `output_file` is not
            defined, then `output_file` will be adapted from `xyt_data` with extension
            *.rwa* and possibly overwrite the input file.

        verbose (bool or int): Verbose output.

        scaling (bool or str):
            Normalization of the data.
            Any of '*unitrange*', '*whiten*' or other methods defined in
            :mod:`tramway.core.scaler`.

        time_scale (bool or float):
            If this argument is defined and intepretable as ``True``, the time axis is
            scaled by this factor and used as a space variable for the tessellation (2D+T or
            3D+T, for example).
            This is equivalent to manually scaling the ``t`` column and passing
            ``scaling=True``.

        knn (int or pair of ints):
            After growing the tessellation, a minimum and maximum numbers of nearest
            neighbours of each cell center can be used instead of the entire cell
            population. Let us denote ``min_nn, max_nn = knn``. Any of ``min_nn`` and
            ``max_nn`` can be ``None``.
            If a single `int` is supplied instead of a pair, then `knn` becomes ``min_nn``.
            ``min_nn`` enables cell overlap and any point may be associated with several
            cells.
            See also :meth:`~tramway.tessellation.base.Delaunay.cell_index`.

        radius (float):
            After growing the tessellation as a set of centroids, a cell will consist of
            the locations within this distance from the centroid.
            See also :meth:`~tramway.tessellation.base.Delaunay.cell_index`.

        distance/ref_distance (float):
            Supposed to be the average translocation distance. Can be modified so that the
            cells are smaller or larger.

        rel_min_distance (float):
            Multiplies with `ref_distance` to define the minimum inter-cell distance.

        rel_avg_distance (float):
            Multiplies with `ref_distance` to define an upper on the average inter-cell
            distance.

        rel_max_distance (float):
            Multiplies with `ref_distance` to define the maximum inter-cell distance.

        min_location_count (int):
            Minimum number of points per cell. Depending on the method, can be strictly
            enforced or regarded as a recommendation.

        avg_location_count (int):
            Average number of points per cell. For non-plugin method, per default, it is
            set to four times `min_location_count`.

        max_location_count (int):
            Maximum number of points per cell. This is used by *kdtree* and *gwr*.

        rel_max_size (float):
            Maximum cell diameter as a number of `ref_distance`. Diameter (or size) is
            estimated as twice the distance between the center of cell and the nearest
            vertex. Cells of excess size are ignored so as the associated locations.

        rel_max_volume (float):
            Maximum cell volume (or surface area in 2D) as a number of `ref_distance`.
            Cells of excess volume are ignored so as the associated locations.

        strict_min_location_count (int):
            Minimum number of points per cell in the eventual partition. Cells with
            insufficient points are ignored so as the associated locations.

        strict_rel_max_size (float):
            Maximum cell diameter as a number of `ref_distance`. Diameter (or size) is
            estimated as the maximum distance between any pair of locations in the cell.
            Cells of excess size are ignored so as the associated locations.

        time_window_duration (float):
            Window duration in seconds (or frames with
            ``time_window_options=dict(frames=True)``).
            This time windowing combines with any other spatial tessellation method.
            To use the :mod:`~tramway.tessellation.window` plugin only, use ``method=window``
            and its *duration* and *shift* arguments instead.
            See also the :mod:`~tramway.tessellation.window` plugin.

        time_window_shift (float):
            Window shift in seconds (or frames with
            ``time_window_options=dict(frames=True)``).
            Default is no overlap, i.e. ``time_window_shift=time_window_duration``.
            See also the :mod:`~tramway.tessellation.window` plugin.

        time_window_options (dict):
            Extra arguments for time windowing.
            See also the :mod:`~tramway.tessellation.window` plugin.

        input_label (str):
            Label for the input tessellation for nesting tessellations.

        label/output_label (int or str):
            Label for the resulting analysis instance.

        inplace (bool):
            If True, `label`/`output_label`/`input_label` are exclusive, they all define
            a same analysis and the resulting analysis replaces the input analysis.

        comment (str):
            Description message for the resulting analysis.

        return_analyses (bool):
            Return a :class:`~tramway.core.analyses.base.Analyses` object instead of
            the default :class:`~tramway.tessellation.base.Partition` output.

        load_options (dict):
            Pass extra keyword arguments to :func:`~tramway.core.xyt.load_xyt` if called.

        tessellation_options (dict):
            Pass explicit keyword arguments to the *__init__* function of the
            tessellation class as well as to the
            :meth:`~tramway.tessellation.base.Tessellation.tessellate` method, and ignore
            the extra input arguments.

        partition_options (dict):
            Pass explicit keyword arguments to the
            :meth:`~tramway.tessellation.base.Tessellation.cell_index` method and ignore
            the extra input arguments.

        save_options (dict):
            Pass extra keyword arguments to :func:`~tramway.core.xyt.save_rwa` if called.

    Returns:
        tramway.tessellation.base.Partition: A partition of the data with its
            :attr:`~tramway.tessellation.base.Partition.tessellation` attribute set.


    Apart from the parameters defined above, extra input arguments are admitted and may be passed
    to the initializer of the selected tessellation method as well as to the
    :meth:`~tramway.tessellation.base.Tessellation.tessellate` and
    :meth:`~tramway.tessellation.base.Tessellation.cell_index` methods.

    See the individual documentation of these methods for more information.

    """
    if verbose:
        plugins.verbose = True

    compress = True
    if inplace:
        labels_exclusive = ValueError("multiple different values in exclusive arguments 'label', 'input_label' and 'output_label'")
        if label:
            if (input_label and input_label != label) or \
                (output_label and output_label != label):
                raise labels_exclusive
        elif input_label:
            if output_label and output_label == input_label:
                raise labels_exclusive
            label = input_label
        elif output_label:
            label = output_label
        output_label = input_label = label
    elif label is None:
        label = output_label
    elif output_label and output_label != label:
        raise ValueError("'label' and 'output_label' are both defined and are different")

    no_nesting_error = ValueError('nesting tessellations does not apply to translocation data')
    multiple_files = lambda a: isinstance(a, (tuple, list, frozenset, set))
    xyt_files = []
    if isinstance(xyt_data, six.string_types):
        # file path
        xyt_files = [xyt_data]
    elif multiple_files(xyt_data):
        # file path(s)
        xyt_files = list(xyt_data)
        if not xyt_data[1:]:
            xyt_data = xyt_data[0]
    if xyt_files:
        if load_options is None:
            load_options = {}
        if multiple_files(xyt_data):
            xyt_file = xyt_data
        else:
            try:
                analyses = load_rwa(xyt_data, lazy=True, verbose=False)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                #traceback.print_exc()
                xyt_file = xyt_data
            else:
                xyt_file = False
                if input_label:
                    input_partition, = find_artefacts(analyses, Partition, input_label)
                xyt_data = analyses.data
        if xyt_file:
            xyt_data, xyt_files = load_xyt(xyt_files, return_paths=True, verbose=verbose,
                **load_options)
            analyses = Analyses(xyt_data)
            if input_label is not None:
                raise no_nesting_error
    else:
        if isinstance(xyt_data, abc.Analyses):
            analyses = xyt_data
            xyt_data = analyses.data
        else:
            analyses = Analyses(xyt_data)
        #warn('TODO: test direct data input', UseCaseWarning)
        if input_label is not None:
            raise no_nesting_error
    input_files = xyt_files

    try:
        setup, module = plugins[method]
        constructor = setup['make']
        if isinstance(constructor, str):
            constructor = getattr(module, setup['make'])
    except KeyError: # former code
        from tramway.tessellation.kdtree import KDTreeMesh
        from tramway.tessellation.kmeans import KMeansMesh
        from tramway.tessellation.gwr import GasMesh
        plugin = False
        methods = dict(grid=RegularMesh, kdtree=KDTreeMesh, kmeans=KMeansMesh, gwr=GasMesh)
        constructor = methods[method]
    else:
        plugin = True
    assert plugin

    transloc_length = min_distance = avg_distance = max_distance = None
    if ref_distance is None and distance is not None:
        # `distance` is only for compatibility with the tramway commandline
        ref_distance = distance
    if input_label and input_partition.param is not None:
        prm = input_partition.param.get('tessellation', input_partition.param)
        if not ref_distance:
            ref_distance = prm.get('ref_distance', None)
        min_distance = prm.get('min_distance', min_distance)
        avg_distance = prm.get('avg_distance', avg_distance)
        max_distance = prm.get('max_distance', max_distance)
    # former default values for `rel_min_distance` and `rel_avg_distance`
    if rel_min_distance is None and min_distance is None:
        rel_min_distance = .8
    if rel_avg_distance is None and avg_distance is None:
        rel_avg_distance = 2.

    if ref_distance is None:
        transloc_xy = np.asarray(translocations(xyt_data))
        if transloc_xy.shape[0] == 0:
            raise ValueError('no translocation found')
        transloc_length = np.nanmean(np.sqrt(np.sum(transloc_xy * transloc_xy, axis=1)))
        if verbose:
            print('average translocation distance: {}'.format(transloc_length))
        ref_distance = transloc_length
    if rel_min_distance is not None:
        min_distance = rel_min_distance * ref_distance
    if rel_avg_distance is not None:
        avg_distance = rel_avg_distance * ref_distance
    if rel_max_distance is not None:
        # applies only to KDTreeMesh and Kohonen
        max_distance = rel_max_distance * ref_distance
        #if method not in ['kdtree', 'gwr', 'kohonen']:
        #    warn('`rel_max_distance` is relevant only with `kdtree`', IgnoredInputWarning)

    if scaling:
        if scaling is True:
            scaling = 'whiten'
        scaler = dict(whiten=whiten, unit=unitrange)[scaling]()
    else:
        scaler = None

    if min_location_count is None: # former default value: 20
        if knn is None and radius is None:
            min_location_count = 20
    n_pts = float(xyt_data.shape[0])
    if min_location_count:
        min_probability = float(min_location_count) / n_pts
    else:
        min_probability = None
        if not plugin:
            warn('undefined `min_location_count`; not tested', UseCaseWarning)
    if not avg_location_count:
        if min_location_count is None:
            avg_location_count = 80 # former default value
        else:
            avg_location_count = 4 * min_location_count
    if avg_location_count:
        avg_probability = float(avg_location_count) / n_pts
    else:
        avg_probability = None
        if not plugin:
            warn('undefined `avg_location_count`; not tested', UseCaseWarning)
    if max_location_count:
        # applies only to KDTreeMesh
        max_probability = float(max_location_count) / n_pts
        if not plugin and method != 'kdtree':
            warn('`max_location_count` is relevant only with `kdtree`', IgnoredInputWarning)
    else:
        max_probability = None

    # actually useful only if no tessellation nesting
    colnames = ['x', 'y']
    if 'z' in xyt_data:
        colnames.append('z')
    if time_scale:
        colnames.append('t')
        if scaler is None:
            raise ValueError('no scaling defined')
        scaler.factor = [('t', time_scale)]

    # distinguish between tessellation and partition arguments
    if tessellation_options is None and partition_options is None:
        tessellation_kwargs = dict(kwargs)
        partition_kwargs = {}
        for _kw in kwargs:
            if _kw.startswith('strict_'):
                partition_kwargs[_kw[7:]] = tessellation_kwargs.pop(_kw)
        for _kw in ('filter', 'filter_descriptors_only', 'metric'):
            try:
                _arg = tessellation_kwargs.pop(_kw)
            except KeyError:
                pass
            else:
                partition_kwargs[_kw] = _arg
        for _kw in ('rel_max_size', 'rel_max_volume'):
            try:
                del tessellation_kwargs[_kw]
            except KeyError:
                pass
    elif tessellation_options is None:
        tessellation_kwargs = kwargs
        partition_kwargs = partition_options
    else:
        tessellation_kwargs = tessellation_options
        partition_kwargs = kwargs

    # initialize a Tessellation object
    params = dict( \
        min_distance=min_distance, \
        avg_distance=avg_distance, \
        max_distance=max_distance, \
        min_probability=min_probability, \
        avg_probability=avg_probability, \
        max_probability=max_probability, \
        )
    if plugin:
        for ignored in ['max_level']:
            try:
                if tessellation_kwargs[ignored] is None:
                    del tessellation_kwargs[ignored]
            except KeyError:
                pass
        params.update(dict( \
            min_location_count=min_location_count, \
            avg_location_count=avg_location_count, \
            max_location_count=max_location_count, \
            ))
        params.update(tessellation_kwargs)
        for key in setup.get('make_arguments', {}):
            try:
                param = params[key]
            except KeyError:
                pass
            else:
                tessellation_kwargs[key] = param
    else:
        params.update(tessellation_kwargs)
        tessellation_kwargs = params
    time_window_kwargs = {}

    if input_label:
        if time_window_duration:
            raise NotImplementedError('spatial tessellation combined with time windowing is not supported yet by tessellation nesting')
        tess = NestedTessellations(scaler, input_partition, factory=constructor,
            **tessellation_kwargs)
        xyt_data = data = input_partition.points
    else:
        if time_window_duration:
            time_window_kwargs['duration'] = time_window_duration
            if time_window_shift:
                time_window_kwargs['shift'] = time_window_shift
            if time_window_options:
                time_window_kwargs.update(time_window_options)
            import tramway.tessellation.window as window
            tess = window.SlidingWindow(**time_window_kwargs)
            tess.spatial_mesh = constructor(scaler, **tessellation_kwargs)
        else:
            tess = constructor(scaler, **tessellation_kwargs)
        data = xyt_data[colnames]

    # grow the tessellation
    tessellate_kwargs = tessellation_kwargs
    tess.tessellate(data, verbose=verbose, **tessellate_kwargs)

    # partition the dataset into the cells of the tessellation
    _filter_f = partition_kwargs.get('filter', None)
    try:
        max_size = kwargs['rel_max_size']
    except KeyError:
        _filter_fg = _filter_f
    else:
        max_size *= ref_distance
        def _filter_g(voronoi, cell, points):
            return voronoi.scaler.unscale_distance(np.sqrt(np.min(np.sum(( \
                    voronoi._center[cell] - \
                    voronoi._vertices[voronoi.cell_vertices[cell]] \
                )*2, axis=1)))) <= max_size
        _filter_fg = _filter_g if _filter_f is None \
            else lambda *a: _filter_g(*a) and _filter_f(*a)
    try:
        max_volume = kwargs['rel_max_volume']
    except KeyError:
        _filter_fgh = _filter_fg
    else:
        max_volume *= ref_distance
        def _filter_h(voronoi, cell, points):
            return voronoi.cell_volume[cell] <= max_volume
        _filter_fgh = _filter_h if _filter_fg is None \
            else lambda *a: _filter_h(*a) and _filter_fg(*a)
    try:
        max_actual_size = partition_kwargs.pop('rel_max_size')
    except KeyError:
        _filter_fghi = _filter_fgh
    else:
        max_actual_size *= ref_distance
        _descr = not partition_kwargs.get('filter_descriptors_only', True)
        def _filter_i(voronoi, cell, x):
            if _descr:
                x = voronoi.descriptors(x, asarray=True)
            x2 = np.sum(x * x, axis=1, keepdims=True)
            d2 = x2 + x2.T - 2. * np.dot(x, x.T)
            d = np.sqrt(np.max(d2.ravel()))
            return d <= max_actual_size
        _filter_fghi = _filter_i if _filter_fgh is None \
            else lambda *a: _filter_i(*a) and _filter_fgh(*a)
    if _filter_fghi is not None:
        partition_kwargs['filter'] = _filter_fghi
        if 'filter_descriptors_only' not in partition_kwargs:
            partition_kwargs['filter_descriptors_only'] = True

    if not (knn is None and radius is None):
        if knn is not None:
            partition_kwargs['knn'] = knn
        if radius is not None:
            if 'radius' in partition_kwargs:
                warn('overwriting `radius`', RuntimeWarning)
            partition_kwargs['radius'] = radius
        if 'min_location_count' not in partition_kwargs:
            partition_kwargs['min_location_count'] = min_location_count
        if 'metric' not in partition_kwargs:
            partition_kwargs['metric'] = 'euclidean'

    try:
        cell_index = tess.cell_index(xyt_data, **partition_kwargs)
    except MemoryError:
        if verbose:
            print(traceback.format_exc())
        warn('memory error: cannot assign points to cells', RuntimeWarning)
        cell_index = None

    stats = Partition(xyt_data, tess, cell_index)

    # store some parameters together with the partition
    stats.param['method'] = method
    #if transloc_length:
    #    stats.param['transloc_length'] = transloc_length
    #else:
    #    stats.param['ref_distance'] = ref_distance
    #for _arg in ('min_distance', 'avg_distance', 'max_distance', 'knn', 'radius', \
    #        'time_window_duration', 'time_window_shift', 'time_window_options'):
    #    _val = eval(_arg)
    #    if _val:
    #        stats.param[_arg] = _val
    if time_window_kwargs:
        stats.param['time_window'] = time_window_kwargs
    if tessellation_kwargs:
        stats.param['tessellation'] = tessellation_kwargs
    if partition_kwargs:
        stats.param['partition'] = partition_kwargs
    if reassignment_kwargs:
        stats.param['reassignment'] = reassignment_kwargs
    stats.param.update(kwargs)

    # insert the resulting analysis in the analysis tree
    if input_label:
        input_analysis = analyses
        if isinstance(input_label, (tuple, list)):
            prefix_labels = list(input_label)
            terminal_label = prefix_labels.pop()
            for _label in prefix_labels:
                input_analysis = input_analysis.instances[_label]
        else:
            terminal_label = input_label
        # `input_analysis` is already in `analyses`
        if inplace and comment:
            input_analysis.comments[terminal_label] = comment
        input_analysis = input_analysis.instances[terminal_label]
        if inplace:
            input_analysis.artefact = stats
        else:
            input_analysis.add(Analyses(stats), label=label, comment=comment)
    else:
        analyses.add(Analyses(stats), label=label, comment=comment)

    # save the analysis tree (`analyses`)
    if output_file or xyt_files:
        if output_file is None:
            output_file = os.path.splitext(xyt_files[0])[0] + '.rwa'
        if save_options is None:
            save_options = {}
        if 'force' not in save_options:
            save_options['force'] = force or (force is not False and len(input_files)==1 and input_files[0]==output_file)

        save_rwa(output_file, analyses, verbose, **save_options)

    if return_analyses:
        return analyses
    else:
        return stats




def cell_plot(cells, xy_layer=None, output_file=None, fig_format=None, \
    show=None, verbose=False, figsize=None, dpi=None, \
    location_count_hist=False, cell_dist_hist=False, location_dist_hist=False, \
    aspect=None, delaunay=None, locations={}, voronoi=None, colors=None, title=None, \
    cell_indices=None, segment=None, label=None, input_label=None, num = None, \
    **kwargs):
    """
    Partition plots.

    Plots a spatial representation of the tessellation and partition if data are 2D.

    Arguments:
        cells (str or Partition or tramway.core.analyses.base.Analyses):
            Path to a *.rwa* file or :class:`~tramway.tessellation.Partition`
            instance or analysis tree; files and analysis trees may require
            `label`/`input_label` to be defined.

        xy_layer ({None, 'delaunay', 'voronoi'}):
            Overlay Delaunay or Voronoi graph over the data points. For 2D data only.
            **Deprecated**! Please use `delaunay` and `voronoi` arguments instead.

        output_file (str):
            Path to a file in which the figure will be saved. If `cells` is a path and
            `fig_format` is defined, `output_file` is automatically set.

        fig_format (str):
            Any image format supported by :func:`matplotlib.pyplot.savefig`.

        show (bool or str):
            Makes `cell_plot` show the figure(s) which is the default behavior if and only
            if the figures are not saved.
            If ``show='draw'``, :func:`~matplotlib.pyplot.draw` is called instead of
            :func:`~matplotlib.pyplot.show`.
            To maintain the current default behaviour in the future, set `show` to
            ``True`` from now on.

        verbose (bool): Verbose output.

        figsize (pair of floats):
            Passed to :func:`matplotlib.pyplot.figure`. Applies only to the spatial
            representation figure.

        dpi (int):
            Passed to :func:`matplotlib.pyplot.savefig`. Applies only to the spatial
            representation figure.

        location_count_hist (bool):
            Plot a histogram of point counts (per cell). If the figure is saved, the
            corresponding file will have sub-extension *.hpc*;
            **deprecated**.

        cell_dist_hist (bool):
            Plot a histogram of distances between neighbour centroids. If the figure is
            saved, the corresponding file will have sub-extension *.hcd*;
            **deprecated**.

        location_dist_hist (bool):
            Plot a histogram of distances between points from neighbour cells. If the figure
            is saved, the corresponding file will have sub-extension *.hpd*;
            **deprecated**.

        aspect (str):
            Aspect ratio. Can be '*equal*'.

        locations (dict):
            Keyword arguments to :func:`~tramway.plot.mesh.plot_points`.

        delaunay (bool or dict):
            Overlay Delaunay graph. If :class:`dict`, keyword arguments to
            :func:`~tramway.plot.mesh.plot_delaunay`.

        voronoi (bool or dict):
            Overlay Voronoi graph (default: ``True``). If :class:`dict`, keyword arguments
            to :func:`~tramway.plot.mesh.plot_voronoi`.

        cell_indices (bool or dict):
            ``True`` or keyworded arguments; plot cell indices instead of centroids.

        label/input_label (int or str or list):
            If `cells` is a filepath or an analysis tree, label of the analysis instance.

        segment (int):
            Segment index; if multiple time segments were defined, show only one segment.

        num (int):
            Figure number. Default: None (new figure for each call).

    Notes:
        See also :mod:`tramway.plot.mesh`.

    """
    input_file = ''
    if not isinstance(cells, Partition):
        if label is None:
            labels = input_label
        else:
            labels, label = label, None
        if not valid_label(labels):
            labels = ()
        elif not isinstance(labels, (tuple, list)):
            labels = (labels, )
        if isinstance(cells, abc.Analyses):
            analyses, cells = cells, None
        else:
            input_file, cells = cells, None
            if isinstance(input_file, (tuple, list)):
                if input_file[1:]:
                    warn('can only process a single file', RuntimeWarning)
                input_file = input_file[0]
            if not isinstance(input_file, str):
                raise TypeError('unsupported input data type: {}'.format(type(input_file)))
            try:
                analyses = load_rwa(input_file, lazy=True)
                #if labels:
                #       analyses = extract_analysis(analyses, labels)
            except KeyError as e:
                if e.args and 'analyses' not in e.args[0]:
                    raise
                # legacy code
                imt_path = input_file
                if imt_path is None:
                    raise ValueError('undefined input file')
                # copy-paste
                if os.path.isdir(imt_path):
                    imt_path = os.listdir(imt_path)
                    files, exts = zip(*os.path.splitext(imt_path))
                    for ext in imt_extensions:
                        if ext in exts:
                            imt_path = imt_path[exts.index(ext)]
                            break
                    if isinstance(imt_path, list):
                        imt_path = imt_path[0]
                    auto_select = True
                elif os.path.isfile(imt_path):
                    auto_select = False
                else:
                    candidates = [ imt_path + ext for ext in imt_extensions ]
                    candidates = [ f for f in candidates if os.path.isfile(f) ]
                    if candidates:
                        imt_path = candidates[0]
                    else:
                        raise IOError('no tessellation file found in {}'.format(imt_path))
                    auto_select = True
                if auto_select and verbose:
                    print('selecting {} as a tessellation file'.format(imt_path))

                # load the data
                input_file = imt_path
                try:
                    from rwa import HDF5Store
                    hdf = HDF5Store(input_file, 'r')
                    hdf.lazy = False
                    cells = hdf.peek('cells')
                    if cells.tessellation is None:
                        cells._tessellation = hdf.peek('_tesselation', hdf.store['cells'])
                except:
                    print(traceback.format_exc())
                    warn('HDF5 libraries may not be installed', ImportWarning)
                finally:
                    hdf.close()
        if cells is None:
            if isinstance(analyses, dict):
                if not analyses:
                    raise ValueError('not any file matches')
                elif len(analyses) == 1:
                    analyses = list(analyses.values())[0]
                else:
                    raise ValueError('multiple files match')
            if not labels:
                labels = list(analyses.labels)
            if labels[1:]:
                raise ValueError('multiple instances; label is required')
            label = labels[-1]
            cells = analyses[label].data

    # identify time segments if any
    try:
        import tramway.tessellation.time as time
        with_segments = isinstance(cells.tessellation, time.TimeLattice) \
                and cells.tessellation.spatial_mesh is not None
    except ImportError:
        with_segments = False
    if with_segments:
        if segment is None:
            xyt = cells.points
            mesh = cells.tessellation.spatial_mesh
            prms = cells.param.get('partition', {})
            prms.pop('exclude_cells_by_location_count', None)
            time_col = prms.pop('time_col', 't')
            try:
                prms.pop('time_knn')
            except KeyError:
                pass
            else:
                warn('ignoring `time_knn`', RuntimeWarning)
            bb = cells.bounding_box
            cells = Partition(xyt, mesh, mesh.cell_index(xyt, **prms))
            cells.bounding_box = bb
        else:
            if isinstance(segment, (tuple, list)):
                if segment[1:]:
                    warn('cannot plot multiple segments in a single `cell_plot` call', RuntimeWarning)
                segment = segment.pop()
                print('plotting segment {}'.format(segment))
            cells = cells.tessellation.split_segments(cells)[segment]
    elif segment is not None:
        warn('cannot find time segments', RuntimeWarning)
        segment = None

    # guess back some input parameters (for backward compatible titles)
    complementary_plots = location_count_hist or cell_dist_hist or location_dist_hist
    if (title and not isinstance(title, str)) or complementary_plots:
        method_name = {}
        try:
            method_name[RegularMesh] = ('grid', 'grid', 'regular grid')
        except NameError:
            pass
        try:
            import tramway.tessellation.kdtree
        except ImportError:
            pass
        else:
            method_name[tramway.tessellation.kdtree.KDTreeMesh] = \
                ('kdtree', 'k-d tree', 'k-d tree based tessellation')
        try:
            import tramway.tessellation.kmeans
        except ImportError:
            pass
        else:
            method_name[tramway.tessellation.kmeans.KMeansMesh] = \
                ('kmeans', 'k-means', 'k-means based tessellation')
        try:
            import tramway.tessellation.gwr
        except ImportError:
            pass
        else:
            method_name[tramway.tessellation.gwr.GasMesh] = ('gwr', 'GWR', 'GWR based tessellation')
        try:
            method_name, pp_method_name, method_title = method_name[type(cells.tessellation)]
        except (KeyError, AttributeError):
            method_name = pp_method_name = method_title = ''

    if complementary_plots:
        warn('complementary plots will be removed in a future release', DeprecationWarning)
        try:
            min_distance = kwargs['min_distance']
        except KeyError:
            min_distance = cells.param['tessellation'].get('min_distance', 0)
        try:
            avg_distance = kwargs['avg_distance']
        except KeyError:
            avg_distance = cells.param['tessellation'].get('avg_distance', None)
    try:
        min_location_count = kwargs['min_location_count']
    except KeyError:
        try:
            min_location_count = cells.param['partition']['min_location_count']
        except (KeyError, AttributeError):
            min_location_count = 0

    print_figs = output_file or (input_file and fig_format)
    if (print_figs and figsize is None) or figsize is True:
        figsize = (12., 9.)

    # import graphics libraries with adequate backend
    if print_figs:
        import matplotlib
        try:
            matplotlib.use('Agg') # head-less rendering (no X server required)
        except:
            pass
    import matplotlib.pyplot as mplt
    import tramway.plot.mesh as tplt

    # plot the data points together with the tessellation
    figs = []
    try:
        dim = cells.tessellation.cell_centers.shape[1]
    except AttributeError as e:
        try:
            dim = cells.dim
        except AttributeError:
            raise e
    fig = None
    if dim == 2 and not complementary_plots:
        if 'figure' in kwargs:
            fig = kwargs['figure']
        elif figs or figsize is not None or num is not None:
            fig = mplt.figure(figsize=figsize, dpi=dpi, num = num)
        else:
            fig = mplt.gcf()
        figs.append(fig)
        if locations is not None:
            if 'axes' in kwargs:
                locations['axes'] = kwargs['axes']
            if 'knn' in cells.param: # if knn <= min_count, min_count is actually ignored
                tplt.plot_points(cells, **locations)
            else:
                tplt.plot_points(cells, min_count=min_location_count, **locations)
        if aspect is not None:
            ax = kwargs.get('axes', None)
            if ax is None:
                ax = fig.gca()
            ax.set_aspect(aspect)
        if voronoi is None:
            voronoi = issubclass(type(cells.tessellation), Voronoi)
        if xy_layer != 'delaunay' and voronoi:
            if not isinstance(voronoi, dict):
                voronoi = {}
            if cell_indices and 'centroid_style' not in voronoi:
                voronoi['centroid_style'] = None
            if 'axes' in kwargs:
                voronoi['axes'] = kwargs['axes']
            tplt.plot_voronoi(cells, **voronoi)
            voronoi = True
        if xy_layer == 'delaunay' or delaunay: # Delaunay above Voronoi
            if not isinstance(delaunay, dict):
                delaunay = {}
            if 'axes' in kwargs:
                delaunay['axes'] = kwargs['axes']
            tplt.plot_delaunay(cells, **delaunay)
            delaunay = True
        if cell_indices:
            if not isinstance(cell_indices, dict):
                cell_indices = {}
            if 'axes' in kwargs:
                cell_indices['axes'] = kwargs['axes']
            tplt.plot_indices(cells, **cell_indices)
        if title:
            if isinstance(title, str):
                _title = title
            elif delaunay == voronoi:
                _title = pp_method_name
            elif delaunay:
                _title = pp_method_name + ' based Delaunay'
            elif voronoi:
                _title = pp_method_name + ' based Voronoi'
            try:
                axes = kwargs['axes']
            except KeyError:
                mplt.title(_title)
            else:
                axes.set_title(_title)


    if print_figs:
        if output_file:
            filename, figext = os.path.splitext(output_file)
            if fig_format:
                figext = fig_format
            elif figext and figext[1:] in fig_formats:
                figext = figext[1:]
            else:
                figext = fig_formats[0]
        else:
            figext = fig_format
            filename, _ = os.path.splitext(input_file)
        subname, subext = os.path.splitext(filename)
        if subext and subext[1:] in ['imt']: # very old file format
            filename = subname
        if fig is not None:
            vor_file = '{}.{}'.format(filename, figext)
            if verbose:
                print('writing file: {}'.format(vor_file))
            fig.savefig(vor_file, dpi=dpi)


    # the complementary histograms below haven't been tested for a while [TODO]

    if location_count_hist:
        # plot a histogram of the number of points per cell
        fig = mplt.figure()
        figs.append(fig)
        mplt.hist(cells.location_count, bins=np.arange(0, min_location_count*20, min_location_count))
        mplt.plot((min_location_count, min_location_count), mplt.ylim(), 'r-')
        mplt.title(method_title)
        mplt.xlabel('point count (per cell)')
        if print_figs:
            hpc_file = '{}.{}.{}'.format(filename, 'hpc', figext)
            if verbose:
                print('writing file: {}'.format(hpc_file))
            fig.savefig(hpc_file)

    if cell_dist_hist:
        # plot a histogram of the distance between adjacent cell centers
        A = sparse.triu(cells.tessellation.cell_adjacency, format='coo')
        i, j, k = A.row, A.col, A.data
        label = cells.tessellation.adjacency_label
        if label is not None:
            i = i[0 < label[k]]
            j = j[0 < label[k]]
        pts = cells.tessellation.cell_centers
        dist = pts[i,:] - pts[j,:]
        dist = np.sqrt(np.sum(dist * dist, axis=1))
        fig = mplt.figure()
        figs.append(fig)
        mplt.hist(np.log(dist), bins=50)
        if avg_distance:
            dmin = np.log(min_distance)
            dmax = np.log(avg_distance)
            ylim = mplt.ylim()
            mplt.plot((dmin, dmin), ylim, 'r-')
            mplt.plot((dmax, dmax), ylim, 'r-')
        mplt.title(method_title)
        mplt.xlabel('inter-centroid distance (log)')
        if print_figs:
            hcd_file = '{}.{}.{}'.format(filename, 'hcd', figext)
            if verbose:
                print('writing file: {}'.format(hcd_file))
            fig.savefig(hcd_file)

    if location_dist_hist:
        adj = point_adjacency_matrix(cells, symetric=False)
        dist = adj.data
        fig = mplt.figure()
        figs.append(fig)
        mplt.hist(np.log(dist), bins=100)
        if avg_distance:
            dmin = np.log(min_distance)
            dmax = np.log(avg_distance)
            ylim = mplt.ylim()
            mplt.plot((dmin, dmin), ylim, 'r-')
            mplt.plot((dmax, dmax), ylim, 'r-')
        mplt.title(method_title)
        mplt.xlabel('inter-point distance (log)')
        if print_figs:
            hpd_file = '{}.{}.{}'.format(filename, 'hpd', figext)
            if verbose:
                print('writing file: {}'.format(hpd_file))
            fig.savefig(hpd_file)

    if show or not print_figs: # 'or' will become 'and'
        if show == 'draw':
            mplt.draw()
        elif show is not False:
            mplt.show()
    else:
        for fig in figs:
            mplt.close(fig)


def delete_low_count_cells(partition, count_threshold, priority_by=None, label=True, partition_kwargs={}):
    tessellation = partition.tessellation
    deleted_cells, = np.nonzero(partition.location_count<count_threshold)
    #print('ncells', partition.number_of_cells, 'npts_min', np.min(partition.location_count), 'npts_max', np.max(partition.location_count), 'ncells_deleted', deleted_cells.size)
    if deleted_cells.size == 0:
        return partition, deleted_cells, label
    if priority_by:
        if priority_by == 'count':
            priority = -partition.location_count[deleted_cells]
        elif priority_by == 'volume':
            priority = tessellation.volume[deleted_cells]
        ordering = np.argsort(priority)
        deleted_cells = deleted_cells[ordering]
        index_mapping, label = tessellation.delete_cells(deleted_cells, exclude_neighbours=True, adjacency_label=label)
    else:
        index_mapping, label = tessellation.delete_cells(deleted_cells)

    points = partition.points
    cell_indices = tessellation.cell_index(points, **partition_kwargs)
    new_partition = Partition(points, tessellation, cell_indices)
    return new_partition, deleted_cells, label


def update_cell_centers(cells, max_iter, partition_kwargs={}):
    points = cells.points[['x','y']].values # TODO: use `descriptors` instead
    tess = cells.tessellation
    cell_indices = cells.cell_index

    if max_iter is True:
        max_iter = np.inf

    k = 0
    while k < max_iter:
        prev_cell_indices = cell_indices
        cell_centers = np.array(tess.cell_centers) # copy
        any_new = False
        for i in range(tess.number_of_cells):
            prev_cell_i = prev_cell_indices==i
            cell_i = cell_indices==i
            if np.any(cell_i) and not np.all(cell_i==prev_cell_i):
                any_new = True
                center = np.mean(points[cell_i], axis=0)
                assert not np.any(np.isnan(center))
                cell_centers[i] = center
        if any_new:
            tess.cell_centers = cell_centers
            cell_indices = tess.cell_index(cells.points, **partition_kwargs)
            k += 1
        else:
            break
    return Partition(cells.points, tess, cell_indices)


tessellate = tessellate1
