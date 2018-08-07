# -*- coding: utf-8 -*-

# Copyright © 2017 2018, Institut Pasteur
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
from ..tessellation import *
from warnings import warn
import six
import traceback
# no module-wide matplotlib import for head-less usage of `tessellate`
# in the case matplotlib's backend is interactive


fig_formats = ['png', 'pdf', 'ps', 'eps', 'svg']


class UseCaseWarning(UserWarning):
        pass
class IgnoredInputWarning(UserWarning):
        pass


def tessellate(xyt_data, method='gwr', output_file=None, verbose=False, \
        scaling=False, time_scale=None, \
        knn=None, radius=None, distance=None, ref_distance=None, \
        rel_min_distance=None, rel_avg_distance=None, rel_max_distance=None, \
        min_location_count=None, avg_location_count=None, max_location_count=None, \
        rel_max_size=None, rel_max_volume=None, \
        label=None, output_label=None, comment=None, input_label=None, inplace=False, \
        force=None, return_analyses=False, tessellation_options=None, partition_options=None, \
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
                xyt_data (str or matrix):
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
                        the default :class:`~tramway.tessellation.base.CellStats` output.

                tessellation_options (dict):
                        Pass explicit keyword arguments to the *__init__* function of the
                        tessellation class as well as to the
                        :meth:`~tramway.tessellation.base.Tessellation.tessellate` method, and ignore
                        the extra input arguments.

                partition_options (dict):
                        Pass explicit keyword arguments to the
                        :meth:`~tramway.tessellation.base.Tessellation.cell_index` method and ignore
                        the extra input arguments.

        Returns:
                tramway.tessellation.base.CellStats: A partition of the data with its
                        :attr:`~tramway.tessellation.base.CellStats.tessellation` attribute set.


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
                if multiple_files(xyt_data):
                        xyt_file = xyt_data
                else:
                        try:
                                analyses = load_rwa(xyt_data)
                        except (KeyboardInterrupt, SystemExit):
                                raise
                        except:
                                #traceback.print_exc()
                                xyt_file = xyt_data
                        else:
                                xyt_file = False
                                if input_label:
                                        input_partition, = find_artefacts(analyses, CellStats, input_label)
                                xyt_data = analyses.data
                if xyt_file:
                        xyt_data, xyt_files = load_xyt(xyt_files, return_paths=True, verbose=verbose)
                        analyses = Analyses(xyt_data)
                        if input_label is not None:
                                raise no_nesting_error
        else:
                if isinstance(xyt_data, Analyses):
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
                if not ref_distance:
                        ref_distance = input_partition.param.get('ref_distance', None)
                min_distance = input_partition.param.get('min_distance', min_distance)
                avg_distance = input_partition.param.get('avg_distance', avg_distance)
                max_distance = input_partition.param.get('max_distance', max_distance)
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
                # applies only to KDTreeMesh
                max_distance = rel_max_distance * ref_distance
                if method not in ['kdtree', 'gwr']:
                        warn('`rel_max_distance` is relevant only with `kdtree`', IgnoredInputWarning)

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
        if input_label:
                tess = NestedTessellations(scaler, input_partition, factory=constructor,
                        **tessellation_kwargs)
                xyt_data = data = input_partition.points
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
        try:
                if knn is None and radius is None:
                        cell_index = tess.cell_index(xyt_data, **partition_kwargs)
                else:
                        if 'min_location_count' not in partition_kwargs:
                                partition_kwargs['min_location_count'] = min_location_count
                        if 'metric' not in partition_kwargs:
                                partition_kwargs['metric'] = 'euclidean'
                        if radius is not None and 'radius' not in partition_kwargs:
                                partition_kwargs['radius'] = radius
                        cell_index = tess.cell_index(xyt_data, knn=knn, **partition_kwargs)
        except MemoryError:
                if verbose:
                        print(traceback.format_exc())
                warn('memory error: cannot assign points to cells', RuntimeWarning)
                cell_index = None

        stats = CellStats(xyt_data, tess, cell_index)

        # store some parameters together with the partition
        stats.param['method'] = method
        if transloc_length:
                stats.param['transloc_length'] = transloc_length
        else:
                stats.param['ref_distance'] = ref_distance
        if min_distance:
                stats.param['min_distance'] = min_distance
        if avg_distance:
                stats.param['avg_distance'] = avg_distance
        if max_distance:
                stats.param['max_distance'] = max_distance
        if knn:
                stats.param['knn'] = knn
        if radius:
                stats.param['radius'] = radius
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

                save_rwa(output_file, analyses, verbose, \
                        force=force or (force is not False and len(input_files)==1 and input_files[0]==output_file))

        if return_analyses:
                return analyses
        else:
                return stats




def cell_plot(cells, xy_layer=None, output_file=None, fig_format=None, \
        show=None, verbose=False, figsize=(24.0, 18.0), dpi=None, \
        location_count_hist=False, cell_dist_hist=False, location_dist_hist=False, \
        aspect=None, delaunay=None, locations={}, voronoi=None, colors=None, title=None, \
        cell_indices=None, label=None, input_label=None):
        """
        Partition plots.

        Plots a spatial representation of the tessellation and partition if data are 2D, and optionally
        histograms.

        Arguments:
                cells (str or CellStats or Analyses):
                        Path to a *.imt.rwa* file or :class:`~tramway.tessellation.CellStats`
                        instance or analysis tree; files and analysis trees may require
                        `label`/`input_label` to be defined.

                xy_layer ({None, 'delaunay', 'voronoi'}):
                        Overlay Delaunay or Voronoi graph over the data points. For 2D data only.
                        *Deprecated!* Please use `delaunay` and `voronoi` arguments instead.

                output_file (str):
                        Path to a file in which the figure will be saved. If `cells` is a path and
                        `fig_format` is defined, `output_file` is automatically set.

                fig_format (str):
                        Any image format supported by :func:`matplotlib.pyplot.savefig`.

                show (bool or str):
                        Makes `cell_plot` show the figure(s) which is the default behavior if and only
                        if the figures are not saved. If ``show='draw'``,
                        :func:`~matplotlib.pyplot.draw` is called instead of
                        :func:`~matplotlib.pyplot.show`.

                verbose (bool): Verbose output.

                figsize (pair of floats):
                        Passed to :func:`matplotlib.pyplot.figure`. Applies only to the spatial
                        representation figure.

                dpi (int):
                        Passed to :func:`matplotlib.pyplot.savefig`. Applies only to the spatial
                        representation figure.

                location_count_hist (bool):
                        Plot a histogram of point counts (per cell). If the figure is saved, the
                        corresponding file will have sub-extension *.hpc*.

                cell_dist_hist (bool):
                        Plot a histogram of distances between neighbour centroids. If the figure is
                        saved, the corresponding file will have sub-extension *.hcd*.

                location_dist_hist (bool):
                        Plot a histogram of distances between points from neighbour cells. If the figure
                        is saved, the corresponding file will have sub-extension *.hpd*.

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

        Notes:
                See also :mod:`tramway.plot.mesh`.

        """
        input_file = ''
        if not isinstance(cells, CellStats):
                if label is None:
                        labels = input_label
                else:
                        labels, label = label, None
                if not labels:
                        labels = ()
                elif not isinstance(labels, (tuple, list)):
                        labels = (labels, )
                if isinstance(cells, Analyses):
                        analyses, cells = cells, None
                else:
                        input_file, cells = cells, None
                        if isinstance(input_file, (tuple, list)):
                                if input_file[1:]:
                                        warn('can only process a single file', RuntimeWarning)
                                input_file = input_file[0]
                        try:
                                analyses = load_rwa(input_file)
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

        # guess back some input parameters (with backward "compatibility")
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

        if location_count_hist or cell_dist_hist or location_dist_hist:
                min_distance = cells.param.get('min_distance', 0)
                avg_distance = cells.param.get('avg_distance', None)
        try:
                min_location_count = cells.param['min_location_count']
        except (KeyError, AttributeError):
                min_location_count = 0

        print_figs = output_file or (input_file and fig_format)

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
        if dim == 2:
                fig = mplt.figure(figsize=figsize)
                figs.append(fig)
                if locations is not None:
                        if 'knn' in cells.param: # if knn <= min_count, min_count is actually ignored
                                tplt.plot_points(cells, **locations)
                        else:
                                tplt.plot_points(cells, min_count=min_location_count, **locations)
                if aspect is not None:
                        fig.gca().set_aspect(aspect)
                if voronoi is None:
                        voronoi = issubclass(type(cells.tessellation), Voronoi)
                if xy_layer != 'delaunay' and voronoi:
                        if not isinstance(voronoi, dict):
                                voronoi = {}
                        if cell_indices and 'centroid_style' not in voronoi:
                                voronoi['centroid_style'] = None
                        tplt.plot_voronoi(cells, **voronoi)
                        voronoi = True
                if xy_layer == 'delaunay' or delaunay: # Delaunay above Voronoi
                        if not isinstance(delaunay, dict):
                                delaunay = {}
                        tplt.plot_delaunay(cells, **delaunay)
                        delaunay = True
                if cell_indices:
                        if not isinstance(cell_indices, dict):
                                cell_indices = {}
                        tplt.plot_indices(cells, **cell_indices)
                if title:
                        if isinstance(title, str):
                                mplt.title(title)
                        elif delaunay == voronoi:
                                mplt.title(pp_method_name)
                        elif delaunay:
                                mplt.title(pp_method_name + ' based Delaunay')
                        elif voronoi:
                                mplt.title(pp_method_name + ' based Voronoi')


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
                if dim == 2:
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
                        mplt.plot((dmin, dmin), mplt.ylim(), 'r-')
                        mplt.plot((dmax, dmax), mplt.ylim(), 'r-')
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
                        mplt.plot((dmin, dmin), mplt.ylim(), 'r-')
                        mplt.plot((dmax, dmax), mplt.ylim(), 'r-')
                mplt.title(method_title)
                mplt.xlabel('inter-point distance (log)')
                if print_figs:
                        hpd_file = '{}.{}.{}'.format(filename, 'hpd', figext)
                        if verbose:
                                print('writing file: {}'.format(hpd_file))
                        fig.savefig(hpd_file)

        if show or not print_figs:
                if show == 'draw':
                        mplt.draw()
                elif show is not False:
                        mplt.show()
        else:
                for fig in figs:
                        mplt.close(fig)



def find_mesh(path, method=None, full_list=False):
        """
        *from version 0.3:* deprecated.
        """
        warn('`find_mesh`, `find_imt` and `find_partition` are deprecated in favor of `load_rwa`/`find_artefacts`', DeprecationWarning)
        if not isinstance(path, (tuple, list)):
                path = (path,)
        paths = []
        for p in path:
                if os.path.isdir(p):
                        paths.append([ os.path.join(p, f) for f in os.listdir(p) if f.endswith('.rwa') ])
                else:
                        if p.endswith('.rwa'):
                                ps = [p]
                        else:
                                d, p = os.path.split(p)
                                p, _ = os.path.splitext(p)
                                if d:
                                        ps = [ os.path.join(d, f) for f in os.listdir(d) \
                                                if f.startswith(p) and f.endswith('.rwa') ]
                                else:
                                        ps = [ f for f in os.listdir('.') \
                                                if f.startswith(p) and f.endswith('.rwa') ]
                        paths.append(ps)
        paths = list(itertools.chain(*paths))
        found = False
        for path in paths:
                try:
                        hdf = HDF5Store(path, 'r')
                        hdf.lazy = False
                        try:
                                cells = hdf.peek('cells')
                                if isinstance(cells, CellStats) and \
                                        (method is None or cells.param['method'] == method):
                                        found = True
                                        if cells.tessellation is None:
                                                cells._tessellation = hdf.peek('_tesselation', hdf.store['cells'])
                        except EnvironmentError:
                                print(traceback.format_exc())
                                warn('HDF5 libraries may not be installed', ImportWarning)
                        finally:
                                try:
                                        hdf.close()
                                except:
                                        pass
                except:
                        print(traceback.format_exc())
                        pass
                if found: break
        if found:
                if full_list:
                        path = paths
                return (path, cells)
        else:
                return (paths, None)


def find_imt(path, method=None, full_list=False):
        """
        Alias for :func:`find_mesh` for backward compatibility.

        *from version 0.3:* deprecated.
        """
        return find_mesh(path, method, full_list)

def find_partition(path, method=None, full_list=False):
        """
        Alias for :func:`find_mesh`.

        *from version 0.3:* deprecated.
        """
        return find_mesh(path, method, full_list)


