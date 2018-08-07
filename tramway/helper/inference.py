# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import *
from tramway.core.hdf5 import *
from tramway.inference import *
import tramway.inference as inference # inference.plugins
from tramway.helper.tessellation import *
from warnings import warn
import os
from time import time
import collections
import traceback
# no module-wide matplotlib import for head-less usage of `infer`
# in the case matplotlib's backend is interactive



def infer(cells, mode='D', output_file=None, partition={}, verbose=False, \
        localization_error=None, diffusivity_prior=None, potential_prior=None, jeffreys_prior=None, \
        max_cell_count=None, dilation=None, worker_count=None, min_diffusivity=None, \
        store_distributed=False, new_cell=None, new_group=None, constructor=None, cell_sampling=None, \
        grad=None, priorD=None, priorV=None, input_label=None, output_label=None, comment=None, \
        return_cells=None, profile=None, force=False, **kwargs):
        """
        Inference helper.

        Arguments:

                cells (str or CellStats or Analyses): data partition or path to partition file

                mode (str or callable): plugin name; see for example
                        :mod:`~tramway.inference.d` (``'d'``),
                        :mod:`~tramway.inference.df` (``'df'``),
                        :mod:`~tramway.inference.dd` (``'dd'``),
                        :mod:`~tramway.inference.dv` (``'dv'``);
                        can be also a function suitable for :meth:`~tramway.helper.inference.base.Distributed.run`

                output_file (str): desired path for the output map file

                partition (dict): keyword arguments for :func:`~tramway.helper.tessellation.find_partition`
                        if `cells` is a path; deprecated

                verbose (bool or int): verbosity level

                localization_error (float): localization error

                diffusivity_prior (float): prior diffusivity

                potential_prior (float): prior potential

                jeffreys_prior (float): Jeffreys' prior

                max_cell_count (int): if defined, divide the mesh into convex subsets of cells

                dilation (int): overlap of side cells if `max_cell_count` is defined

                worker_count (int): number of parallel processes to be spawned

                min_diffusivity (float): (possibly negative) lower bound on local diffusivities

                store_distributed (bool): store the :class:`~tramway.inference.base.Distributed` object
                        in the map file

                new_cell (callable): see also :func:`~tramway.inference.base.distributed`

                new_group (callable): see also :func:`~tramway.inference.base.distributed`

                constructor (callable): *deprecated*; see also :func:`~tramway.inference.base.distributed`;
                        please use `new_group` instead

                cell_sampling (str): either ``None``, ``'individual'`` or ``'group'``; may ignore
                        `max_cell_count` and `dilation`

                grad (callable or str): spatial gradient function; admits a callable (see
                        :meth:`~tramway.inference.base.Distributed.grad`) or any of '*grad1*',
                        '*gradn*'

                input_label (list): label path to the input :class:`~tramway.tessellation.base.Tessellation`
                        object in `cells` if the latter is an `Analyses` or filepath

                output_label (str): label for the resulting analysis instance

                comment (str): description message for the resulting analysis

                return_cells (bool): return a tuple with a :class:`~tramway.tessellation.base.CellStats`
                        object as extra element

                profile (bool or str): profile each child job if any;
                        if `str`, dump the output stats into *.prof* files;
                        if `tuple`, print a report with :func:`~pstats.Stats.print_stats` and
                        tuple elements as input arguments.

        Returns:

                Maps or pandas.DataFrame or tuple:

        `priorD` and `priorV` are legacy arguments.
        They are deprecated and `diffusivity_prior` and `potential_prior` should be used instead
        respectively.
        """
        if verbose:
                inference.plugins.verbose = True

        input_file = None
        all_analyses = analysis = None
        if isinstance(cells, str):
                try:
                        input_file = cells
                        all_analyses = load_rwa(input_file)
                        if output_file and output_file == input_file:
                                all_analyses = extract_analysis(all_analyses, input_label)
                        cells = None
                except KeyError:
                        # legacy format
                        input_file, cells = find_partition(cells, **partition)
                        if cells is None:
                                raise ValueError('no cells found')
                if verbose:
                        print('loading file: {}'.format(input_file))
        elif isinstance(cells, Analyses):
                all_analyses, cells = cells, None
        elif not isinstance(cells, CellStats):
                raise TypeError('wrong type for argument `cells`')

        if cells is None:
                if not all_analyses:
                        raise ValueError('no cells found')
                if not input_label:
                        labels = tuple(all_analyses.labels)
                        if labels[1:]:
                                raise ValueError('multiple instances; input_label is required')
                        input_label = labels[-1]
                if isinstance(input_label, (tuple, list)):
                        if input_label[1:]:
                                analysis = all_analyses
                                for label in input_label:#[:-1]
                                        analysis = analysis[label]
                                cells = analysis.data
                                #analysis = analysis[input_label[-1]]
                                #if not isinstance(cells, CellStats):
                                #       cells = analysis.data
                        else:
                                input_label = input_label[0]
                if cells is None:
                        analysis = all_analyses[input_label]
                        cells = analysis.data
                if not isinstance(cells, (CellStats, Distributed)):
                        raise ValueError('cannot find cells at the specified label')
        elif all_analyses is None:
                all_analyses = Analyses(cells.points)
                assert analysis is None
                analysis = Analyses(cells)
                all_analyses.add(analysis)
                assert input_label is None
                input_label = tuple(all_analyses.labels)

        if mode in ('D', 'DF', 'DD', 'DV'):
                mode = mode.lower()
        setup, module = inference.plugins[mode]

        if isinstance(analysis.data, Distributed):
                _map = analysis.data
        else:

                if cells is None or cells.tessellation is None:
                        raise ValueError('no cells found')

                # prepare the data for the inference
                if new_group is None:
                        if constructor is None:
                                new_group = Distributed
                        else:
                                new_group = constructor
                if grad is not None:
                        if not callable(grad):
                                if grad == 'grad1':
                                        grad = grad1
                                elif grad == 'gradn':
                                        grad = gradn
                                else:
                                        raise ValueError('unsupported gradient')
                                        grad = None
                        if grad is not None:
                                class Distr(new_group):
                                        def grad(self, *args, **kwargs):
                                                return grad(self, *args, **kwargs)
                                new_group = Distr
                detailled_map = distributed(cells, new_cell=new_cell, new_group=new_group)

                if cell_sampling is None:
                        try:
                                cell_sampling = setup['cell_sampling']
                        except KeyError:
                                pass
                multiscale = cell_sampling in ['individual', 'group', 'connected']
                if multiscale and max_cell_count is None:
                        if cell_sampling == 'individual':
                                max_cell_count = 1
                        #else: # adaptive scaling is no longer default
                        #       max_cell_count = 20
                if cell_sampling == 'connected':
                        multiscale_map = detailled_map.group(connected=True)
                        _map = multiscale_map
                elif max_cell_count:
                        if dilation is None:
                                if cell_sampling == 'individual':
                                        dilation = 0
                                else:
                                        dilation = 2
                        multiscale_map = detailled_map.group(max_cell_count=max_cell_count, \
                                adjacency_margin=dilation)
                        _map = multiscale_map
                else:
                        _map = detailled_map

                if store_distributed:
                        if output_label is None:
                                output_label = analysis.autoindex()
                        analysis.add(Analysis(_map), label=output_label)
                        analysis = analysis[output_label]
                        output_label = None

        runtime = time()

        if mode in inference.plugins:

                args = setup.get('arguments', {})
                for arg in ('localization_error', 'diffusivity_prior', 'potential_prior',
                                'jeffreys_prior', 'min_diffusivity', 'worker_count', 'verbose'):
                        try:
                                args[arg]
                        except KeyError:
                                pass
                        else:
                                val = eval(arg)
                                if val is not None:
                                        kwargs[arg] = val
                if profile:
                        kwargs['profile'] = profile
                x = _map.run(getattr(module, setup['infer']), **kwargs)

        else:
                raise ValueError('unknown ''{}'' mode'.format(mode))

        maps = Maps(x, mode=mode)
        for p in kwargs:
                if p not in ['worker_count']:
                        setattr(maps, p, kwargs[p])
        analysis.add(Analyses(maps), label=output_label, comment=comment)

        runtime = time() - runtime
        if verbose:
                print('{} mode: elapsed time: {}ms'.format(mode, int(round(runtime*1e3))))
        maps.runtime = runtime

        if input_file and not output_file:
                output_file = input_file

        if output_file:
                # store the result
                save_rwa(output_file, all_analyses, verbose, force=input_file == output_file or force)

        if return_cells == True: # NOT `is`
                return (maps, cells)
        elif return_cells == False:
                return maps
        elif input_file:
                if return_cells is not None:
                        warn("3-element return value will no longer be the default; pass return_cells='first' to maintain this behavior", FutureWarning)
                return (cells, mode, x)
        else:
                return x


def map_plot(maps, cells=None, clip=None, output_file=None, fig_format=None, \
        figsize=(24., 18.), dpi=None, aspect=None, show=None, verbose=False, \
        alpha=None, point_style=None, \
        label=None, input_label=None, mode=None, \
        **kwargs):
        """
        Plot scalar/vector 2D maps.

        Arguments:

                maps (str or Analyses or pandas.DataFrame or Maps): maps as a path to a rwa map file,
                        an analysis tree, a dataframe or a :class:`Maps`;
                        filepaths and analysis trees may require `label` (or equivalently `input_label`)
                        to be defined; dataframes and encapsulated maps require `cells` to be defined

                cells (CellStats or Tessellation or Distributed): mesh with optional partition

                clip (float): clips map values by absolute values;
                        if ``clip < 1``, it is the quantile at which to clip absolute values of the map;
                        otherwise it defines: ``threshold = median + clip * (third_quartile - first_quartile)``

                output_file (str): path to output file

                fig_format (str): for example '*.png*'

                figsize ((float, float)): figure size (width, height) in inches

                dpi (int): dots per inch

                aspect (float or str): aspect ratio or '*equal*'

                show (bool or str): call :func:`~matplotlib.pyplot.show`; if ``show='draw'``, call
                        :func:`~matplotlib.pyplot.draw` instead

                verbose (bool): verbosity level

                alpha (float): alpha value for scalar maps; useful in combination with `point_style`;
                        if ``False``, the alpha value is not explicitly set

                point_style (dict): if defined, points are overlaid

                label/input_label (int or str): analysis instance label

                mode (bool or str): inference mode; can be ``False`` so that mode information from
                        files, analysis trees and encapsulated maps are not displayed

        Extra keyword arguments may be passed to :func:`~tramway.plot.map.scalar_map_2d` and
        :func:`~tramway.plot.map.field_map_2d`.

        """
        # get cells and maps objects from the first input argument
        input_file = None
        if isinstance(maps, tuple):
                warn('`maps` as (CellStats, str, DataFrame) tuple are deprecated', DeprecationWarning)
                cells, mode, maps = maps
        elif isinstance(maps, (pd.DataFrame, Maps)):
                if cells is None:
                        raise ValueError('`cells` is not defined')
        elif isinstance(maps, Analyses):
                analyses = maps
                if label is None:
                        label = input_label
                cells, maps = find_artefacts(analyses, ((CellStats, Distributed), Maps), label)
        else: # `maps` is a file path
                input_file = maps
                if label is None:
                        label = input_label
                try:
                        analyses = load_rwa(input_file)
                        #if label:
                        #       analyses = extract_analysis(analyses, label)
                except KeyError:
                        print(traceback.format_exc())
                        try:
                                # old format
                                store = HDF5Store(input_file, 'r')
                                store.lazy = False
                                maps = peek_maps(store, store.store)
                        finally:
                                store.close()
                        try:
                                tess_file = maps.rwa_file
                        except AttributeError:
                                # even older
                                tess_file = maps.imt_file
                        if not isinstance(tess_file, str):
                                tess_file = tess_file.decode('utf-8')
                        tess_file = os.path.join(os.path.dirname(input_file), tess_file)
                        store = HDF5Store(tess_file, 'r')
                        store.lazy = False
                        try:
                                cells = store.peek('cells')
                                if cells.tessellation is None:
                                        cells._tessellation = store.peek('_tesselation', store.store['cells'])
                        finally:
                                store.close()
                except ImportError:
                        warn('HDF5 libraries may not be installed', ImportWarning)
                else:
                        cells, maps = find_artefacts(analyses, ((CellStats, Distributed), Maps), label)
        if isinstance(maps, Maps):
                if mode != False:
                        mode = maps.mode
                maps = maps.maps
        if isinstance(cells, Distributed):
                # fix for rwa-0.5 OrderedDict
                cells.cells = collections.OrderedDict((k, cells[k]) for k in range(max(cells.keys())+1) if k in cells )

        if not cells._lazy.get('bounding_box', True):
                maps = box_crop(maps, cells.bounding_box, cells.tessellation)

        xlim, ylim = kwargs.get('xlim', None), kwargs.get('ylim', None)
        if xlim and ylim:
                maps = box_crop(maps,
                        pd.DataFrame(
                                np.array([[xlim[0], ylim[0]], [xlim[1], ylim[1]]]),
                                columns=['x', 'y']),
                        cells.tessellation)

        # `mode` type may be inadequate because of loading a Py2-generated rwa file in Py3 or conversely
        if mode and not isinstance(mode, str):
                try: # Py2
                        mode = mode.encode('utf-8')
                except AttributeError: # Py3
                        mode = mode.decode('utf-8')

        # output filenames
        print_figs = output_file or (input_file and fig_format)

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

        # import graphics libraries with adequate backend
        if print_figs:
                import matplotlib
                try:
                        matplotlib.use('Agg') # head-less rendering (no X server required)
                except:
                        pass
        import matplotlib.pyplot as mplt
        import tramway.plot.mesh as xplt
        import tramway.plot.map  as yplt

        # identify and plot the possibly various maps
        figs = []
        nfig = 0

        all_vars = splitcoord(maps.columns)
        scalar_vars = {'diffusivity': 'D', 'potential': 'V'}
        scalar_vars = [ (v, scalar_vars.get(v, None)) for v in all_vars if len(all_vars[v]) == 1 ]

        for col, short_name in scalar_vars:

                col_kwargs = {}
                for a in kwargs:
                        if isinstance(kwargs[a], (dict, pd.Series, pd.DataFrame)) and col in kwargs[a]:
                                col_kwargs[a] = kwargs[a][col]
                        else:
                                col_kwargs[a] = kwargs[a]

                if figsize:
                        fig = mplt.figure(figsize=figsize, dpi=dpi)
                else:
                        fig = mplt.gcf()
                figs.append(fig)

                _map = maps[col]
                if isinstance(clip, (dict, pd.Series, pd.DataFrame)):
                        try:
                                __clip = clip[col]
                        except:
                                __clip = None
                else:
                        __clip = clip
                if __clip:
                        _map = _clip(_map, __clip)
                # debug
                if isinstance(maps, pd.DataFrame) and 'x' in maps.columns and col not in 'xyzt':
                        _map = maps[[ col for col in 'xyzt' if col in maps.columns ]].join(_map)
                #
                yplt.scalar_map_2d(cells, _map, aspect=aspect, alpha=alpha, **col_kwargs)

                if point_style is not None:
                        points = cells.descriptors(cells.points, asarray=True) # `cells` should be a `CellStats`
                        if 'color' not in point_style:
                                point_style['color'] = None
                        xplt.plot_points(points, **point_style)

                if mode:
                        if short_name:
                                title = '{} ({} - {} mode)'.format(short_name, col, mode)
                        else:
                                title = '{} ({} mode)'.format(col, mode)
                elif short_name:
                        title = '{} ({})'.format(short_name, col)
                else:
                        title = '{}'.format(col)
                mplt.title(title)

                if print_figs:
                        if maps.shape[1] == 1:
                                figfile = '{}.{}'.format(filename, figext)
                        elif short_name:
                                figfile = '{}_{}.{}'.format(filename, short_name.lower(), figext)
                        else:
                                figfile = '{}_{}.{}'.format(filename, nfig, figext)
                                nfig += 1
                        if verbose:
                                print('writing file: {}'.format(figfile))
                        fig.savefig(figfile, dpi=dpi)

        vector_vars = {'force': 'F'}
        vector_vars = [ (v, vector_vars.get(v, None)) for v in all_vars if len(all_vars[v]) == 2 ]
        for name, short_name in vector_vars:
                cols = all_vars[name]

                var_kwargs = {}
                for a in kwargs:
                        if isinstance(kwargs[a], (dict, pd.Series, pd.DataFrame)) and name in kwargs[a]:
                                var_kwargs[a] = kwargs[a][name]
                        else:
                                var_kwargs[a] = kwargs[a]

                if figsize:
                        fig = mplt.figure(figsize=figsize, dpi=dpi)
                else:
                        fig = mplt.gcf()
                figs.append(fig)

                _vector_map = maps[cols]
                if isinstance(clip, (dict, pd.Series, pd.DataFrame)):
                        try:
                                __clip = clip[name]
                        except:
                                __clip = None
                else:
                        __clip = clip
                if __clip:
                        _vector_map = _clip(_vector_map, __clip)
                if point_style is None:
                        yplt.field_map_2d(cells, _vector_map, aspect=aspect, **var_kwargs)
                else:
                        _scalar_map = _vector_map.pow(2).sum(1).apply(np.sqrt)
                        yplt.scalar_map_2d(cells, _scalar_map, aspect=aspect, alpha=alpha, **var_kwargs)
                        points = cells.descriptors(cells.points, asarray=True) # `cells` should be a `CellStats`
                        if 'color' not in point_style:
                                point_style['color'] = None
                        xplt.plot_points(points, **point_style)
                        yplt.field_map_2d(cells, _vector_map, aspect=aspect, overlay=True, **var_kwargs)

                extra = None
                if short_name:
                        main = short_name
                        extra = name
                else:
                        main = name
                if mode:
                        if extra:
                                extra += ' - {} mode'.format(mode)
                        else:
                                extra = '{} mode'.format(mode)
                if extra:
                        title = '{} ({})'.format(main, extra)
                else:
                        title = main
                mplt.title(title)

                if print_figs:
                        if maps.shape[1] == 1:
                                figfile = '{}.{}'.format(filename, figext)
                        else:
                                if short_name:
                                        ext = short_name.lower()
                                else:
                                        ext = name
                                figfile = '{}_{}.{}'.format(filename, ext, figext)
                        if verbose:
                                print('writing file: {}'.format(figfile))
                        fig.savefig(figfile, dpi=dpi)

        if show or not print_figs:
                if show == 'draw':
                        mplt.draw()
                elif show is not False:
                        mplt.show()
        elif print_figs:
                for fig in figs:
                        mplt.close(fig)


def _clip(m, q):
        if q <= 0:
                return m
        amplitude = m.pow(2)
        if m.shape[1:]:
                amplitude = amplitude.sum(1)
                columns = m.columns
        amplitude = amplitude.apply(np.sqrt)
        if q < 1:
                amax = amplitude.quantile(q)
        else:
                amax = amplitude.quantile(.5) + q * (amplitude.quantile(.75) - amplitude.quantile(.25))
                amax = amplitude[amplitude<=amax].max()
        amplitude = amplitude.values
        exceed = amplitude > amax
        factor = amax / amplitude[exceed]
        M, index, m = type(m), m.index, m.values
        if m.shape[1:]:
                m[exceed, :] = m[exceed, :] * factor[:, np.newaxis]
                m = M(m, columns=columns, index=index)
        else:
                m[exceed] = m[exceed] * factor
                m = M(m, index=index)
        return m


def box_crop(maps, bounding_box, tessellation):
        centers = tessellation.cell_centers[maps.index]
        try:
                vertices = tessellation.vertices
        except (KeyboardInterrupt, SystemExit):
                raise
        except:
                vertices = None
        dims = columns(tessellation.descriptors(bounding_box))
        for col, dim in enumerate(dims):
                lower, upper = bounding_box[dim]
                _in = (lower <= centers[:,col]) & (centers[:,col] <= upper)
                if vertices is not None:
                        _v_in = (lower <= vertices[:,col]) & (vertices[:,col] <= upper)
                        for i, j in enumerate(maps.index):
                                if _in[i]:
                                        continue
                                vs = tessellation.cell_vertices[j]
                                _in[i] = np.any(_v_in[vs])
                if col == 0:
                        inside = _in
                else:
                        inside &= _in
        return maps[inside]

