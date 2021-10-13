# -*- coding: utf-8 -*-

# Copyright © 2018-2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core import *
from tramway.core.hdf5 import *
import tramway.core.analyses.abc as abc
from tramway.inference import *
import tramway.inference as inference  # inference.plugins
import tramway.tessellation.time
import tramway.inference.time
from tramway.helper.tessellation import *
from warnings import warn
import os.path
import time
import collections
import traceback

# no module-wide matplotlib import for head-less usage of `infer`
# in the case matplotlib's backend is interactive


def infer0(
    cells,
    mode="D",
    output_file=None,
    partition={},
    verbose=False,
    localization_error=None,
    diffusivity_prior=None,
    potential_prior=None,
    jeffreys_prior=None,
    max_cell_count=None,
    dilation=None,
    worker_count=None,
    min_diffusivity=None,
    store_distributed=False,
    new_cell=None,
    new_group=None,
    constructor=None,
    cell_sampling=None,
    merge_threshold_count=False,
    grad=None,
    priorD=None,
    priorV=None,
    input_label=None,
    output_label=None,
    comment=None,
    return_cells=None,
    profile=None,
    force=False,
    **kwargs
):
    """
    Inference helper.

    Arguments:

        cells (str or Partition or tramway.core.analyses.base.Analyses):
            data partition or path to partition file

        mode (str or callable): plugin name; see for example
            :mod:`~tramway.inference.d` (``'d'``),
            :mod:`~tramway.inference.df` (``'df'``),
            :mod:`~tramway.inference.dd` (``'dd'``),
            :mod:`~tramway.inference.dv` (``'dv'``);
            can be also a function suitable for :meth:`~tramway.helper.inference.base.Distributed.run`

        output_file (str): desired path for the output map file

        partition (dict): keyword arguments for :func:`~tramway.helper.tessellation.find_partition`
            if `cells` is a path; **deprecated**

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

        constructor (callable): **deprecated**; see also :func:`~tramway.inference.base.distributed`;
            please use `new_group` instead

        cell_sampling (str): either ``None``, ``'individual'``, ``'group'`` or
            ``'connected'``; may ignore `max_cell_count` and `dilation`

        merge_threshold_count (int):
            Merge cells that are have a number of (trans-)locations lower than the
            number specified; each smaller cell is merged together with the nearest
            large-enough neighbour cell.

        grad (callable or str): spatial gradient function; admits a callable (see
            :meth:`~tramway.inference.base.Distributed.grad`) or any of '*grad1*',
            '*gradn*'

        input_label (list): label path to the input :class:`~tramway.tessellation.base.Tessellation`
            object in `cells` if the latter is an :class:`~tramway.core.analyses.base.Analyses`
            or filepath

        output_label (str): label for the resulting analysis instance

        comment (str): description message for the resulting analysis

        return_cells (bool): return a tuple with a :class:`~tramway.tessellation.base.Partition`
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

    if diffusivity_prior is None and priorD is not None:
        diffusivity_prior = priorD
    if potential_prior is None and priorV is not None:
        potential_prior = priorV

    input_file = None
    all_analyses = analysis = None
    if isinstance(cells, str):
        try:
            input_file = cells
            all_analyses = load_rwa(input_file, lazy=True)
            if output_file and output_file == input_file:
                all_analyses = extract_analysis(all_analyses, input_label)
            cells = None
        except KeyError:
            raise
            # legacy format
            input_file, cells = find_partition(cells, **partition)
            if cells is None:
                raise ValueError("no cells found")
        if verbose:
            print("loading file: {}".format(input_file))
    elif isinstance(cells, abc.Analyses):
        all_analyses, cells = cells, None
    elif not isinstance(cells, Partition):
        raise TypeError("wrong type for argument `cells`")

    if cells is None:
        if not all_analyses:
            raise ValueError("no cells found")
        if not input_label:
            labels = tuple(all_analyses.labels)
            if labels[1:]:
                raise ValueError("multiple instances; input_label is required")
            input_label = labels[-1]
        if isinstance(input_label, (tuple, list)):
            if input_label[1:]:
                analysis = all_analyses
                for label in input_label:  # [:-1]
                    analysis = analysis[label]
                cells = analysis.data
                # analysis = analysis[input_label[-1]]
                # if not isinstance(cells, Partition):
                #       cells = analysis.data
            else:
                input_label = input_label[0]
        if cells is None:
            analysis = all_analyses[input_label]
            cells = analysis.data
        if not isinstance(cells, (Partition, Distributed)):
            raise ValueError("cannot find cells at the specified label")
    elif all_analyses is None:
        all_analyses = Analyses(cells.points)
        assert analysis is None
        analysis = Analyses(cells)
        all_analyses.add(analysis)
        assert input_label is None
        input_label = tuple(all_analyses.labels)

    if mode in ("D", "DF", "DD", "DV"):
        mode = mode.lower()
    setup, module = inference.plugins[mode]

    if isinstance(analysis.data, Distributed):
        _map = analysis.data
    else:

        if cells is None or cells.tessellation is None:
            raise ValueError("no cells found")

        # prepare the data for the inference
        distributed_kwargs = {}
        if new_group is None:
            if constructor is None:
                if merge_threshold_count:
                    new_group = DistributeMerge
                    distributed_kwargs["new_group_kwargs"] = {
                        "min_location_count": merge_threshold_count
                    }
                else:
                    new_group = Distributed
            else:
                new_group = constructor
        if grad is not None:
            if not callable(grad):
                if grad == "grad1":
                    grad = grad1
                elif grad == "gradn":
                    grad = gradn
                else:
                    raise ValueError("unsupported gradient")
                    grad = None
            if grad is not None:

                class Distr(new_group):
                    def grad(self, *args, **kwargs):
                        return grad(self, *args, **kwargs)

                new_group = Distr
        detailled_map = distributed(
            cells, new_cell=new_cell, new_group=new_group, **distributed_kwargs
        )

        if cell_sampling is None:
            try:
                cell_sampling = setup["cell_sampling"]
            except KeyError:
                pass
        multiscale = cell_sampling in ["individual", "group", "connected"]
        if multiscale and max_cell_count is None:
            if cell_sampling == "individual":
                max_cell_count = 1
            # else: # adaptive scaling is no longer default
            #       max_cell_count = 20
        if cell_sampling == "connected":
            multiscale_map = detailled_map.group(connected=True)
            _map = multiscale_map
        elif max_cell_count:
            if dilation is None:
                if cell_sampling == "individual":
                    dilation = 0
                else:
                    dilation = 2
            multiscale_map = detailled_map.group(
                max_cell_count=max_cell_count, adjacency_margin=dilation
            )
            _map = multiscale_map
        else:
            _map = detailled_map

        if store_distributed:
            if output_label is None:
                output_label = analysis.autoindex()
            analysis.add(Analysis(_map), label=output_label)
            analysis = analysis[output_label]
            output_label = None

    runtime = time.time()

    if mode in inference.plugins:

        args = setup.get("arguments", {})
        for arg in (
            "localization_error",
            "diffusivity_prior",
            "potential_prior",
            "jeffreys_prior",
            "min_diffusivity",
            "worker_count",
            "verbose",
        ):
            try:
                args[arg]
            except KeyError:
                pass
            else:
                val = eval(arg)
                if val is not None:
                    kwargs[arg] = val
        if profile:
            kwargs["profile"] = profile
        x = _map.run(getattr(module, setup["infer"]), **kwargs)

    else:
        raise ValueError("unknown '{}' mode".format(mode))

    if isinstance(x, tuple):
        maps = Maps(x[0], mode=mode, posteriors=x[1])
        if x[2:]:
            maps.other = x[2:]  # Python 3 only
    else:
        maps = Maps(x, mode=mode)

    for p in kwargs:
        if p not in ["worker_count"]:
            setattr(maps, p, kwargs[p])
    analysis.add(Analyses(maps), label=output_label, comment=comment)

    runtime = time.time() - runtime
    if verbose:
        print("{} mode: elapsed time: {}ms".format(mode, int(round(runtime * 1e3))))
    maps.runtime = runtime

    if input_file and not output_file:
        output_file = input_file

    if output_file:
        # store the result
        save_rwa(
            output_file, all_analyses, verbose, force=input_file == output_file or force
        )

    if return_cells == True:  # NOT `is`
        return (maps, cells)
    elif return_cells == False:
        return maps
    elif input_file:
        if return_cells is not None:
            warn(
                "3-element return value will no longer be the default; pass return_cells='first' to maintain this behavior",
                FutureWarning,
            )
        return (cells, mode, x)
    else:
        return x


def map_plot0(
    maps,
    cells=None,
    clip=None,
    output_file=None,
    fig_format=None,
    figsize=None,
    dpi=None,
    aspect=None,
    show=None,
    verbose=False,
    alpha=None,
    point_style=None,
    feature=None,
    variable=None,
    segment=None,
    label=None,
    input_label=None,
    mode=None,
    title=True,
    inferencemap=False,
    use_bokeh=None,
    **kwargs
):
    """
    Plot scalar/vector 2D maps.

    For consistency with *InferenceMAP*, the effective potential is showed
    in :math:`µm^{-1}` while -- using ``unit='std'`` -- the colorbar
    mentions :math:`k_{B}T`, whereas no conversion factor is applied.

    Arguments:

        maps (str or tramway.core.analyses.base.Analyses or pandas.DataFrame or Maps):
            maps as a path to a rwa map file,
            an analysis tree, a dataframe or a :class:`Maps`;
            filepaths and analysis trees may require `label` (or equivalently `input_label`)
            to be defined; dataframes and encapsulated maps require `cells` to be defined

        cells (Partition or Tessellation or Distributed): mesh with optional partition

        clip (float): clips map values by absolute values;
            if ``clip < 1``, it is the quantile at which to clip absolute values of the map;
            otherwise it defines: ``threshold = median + clip * (third_quartile - first_quartile)``

        output_file (str): path to output file

        fig_format (str): for example '*.png*'

        figsize (bool or (float, float)): figure size (width, height) in inches;
            `figsize` is defined if multiple figures are drawn or the figures are printed
            to files, which opens a new figure for each plot;
            this can be prevented setting `figsize` to ``False``

        dpi (int): dots per inch

        aspect (float or str): aspect ratio or '*equal*'

        show (bool or str): call :func:`~matplotlib.pyplot.show`; if ``show='draw'``, call
            :func:`~matplotlib.pyplot.draw` instead.
            `show` is ``True`` if the figures are not printed to files; to maintain this
            default behaviour in future releases, set `show` to ``True`` from now on.

        verbose (bool): verbosity level

        alpha (float): alpha value for scalar maps; useful in combination with `point_style`;
            if ``False``, the alpha value is not explicitly set

        point_style (dict): if defined, points are overlaid

        feature/variable (str): feature name (e.g. 'diffusivity', 'force')

        segment (int): segment index;
            if multiple time segments were defined, show only this segment

        label/input_label (int or str): analysis instance label

        mode (bool or str): inference mode; can be ``False`` so that mode information from
            files, analysis trees and encapsulated maps are not displayed

        title (bool or str): add titles to the figures, based on the feature name;
            from version *0.4*, the inference mode is no longer appended

        inferencemap (bool): scales the arrows wrt to cell size only; for field maps only

        xlim (array-like): min and max values for the x-axis; this argument is keyworded only

        ylim (array-like): min and max values for the y-axis; this argument is keyworded only

        zlim (array-like): min and max values for the z-axis; this argument is keyworded only;
            applies only to scalar maps that are consequently plotted in 3D

        clim (array-like): min and max values for the colormap; this argument is keyworded only

        unit (str): colorbar label; can be :const:`'std'` to take usual units,
            with space coordinates in *µm* and and times in *s*

    Extra keyword arguments may be passed to :func:`~tramway.plot.map.scalar_map_2d`,
    :func:`~tramway.plot.map.field_map_2d` and :func:`scalar_map_3d`.
    They can be dictionnaries with feature names as keys and the corresponding values for the
    parameters.

    """
    # get cells and maps objects from the first input argument
    input_file = None
    if isinstance(maps, tuple):
        warn(
            "`maps` as (Partition, str, DataFrame) tuple are deprecated",
            DeprecationWarning,
        )
        cells, mode, maps = maps
    elif isinstance(maps, (pd.DataFrame, Maps, pd.Series)):
        if cells is None:
            raise ValueError("`cells` is not defined")
    elif isinstance(maps, abc.Analyses):
        analyses = maps
        if label is None:
            label = input_label
        cells, maps = find_artefacts(analyses, ((Partition, Distributed), Maps), label)
    elif isinstance(maps, str):  # `maps` is a file path
        input_file = maps
        if not os.path.isfile(input_file):
            raise OSError("cannot find file: {}".format(input_file))
        if label is None:
            label = input_label
        try:
            analyses = load_rwa(input_file, lazy=True)
            # if label:
            #       analyses = extract_analysis(analyses, label)
        except KeyError:
            print(traceback.format_exc())
            from rwa import HDF5Store

            store = HDF5Store(input_file, "r")
            store.lazy = False
            try:
                # old format
                maps = peek_maps(store, store.store)
            finally:
                store.close()
            try:
                tess_file = maps.rwa_file
            except AttributeError:
                # even older
                tess_file = maps.imt_file
            if not isinstance(tess_file, str):
                tess_file = tess_file.decode("utf-8")
            tess_file = os.path.join(os.path.dirname(input_file), tess_file)
            store = HDF5Store(tess_file, "r")
            store.lazy = False
            try:
                cells = store.peek("cells")
                if cells.tessellation is None:
                    cells._tessellation = store.peek(
                        "_tesselation", store.store["cells"]
                    )
            finally:
                store.close()
        except ImportError:
            warn("HDF5 libraries may not be installed", ImportWarning)
        else:
            cells, maps = find_artefacts(
                analyses, ((Partition, Distributed), Maps), label
            )
    else:
        raise TypeError("unsupported type for maps: {}".format(type(maps)))
    if isinstance(maps, Maps):
        if mode != False:
            mode = maps.mode
        maps = maps.maps
    elif isinstance(maps, pd.Series):
        maps = pd.DataFrame(maps.values, index=maps.index, columns=["unknown feature"])
    if isinstance(cells, Distributed):
        # fix for rwa-0.5 OrderedDict
        cells.cells = collections.OrderedDict(
            (k, cells[k]) for k in range(max(cells.keys()) + 1) if k in cells
        )

    if not cells._lazy.get("bounding_box", True):
        maps = box_crop(maps, cells.bounding_box, cells.tessellation)

    xlim, ylim, zlim = (
        kwargs.get("xlim", None),
        kwargs.get("ylim", None),
        kwargs.pop("zlim", None),
    )
    if xlim and ylim:
        maps = box_crop(
            maps,
            pd.DataFrame(
                np.array([[xlim[0], ylim[0]], [xlim[1], ylim[1]]]), columns=["x", "y"]
            ),
            cells.tessellation,
        )

    unit = kwargs.pop("unit", None)
    if unit == "std":
        # standard units are defined at multiple locations:
        # * tramway.plot.bokeh.analyzer.Controller.draw_map
        # * tramway.helper.inference.map_plot
        # * tramway.analyzer.mapper.mpl.Mpl.clabel
        unit = {
            "diffusivity": r"$\mu\rm{m}^2\rm{s}^{-1}$",
            "potential": r"$k_{\rm{B}}T$",
            "force": r"$k_{\rm{B}}T\mu\rm{m}^{-1}$",
            "drift": r"$\mu\rm{m}\rm{s}^{-1}$",
        }

    # identify time segments, if any
    try:
        import tramway.tessellation.time as lattice

        with_segments = (
            isinstance(cells.tessellation, lattice.TimeLattice)
            and cells.tessellation.spatial_mesh is not None
        )
    except ImportError:
        with_segments = False
    if with_segments:
        if segment is None:
            raise ValueError("`segment` is required")
        elif isinstance(segment, (tuple, list)):
            if segment[1:]:
                warn(
                    "cannot plot multiple segments in a single `map_plot` call",
                    RuntimeWarning,
                )
            segment = segment.pop()
            print("plotting segment {}".format(segment))
        _cells, cells = cells, cells.tessellation.split_segments(cells)[segment]
    elif segment is not None:
        warn("cannot find time segments", RuntimeWarning)

    # `mode` type may be inadequate because of loading a Py2-generated rwa file in Py3 or conversely
    if mode and not isinstance(mode, str):
        try:  # Py2
            mode = mode.encode("utf-8")
        except AttributeError:  # Py3
            mode = mode.decode("utf-8")

    # determine whether the figures should be printed to file or not
    print_figs = output_file or (input_file and fig_format)

    # output filenames
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
        if use_bokeh is None and figext == "html":
            use_bokeh = True

    # figure size
    new_fig = bool(figsize) or (print_figs and figsize is not False)
    if figsize in (None, True):
        if use_bokeh:
            figsize = None
        else:
            figsize = (12.0, 9.0)

    # import graphics libraries with adequate backend
    if print_figs:
        if not use_bokeh:
            import matplotlib

            try:
                matplotlib.use("Agg")  # head-less rendering (no X server required)
            except:
                pass
    if use_bokeh:
        import bokeh.plotting as mplt
        import tramway.plot.bokeh as tplt

        if figsize:
            fig_kwargs = dict(plot_width=figsize[0], plot_height=figsize[1])
        else:
            fig_kwargs = {}
    else:
        import matplotlib.pyplot as mplt
        import tramway.plot as tplt

        if "figure" in kwargs:
            fig = kwargs["figure"]
        if point_style is not None and "axes" in kwargs:
            point_style["axes"] = kwargs["axes"]

    # identify and plot the possibly various maps
    figs = []
    nfig = 0

    if feature is None:
        feature = variable
    all_vars = dict(splitcoord(maps.columns))  # not a defaultdict
    if isinstance(feature, (frozenset, set, tuple, list)):
        all_vars = {v: all_vars[v] for v in feature}
    elif feature is not None:
        all_vars = {feature: all_vars[feature]}

    standard_kwargs = {}
    differential_kwargs = {}
    for kw in kwargs:
        arg = kwargs[kw]
        if isinstance(arg, dict):
            keys = arg.keys()
        elif isinstance(arg, pd.Series):
            keys = arg.index
        elif isinstance(arg, pd.DataFrame):
            keys = arg.columns
        else:
            keys = ()
        if 0 < len(keys) and all(key in all_vars for key in keys):
            differential_kwargs[kw] = arg
        else:
            standard_kwargs[kw] = arg
    kwargs = standard_kwargs

    if unit:
        if isinstance(unit, dict):
            differential_kwargs["unit"] = unit
        else:
            kwargs["unit"] = unit

    scalar_vars = {"diffusivity": "D", "potential": "V"}
    scalar_vars = [
        (v, scalar_vars.get(v, None)) for v in all_vars if len(all_vars[v]) == 1
    ]

    for col, short_name in scalar_vars:

        col_kwargs = dict(standard_kwargs)
        for kw in differential_kwargs:
            try:
                arg = differential_kwargs[kw][col]
            except KeyError:
                pass
            else:
                col_kwargs[kw] = arg

        if use_bokeh:
            if print_figs:
                mplt.output_file(output_file)
            fig = mplt.figure(**fig_kwargs)
            col_kwargs["figure"] = fig
            if point_style is not None:
                point_style["figure"] = fig
        else:
            if "figure" in kwargs:
                pass
            elif new_fig or figs:
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
        if isinstance(maps, pd.DataFrame) and "x" in maps.columns and col not in "xyzt":
            _map = maps[[col for col in "xyzt" if col in maps.columns]].join(_map)
        #

        # split time segments, if any
        if with_segments:
            if "clim" not in col_kwargs:
                col_kwargs["clim"] = [_map.min(), _map.max()]
            _map = _cells.tessellation.split_frames(_map)
            try:
                _map = _map[segment]
            except IndexError:
                raise IndexError(
                    "segment index {} is out of bounds (max {})".format(
                        segment, len(_map) - 1
                    )
                )

        if zlim is None:
            plot = tplt.scalar_map_2d
        else:
            plot = tplt.scalar_map_3d
            col_kwargs["zlim"] = zlim

        plot(cells, _map, aspect=aspect, alpha=alpha, **col_kwargs)

        if point_style is not None:
            points = cells.descriptors(
                cells.points, asarray=True
            )  # `cells` should be a `Partition`
            if "color" not in point_style:
                point_style["color"] = None
            tplt.plot_points(points, **point_style)

        if title and not use_bokeh:
            if isinstance(title, str):
                _title = title
            # elif mode:
            #    if short_name:
            #        _title = '{} ({} - {} mode)'.format(short_name, col, mode)
            #    else:
            #        _title = '{} ({} mode)'.format(col, mode)
            # elif short_name:
            #    _title = '{} ({})'.format(short_name, col)
            else:
                _title = "{}".format(col)
            try:
                axes = kwargs["axes"]
            except KeyError:
                mplt.title(_title)
            else:
                axes.set_title(_title)

        if print_figs and not use_bokeh:
            if maps.shape[1] == 1:
                figfile = "{}.{}".format(filename, figext)
            elif short_name:
                figfile = "{}_{}.{}".format(filename, short_name.lower(), figext)
            else:
                figfile = "{}_{}.{}".format(filename, nfig, figext)
                nfig += 1
            if verbose:
                print("writing file: {}".format(figfile))
            fig.savefig(figfile, dpi=dpi)

    vector_vars = {"force": "F"}
    vector_vars = [
        (v, vector_vars.get(v, None)) for v in all_vars if len(all_vars[v]) == 2
    ]

    for name, short_name in vector_vars:
        cols = all_vars[name]

        var_kwargs = dict(standard_kwargs)
        for kw in differential_kwargs:
            try:
                arg = differential_kwargs[kw][name]
            except KeyError:
                pass
            else:
                var_kwargs[kw] = arg

        plot = tplt.field_map_2d
        if point_style is not None:
            var_kwargs["overlay"] = True
        if inferencemap:
            var_kwargs["inferencemap"] = inferencemap

        if use_bokeh:
            if print_figs:
                mplt.output_file(output_file)
            fig = mplt.figure(**fig_kwargs)
            var_kwargs["figure"] = fig
            if point_style is not None:
                point_style["figure"] = fig
        else:
            if "figure" in kwargs:
                pass
            elif new_fig or figs:
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

        # split time segments, if any
        if with_segments:
            if "clim" not in var_kwargs:
                _scalar_map = _vector_map.pow(2).sum(1).apply(np.sqrt)
                var_kwargs["clim"] = [
                    _scalar_map.values.min(),
                    _scalar_map.values.max(),
                ]
            _vector_map = _cells.tessellation.split_frames(_vector_map)[segment]

        if point_style is not None:
            _scalar_map = _vector_map.pow(2).sum(1).apply(np.sqrt)
            tplt.scalar_map_2d(
                cells, _scalar_map, aspect=aspect, alpha=alpha, **var_kwargs
            )
            points = cells.descriptors(
                cells.points, asarray=True
            )  # `cells` should be a `Partition`
            if "color" not in point_style:
                point_style["color"] = None
            tplt.plot_points(points, **point_style)

        plot(cells, _vector_map, aspect=aspect, **var_kwargs)

        extra = None
        if short_name:
            main = short_name
            extra = name
        else:
            main = name

        if title and not use_bokeh:
            if isinstance(title, str):
                _title = title
            else:
                # if mode:
                #    if extra:
                #        extra += ' - {} mode'.format(mode)
                #    else:
                #        extra = '{} mode'.format(mode)
                # if extra:
                #    _title = '{} ({})'.format(main, extra)
                # else:
                #    _title = main
                _title = name
            try:
                axes = kwargs["axes"]
            except KeyError:
                mplt.title(_title)
            else:
                axes.set_title(_title)

        if print_figs and not use_bokeh:
            if maps.shape[1] == 1:
                figfile = "{}.{}".format(filename, figext)
            else:
                if short_name:
                    ext = short_name.lower()
                else:
                    ext = name
                figfile = "{}_{}.{}".format(filename, ext, figext)
            if verbose:
                print("writing file: {}".format(figfile))
            fig.savefig(figfile, dpi=dpi)

    if show and not print_figs:
        if show == "draw":
            if use_bokeh:
                warn("draw not implemented with bokeh", RuntimeWarning)
            else:
                mplt.draw()
        elif show is not False:
            if use_bokeh:
                for fig in figs:
                    mplt.show(fig)
            else:
                mplt.show()
    elif print_figs and not use_bokeh:
        for fig in figs:
            mplt.close(fig)
    else:
        return figs


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
        amax = amplitude.quantile(0.5) + q * (
            amplitude.quantile(0.75) - amplitude.quantile(0.25)
        )
        amax = amplitude[amplitude <= amax].max()
    amplitude = amplitude.values
    exceed = amplitude > amax
    factor = amax / amplitude[exceed]
    M, index, m = type(m), m.index, np.array(m.values)
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
        _in = (lower <= centers[:, col]) & (centers[:, col] <= upper)
        if vertices is not None:
            _v_in = (lower <= vertices[:, col]) & (vertices[:, col] <= upper)
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


__all__ = ["infer0", "map_plot0", "_clip", "box_crop"]
