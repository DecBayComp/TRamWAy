# -*- coding: utf-8 -*-

# Copyright © 2020-2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import AnalyzerNode
from ..artefact import Analysis
import warnings
import numpy as np
import pandas as pd
from tramway.tessellation.base import Partition
from tramway.inference.base import Maps
from tramway.plot.map import *
from matplotlib import animation


class PatchCollection(object):
    """
    Makes glyphs for space bins to be drawn as patches.

    For constant spatial mesh and 2D localization data only.

    Vector maps are not supported yet.
    """
    __slots__ = ('glyphs','bin_indices','point_glyphs','time_label','format_time_label')
    def __init__(self, ax, mesh, bounding_box=None, overlay_locations=False,
            time_label_fmt='t= %t-%ts', time_label_loc=(.01, 1.01), **kwargs):
        """
        Arguments:
        
            ax (Axes): matplotlib axes.
            
            mesh (*Voronoi* or *Partition*): space bins.

            bounding_box (DataFrame): map bounding box;
                required if *mesh* is not `Partition`.

            overlay_locations (*bool* or *dict*): styling options
                for overlaid locations.

            time_label_fmt (str): format string for the time label;
                the pattern :const:`%t` represent time
                and can appear twice;
                if it appears twice, each are a different time bound;
                if once, it is replaced by the middle time point
                between the bounds.
            
        Extra keyword arguments are passed to :func:`~tramway.plot.map.scalar_map_2d`.
        """
        if not isinstance(mesh, Partition):
            mesh = Partition(pd.DataFrame([], columns=list('nxyt')),
                    mesh,
                    location_count=np.full(mesh.number_of_cells, np.iinfo(np.int).max, dtype=np.int),
                    bounding_box=bounding_box)
        self.glyphs, self.bin_indices = scalar_map_2d(
                mesh, pd.Series(np.zeros(mesh.number_of_cells, dtype=np.float)),
                axes=ax, return_patches=True, **kwargs)
        if overlay_locations not in (None, False):
            point_style = dict(marker='.', markersize=4, linestyle='none', color='r', alpha=.2)
            if isinstance(overlay_locations, dict):
                point_style.update(overlay_locations)
            self.point_glyphs, = ax.plot([], [], **point_style)
        else:
            self.point_glyphs = None
        if time_label_fmt:
            c = time_label_fmt.count('%t')
            fmt= time_label_fmt.replace('%t','{}')
            format_time = lambda t: '{:.3f}'.format(t).rstrip('0').rstrip('.')
            if c == 2:
                self.format_time_label = lambda ts: fmt.format(*[ format_time(t) for t in ts ])
            elif c == 1:
                self.format_time_label = lambda ts: fmt.format(format_time(np.mean(ts)))
            x, y = time_label_loc
            self.time_label = ax.text(x, y, '', transform=ax.transAxes)
        else:
            self.time_label = None
    def plot(self, map, times=None, locations=None):
        """
        Updates the color of the patches.
        
        Arguments:
        
            map (Series): parameter values.

            times (tuple): time segment bounds, in seconds.

            locations (DataFrame): segment locations.
            
        Returns:
        
            tuple: sequence of updated glyphs.
        """
        n = self.bin_indices.max()
        if map.index.size:
            n = max(map.index.max(), n)
        values = np.full(n+1, np.nan, dtype=np.float)
        values[map.index] = map.values
        self.glyphs.set_array(values[self.bin_indices])
        ret = [self.glyphs]
        if self.point_glyphs and locations is not None:
            self.point_glyphs.set_data(locations['x'].values, locations['y'].values)
            ret.append(self.point_glyphs)
        if self.time_label and times is not None:
            label = self.format_time_label(times)
            self.time_label.set_text(label)
            ret.append(self.time_label)
        return tuple(ret)
    def init_func(self):
        """
        To be passed as argument `init_func` to `FuncAnimation`.
        """
        if self.point_glyphs:
            if self.time_label:
                return self.glyphs, self.point_glyphs, self.time_label
            else:
                return self.glyphs, self.point_glyphs
        else:
            if self.time_label:
                return self.glyphs, self.time_label
            else:
                return self.glyphs,
    def animate(self, map):
        """
        To be passed as second positional argument to `FuncAnimation`.
        
        Arguments:
        
            map (Series): parameter values;
                can also be a tuple with the actual *map* argument being last.
            
        Returns:
        
            callable: list of glyphs to update.
        """
        if isinstance(map, tuple):
            ts, sampling, map = map
            df = sampling.points
        else:
            df = None
            ts = None
        return self.plot(map, times=ts, locations=df)


class FuncAnimations(animation.FuncAnimation):
    """
    Proxy for the :class:`matplotlib.animation.FuncAnimation` class.

    It inherits from :class:`~matplotlib.animation.FuncAnimation` for typing purposes,
    but does not reuse (or intend to reuse) any attribute from the parent class,
    very much like if it inherited from :class:`object` instead.
    """
    def __init__(self, fig, *args, **kwargs):
        self._proxied = None
        self._rendered = False
        self._fig = fig
        self._func = []
        self._frames = []
        self._init_func = []
        self._fargs = []
        self._kwargs = {}
        if args:
            self.add(fig, *args, **kwargs)

    def add(self, fig, func, frames=None, init_func=None, fargs=None, **kwargs):
        if self._proxied is not None:
            raise RuntimeError('cannot append more animations to an already rendered FuncAnimation')
        assert fig is self._fig
        self._func.append(func)
        self._frames.append(frames)
        self._init_func.append(init_func)
        self._fargs.append(fargs if fargs else ())
        self._kwargs.update(kwargs)
        return self

    @property
    def proxied(self):
        if self._proxied is None:
            if self.rendered:
                raise ValueError('a FuncAnimations movie can be rendered only once')

            def func(arg):
                combined_ret = []
                for f, arg0, args in zip(self._func, arg, self._fargs):
                    if arg0 is not None:
                        ret = f(arg0, *args)
                        if ret is None:
                            pass
                        elif isinstance(ret, (tuple, list, frozenset, set)):
                            combined_ret += list(ret)
                        else:
                            combined_ret.append(ret)
                if combined_ret:
                    return tuple(combined_ret)

            def init_func():
                combined_ret = []
                for f in self._init_func:
                    ret = f()
                    if ret is None:
                        pass
                    elif isinstance(ret, (tuple, list, frozenset, set)):
                        combined_ret += list(ret)
                    else:
                        combined_ret.append(ret)
                if combined_ret:
                    return tuple(combined_ret)

            def frames():
                updated_generators = [ None if gen is None else iter(gen) for gen in self._frames ]
                while True:
                    dry_run = True
                    ret = []
                    generators, updated_generators = updated_generators, []
                    for gen in generators:
                        if gen is None:
                            ret.append(None)
                        else:
                            try:
                                ret.append(next(gen))
                            except StopIteration:
                                gen = None
                                ret.append(None)
                            else:
                                dry_run = False
                        updated_generators.append(gen)
                    if dry_run:
                        break
                    else:
                        yield ret

            self._proxied = animation.FuncAnimation(
                    self._fig, func, frames(), init_func, **self._kwargs)

        return self._proxied

    @property
    def rendered(self):
        return self._rendered

    def __del__(self):
        self._proxied = None
        self._func = []
        self._frames = []
        self._init_func = []
        self._fargs = []
        self._kwargs = {}

    def new_frame_seq(self):
        return self.proxied.new_frame_seq()

    def new_saved_frame_seq(self):
        return self.proxied.new_saved_frame_seq()

    def save(self, *args, **kwargs):
        try:
            return self.proxied.save(*args, **kwargs)
        finally:
            self._rendered = True

    def to_html5_video(self, *args, **kwargs):
        try:
            return self.proxied.to_html5_video(*args, **kwargs)
        finally:
            self._rendered = True

    def to_jshtml(self, *args, **kwargs):
        try:
            return self.proxied.to_jshtml(*args, **kwargs)
        finally:
            self._rendered = True


class Mpl(AnalyzerNode):
    """
    Matplotlib interface for maps.
    """
    __slots__ = ()
    @property
    def plotter(self):
        return PatchCollection
    @property
    def default_log(self):
        return 'log'
    def clabel(self, feature, kwargs, logscale=None, map_kwargs=None):
        """
        Extracts from `kwargs` arguments related to the colorbar label
        and adds a :const:`'unit'` or :const:`clabel` argument to `map_kwargs`
        if necessary.
        """
        try:
            unit = kwargs.pop('unit')
        except KeyError:
            unit = None
        else:
            if unit == 'std':
                if logscale is True:
                    logscale = self.default_log
                if logscale == 'natural':
                    logscale = 'log'
                if logscale:
                    if logscale.startswith('log'):
                        if logscale == 'log':
                            scale = lambda u: f'$\\rm{{log}}. {u}$'
                        else:
                            base = logscale[3:]
                            scale = lambda u: f'$\\rm{{log_{{{base}}}}}. {u}$'
                    else:
                        scale = lambda u: f'$\\rm{{{logscale}}}. {u}$'
                else:
                    scale = lambda u: f'${u}$'
                # standard units are defined at multiple locations:
                # * tramway.plot.bokeh.analyzer.Controller.draw_map
                # * tramway.helper.inference.map_plot
                # * tramway.analyzer.mapper.mpl.Mpl.clabel
                unit = dict(
                        diffusivity=scale(r'\mu\rm{m}^2\rm{s}^{-1}'),
                        potential=scale(r'k_{\rm{B}}T'),
                        force=scale(r'k_{\rm{B}}T\mu\rm{m}^{-1}'),
                        drift=scale(r'\mu\rm{m}\rm{s}^{-1}'),
                    ).get(feature, None)
            if unit is not None:
                if map_kwargs is None:
                    map_kwargs = kwargs
                map_kwargs['unit'] = unit
        return unit
    def animate(self, fig, maps, feature, sampling=None,
            overlay_locations=False, axes=None, aspect='equal', logscale=False,
            composable=True, **kwargs):
        """
        Animates the time-segmented inference parameters.

        Vector features are represented as amplitude.

        The `RWAnalyzer.time` attribute is accessed.

        Arguments:

            fig (matplotlib.figure.Figure): figure.

            maps (*Analysis* or *Maps*): map series.

            feature (str): parameter to be drawn.

            sampling (*Analysis* or *Partition*):
                spatial bins and time segments;
                optional only if *maps* is an :class:`~tramway.analyzer.Analysis`.

            overlay_locations (*bool* or *dict*):
                styling options for the overlaid locations.

            axes (matplotlib.axes.Axes): figure axes.

            aspect (*str* or None): aspect ratio.

            logscale (*bool* or *str*): transform the color-coded values in
                natural logarithm; can also be 'log', 'natural' or 'log10'.

            composable (bool): returns an overloaded :class:`FuncAnimation`
                object that can be passed in place of argument `fig` in later
                calls to :meth:`animate` so to stack animations on
                different axes (`axes`) of a same figure (`fig`).

        Returns:

            matplotlib.animation.FuncAnimation: animation object.

        Extra input arguments are passed to :class:`~matplotlib.animation.FuncAnimation`
        or :class:`PatchCollection` (and :func:`~tramway.plot.map.scalar_map_2d`).

        Notebook example:

        .. code-block::python

            # Cell 1
            from tramway.analyzer import *

            a          = RWAnalyzer()
            # ...

            features   = ('diffusivity', 'potential')

        .. code-block::python

            # Cell 2
            %% capture
            from matplotlib import pyplot as plt

            roi_obj   = first(a.roi.as_support_regions())

            sampling  = roi_obj.get_sampling()
            maps      = sampling.get_child()

            fig, axes = plt.subplots(1, len(features), figsize=(15, 5))

            for ax, ftr in zip(axes, features):
                fig   = a.mapper.mpl.animate(fig, maps,
                        feature=ftr, unit='std', axes=ax,
                        overlay_locations=True, composable=True)

            movie     = fig

        .. code-block::python

            # Cell 3
            from IPython.display import HTML

            HTML(movie.to_jshtml())

        Note: the above example seems to work equally well without
        ``composable=True``.

        """
        if axes is None:
            if isinstance(fig, FuncAnimations):
                raise ValueError('composing animations without axes')
            axes = fig.gca()
        #
        if isinstance(maps, Analysis):
            if sampling is None:
                sampling = maps.parent.data
            maps = maps.data
        if isinstance(sampling, Analysis):
            sampling = sampling.data
        nsegments = self._eldest_parent.time.n_time_segments(sampling)
        #
        anim_kwargs = dict(blit=True, cache_frame_data=False, save_count=nsegments,
                repeat=False, interval=600)
        more_kwargs = dict(repeat_delay=None, fargs=None)
        more_kwargs.update(anim_kwargs)
        for kw in more_kwargs:
            try:
                arg = kwargs.pop(kw)
            except KeyError:
                pass
            else:
                if kw == 'interval' and arg in ('rt', 'realtime', 'real-time'):
                    arg = self._eldest_parent.time.window_shift * 1e3
                anim_kwargs[kw] = arg
        maps = maps[feature]
        #
        if logscale is True:
            logscale = self.default_log
        if logscale:
            if logscale in ('log', 'natural'):
                log = np.log
            elif logscale == 'log10':
                log = np.log10
            elif logscale:
                raise NotImplementedError(f"logscale='{logscale}'")
        #
        if maps.shape[1] == 2:
            if logscale:
                maps = maps.pow(2).sum(1).apply(log)*.5
            else:
                maps = maps.pow(2).sum(1).apply(np.sqrt)
        else:
            if logscale:
                maps = maps.apply(log)
            maps = maps[feature] # to Series
        assert isinstance(maps, pd.Series)
        clim = [maps.min(), maps.max()]
        map_kwargs = dict(clim=clim, aspect=aspect)
        self.clabel(feature, kwargs, logscale, map_kwargs)
        map_kwargs.update(kwargs)
        map_kwargs['overlay_locations'] = overlay_locations
        #
        _iter = self._eldest_parent.time.as_time_segments
        patches = self.plotter(axes, sampling.tessellation.spatial_mesh, sampling.bounding_box,
                **map_kwargs)
        #
        if isinstance(fig, FuncAnimations):
            animators = fig
            cls, fig = animators.add, animators._fig
        elif composable:
            animators = FuncAnimations(fig)
            cls = animators.add
        else:
            cls = animation.FuncAnimation
        #
        return cls(fig, patches.animate, init_func=patches.init_func,
                frames=_iter(sampling, maps, return_times=True), **anim_kwargs)

    def plot(self, maps, feature, sampling=None, axes=None, aspect='equal',
            interior_contour=None, overlay_locations=False, logscale=None,
            **kwargs):
        """
        Calls :func:`~tramway.helper.inference.map_plot`.

        May be reworked in the future to remove the :mod:`~tramway.helper` dependency.

        `logscale` applies only to the color-coded background of field maps.

        *new in 0.5.2*:
        argument *interior_contour* is a :class`dict` with the following keys allowed:
        *margin*, *linestyle*, *linewidth* and *color*.
        The default value for *margin* (or *relative_margin*) is `0.01`.

        `plot` does not access any analyzer attributes and can be safely
        called from any analyzer:

            from tramway.analyzer import RWAnalyzer

            RWAnalyzer().mapper.mpl.plot(...)

        or instanciating an `Mpl` object:

            from tramway.analyzer.mapper.mpl import Mpl

            Mpl().plot(...)

        """
        from tramway.helper.inference import map_plot
        if isinstance(maps, Analysis):
            if sampling is None:
                sampling = maps.get_parent()
            maps = maps.data
        if isinstance(sampling, Analysis):
            sampling = sampling.data
        if 'title' not in kwargs:
            kwargs['title'] = None
        if 'show' not in kwargs:
            kwargs['show'] = False
        if axes is not None:
            kwargs['axes'] = axes
        kwargs['aspect'] = aspect
        if overlay_locations is not False:
            if overlay_locations is True:
                overlay_locations = dict(color='r', alpha=.1)
            kwargs['point_style'] = overlay_locations
        #
        if logscale == 'natural':
            logscale = 'log'
        tr = kwargs.get('transform', None)
        if tr:
            if logscale:
                raise ValueError("both `logscale` and `transform` are defined")
            if tr in ('log', 'log10'):
                logscale = tr
            elif kwargs.get('unit', None) == 'std':
                self.logger.warning("units do not account for argument 'transform'")
        elif logscale:
            kwargs['transform'] = logscale if isinstance(logscale, str) else self.default_log
        self.clabel(feature, kwargs, logscale) # modifies `kwargs` inplace
        #
        map_plot(maps, sampling, feature=feature, **kwargs)

        # added in 0.5.2
        if interior_contour:
            from tramway.tessellation.utils2d import get_interior_contour
            if interior_contour is True:
                interior_contour = {}
            margin = interior_contour.pop('margin',
                     interior_contour.pop('relative_margin', .01))
            contour = get_interior_contour(sampling, relative_margin=margin)
            contour = np.vstack((contour, contour[[0]]))
            if axes is None:
                import matplotlib.pyplot as plt
                axes = plt
            axes.plot(contour[:,0], contour[:,1], 'r-', **interior_contour)


__all__ = ['PatchCollection', 'FuncAnimation', 'Mpl']

