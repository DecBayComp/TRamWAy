# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import AnalyzerNode, first
from ..artefact import Analysis
import warnings
from tramway.tessellation.base import Partition, Voronoi
from tramway.tessellation.time import TimeLattice


class Mpl(AnalyzerNode):
    """
    Matplotlib plotting utilities for 2D data; no time support.
    """
    __slots__ = ()
    def plot(self, tessellation, locations=None, axes=None,
            voronoi_options=dict(), delaunay_options=None,
            location_options=dict(), **kwargs):
        from tramway.helper.tessellation import cell_plot
        try:
            voronoi_options = dict(centroid_style=None, color='rrrr') | voronoi_options
            location_options = dict(alpha=.2, color='k') | location_options
            kwargs = dict(aspect='equal', show=False) | kwargs
        except TypeError: # Python < 3.9
            voronoi_options, _options = dict(centroid_style=None, color='rrrr'), voronoi_options
            voronoi_options.update(_options)
            location_options, _options = dict(alpha=.2, color='k'), location_options
            location_options.update(_options)
            kwargs, _kwargs = dict(aspect='equal', show=False), kwargs
            kwargs.update(_kwargs)
        if isinstance(tessellation, Analysis):
            tessellation = tessellation.data
        if isinstance(tessellation, Partition):
            sampling = tessellation
            tessellation = sampling.tessellation
            if locations is None:
                locations = sampling.points
            else:
                sampling = None
        else:
            sampling = None
        if isinstance(tessellation, TimeLattice):
            if tessellation.spatial_mesh is None:
                raise TypeError('no tessellation found; only time segments')
            tessellation = tessellation.spatial_mesh
        if isinstance(tessellation, Voronoi):
            if sampling is None:
                sampling = Partition(locations, tessellation)
        else:
            raise TypeError('tessellation type not supported: {}'.format(type(tessellation)))
        return cell_plot(sampling, axes=axes, voronoi=voronoi_options, locations=location_options,
                delaunay=delaunay_options, **kwargs)

    def animate(self, fig, sampling, axes=None,
            voronoi_options=dict(), location_options=dict(), **kwargs):
        """
        As this method is of limited interest, it has been poorly tested.
        """
        from matplotlib import animation
        from tramway.plot import mesh as tplt
        if axes is None:
            axes = fig.gca()
        #
        if isinstance(sampling, Analysis):
            sampling = sampling.data
        nsegments = self._eldest_parent.time.n_time_segments(sampling)
        # copied/pasted from mapper.mpl.Mpl.animate
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
        #
        _iter = self._eldest_parent.time.as_time_segments
        #
        try:
            voronoi_options = dict(centroid_style=None, color='rrrr') | voronoi_options
            location_options = dict(alpha=.2, color='k') | location_options
        except TypeError: # Python < 3.9
            voronoi_options, _options = dict(centroid_style=None, color='rrrr'), voronoi_options
            voronoi_options.update(_options)
            location_options, _options = dict(alpha=.2, color='k'), location_options
            location_options.update(_options)
        location_options['markersize'] = location_options.pop('size', 3) # for plot
        #location_options['s'] = location_options.pop('size', 8) # for scatter
        #
        first_segment = first(_iter(sampling, return_times=False))
        x, y = [ sampling.points[col].values for col in 'xy' ]
        glyphs, = axes.plot(x, y, '.', **location_options)
        tplt.plot_voronoi(first_segment, axes=axes, **voronoi_options)
        axes.set_aspect(kwargs.get('aspect', 'equal'))
        def init():
            return glyphs,
        def draw_segment(sampling):
            if isinstance(sampling, tuple):
                times, sampling = sampling
            else:
                times = None
            x, y = [ sampling.points[col].values for col in 'xy' ]
            glyphs.set_data(x, y)
            #xy = sampling.points[list('xy')].values
            #glyphs.set_array(xy)
            return glyphs,
        return animation.FuncAnimation(fig, draw_segment, init_func=init,
                frames=_iter(sampling, return_times=True), **anim_kwargs)


__all__ = ['Mpl']

