# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
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
    __slots__ = ('glyphs','bin_indices')
    def __init__(self, ax, mesh, bounding_box=None, **kwargs):
        """
        Arguments:
        
            ax (Axes): matplotlib axes.
            
            mesh (Voronoi or Partition): space bins.

            bounding_box (DataFrame): map bounding box;
                required if *mesh* is not `Partition`.
            
        Extra keyword arguments are passed to `scalar_map_2d`.
        """
        if not isinstance(mesh, Partition):
            mesh = Partition(pd.DataFrame([], columns=list('nxyt')),
                    mesh,
                    location_count=np.full(mesh.number_of_cells, np.iinfo(np.int).max, dtype=np.int),
                    bounding_box=bounding_box)
        self.glyphs, self.bin_indices = scalar_map_2d(
                mesh, pd.Series(np.zeros(mesh.number_of_cells, dtype=np.float)),
                axes=ax, return_patches=True, **kwargs)
    def plot(self, map):
        """
        Updates the color of the patches.
        
        Arguments:
        
            map (Series): parameter values.
            
        Returns:
        
            tuple: sequence of updated glyphs.
        """
        values = np.full(self.bin_indices.max()+1, np.nan, dtype=np.float)
        values[map.index] = map.values
        self.glyphs.set_array(values[self.bin_indices])
        return (self.glyphs,)
    def init_func(self):
        """
        To be passed as argument `init_func` to `FuncAnimation`.
        """
        return (self.glyphs,)
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
            map = map[-1]
        return self.plot(map)


class Mpl(AnalyzerNode):
    """  """
    __slots__ = ()
    @property
    def plotter(self):
        return PatchCollection
    def animate(self, fig, maps, feature, sampling=None, axes=None, aspect='equal', **kwargs):
        """
        Animates the time-segmented inference parameters.

        Vector features are represented as amplitude,
        and especially force as log. amplitude.

        Arguments:

            fig (matplotlib.figure.Figure): figure.

            maps (~tramway.analyzer.Analysis or Maps): map series.

            feature (str): parameter to be drawn.

            sampling (~tramway.analyzer.Analysis or Partition): spatial bins and time segments;
                optional only if *maps* is an :class:`~tramway.analyzer.Analysis`.

            axes (matplotlib.axes.Axes): figure axes.

            aspect (str or None): aspect ratio.

        Returns:

            matplotlib.animation.FuncAnimation: animation object.

        Extra input arguments are passed to :class:`~matplotlib.animation.FuncAnimation`
        or :class:`PatchCollection` (and `scalar_map_2d`).
        """
        if axes is None:
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
        #
        maps = maps[feature]
        if maps.shape[1] == 2:
            if feature == 'force':
                maps = maps.pow(2).sum(1).apply(np.log)*.5
            else:
                maps = maps.pow(2).sum(1).apply(np.sqrt)
        else:
            maps = maps[feature] # to Series
        assert isinstance(maps, pd.Series)
        clim = [maps.min(), maps.max()]
        map_kwargs = dict(clim=clim, aspect=aspect)
        try:
            unit = kwargs.pop('unit')
        except KeyError:
            pass
        else:
            if unit == 'std':
                unit = dict(
                        diffusivity='$\mu\\rm{m}^2\\rm{s}^{-1}$',
                        potential='$k_{\\rm{B}}T$',
                        force='Log. amplitude',
                        drift='$\mu\\rm{m}\\rm{s}^{-1}$',
                    ).get(feature, None)
            if unit is not None:
                map_kwargs['unit'] = unit
        map_kwargs.update(kwargs)
        #
        _iter = self._eldest_parent.time.as_time_segments
        patches = self.plotter(axes, sampling.tessellation.spatial_mesh, sampling.bounding_box,
                **map_kwargs)
        return animation.FuncAnimation(fig, patches.animate, init_func=patches.init_func,
                frames=_iter(sampling, maps, return_times=False), **anim_kwargs)


__all__ = ['PatchCollection', 'Mpl']

