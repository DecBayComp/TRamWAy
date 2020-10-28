# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.tessellation.base import Partition
from tramway.tessellation.time import TimeLattice
from tramway.plot.map import *
from tramway.plot.animation import *
import numpy as np


def animate_map_2d(_map, cells, output_file=None,
        frame_rate=1, bit_rate=None, dots_per_inch=200, play=False,
        time_step=None, time_unit='s', figure=None, axes=None,
        bounding_box=None, colormap=None, colorbar=True, axis=True,
        verbose=True, time_precision=2, **kwargs):
    """
    Animate 2D maps.

    Arguments:

        _map (pandas.DataFrame): scalar or vector map.

        cells (tramway.tessellation.base.Partition): spatial-temporal binning.

        output_file (str): path to .mp4 file.

        frame_rate (float): number of frames per second.

        bit_rate (int): movie bitrate; see also :class:`~matplotlib.animation.FFMpegWriter`.

        dots_per_inch (int): dpi argument passed to :meth:`~matplotlib.animation.FFMpegWriter.saving`.

        play (bool): play the movie once generated.

        time_step (float): time step between successive frames.

        time_unit (str): time unit for time display; if None, no title is displayed.

        figure (matplotlib.figure.Figure): figure handle.

        axes (matplotlib.axes.Axes): axes handle.

        bounding_box (matplotlib.transforms.Bbox): axis bounding box.

        colormap (str): colormap name; see also https://matplotlib.org/users/colormaps.html.

        colorbar (bool or str or dict): see also :func:`~tramway.plot.map.scalar_map_2d`.

        axis (bool or str): make the axes visible.

        verbose (bool): ask for confirmation if the output file is to be overwritten
            and draw a progress bar (module :mod:`tqdm` must be available).

        time_precision (int): number of decimals for time display.

    Extra keyword arguments are passed to :class:`~matplotlib.animation.FFMpegWriter`.

    """

    dim = _map.shape[1]
    if dim == 1:
        clim = [_map.values.min(), _map.values.max()]
        _render = scalar_map_2d
    elif dim == 2:
        _amplitude = _map.pow(2).sum(1).apply(np.sqrt)
        clim = [_amplitude.values.min(), _amplitude.values.max()]
        _render = field_map_2d
    else:
        raise ValueError('nD data not supported for n not 1 or 2')

    if isinstance(cells.tessellation, TimeLattice) \
            and cells.tessellation.spatial_mesh is not None:
        segments = cells.tessellation.time_lattice
        tmin, tmax = segments.min(), segments.max()
        _map = cells.tessellation.split_frames(_map)
        _mesh = cells.tessellation.spatial_mesh
        cells = Partition(tessellation=_mesh, location_count=np.ones(_mesh.number_of_cells), points=cells.points)
    else:
        _map = [_map]
        t = cells.locations['t']
        tmin, tmax = t.min(), t.max()
        segments = [[tmin, tmax]]

    if time_step is None:
        N = len(segments) # len(_map)
    else:
        N = round((tmax - tmin) / time_step) + 1
        if frame_rate is None:
            frame_rate = 1. / time_step # assume dt is in seconds

    if time_unit:
        title_pattern = "time = {{:.{:d}f}} {}".format(time_precision, time_unit)

    try:
        with VideoWriterReader(output_file, frame_rate, bit_rate, dots_per_inch,
                figure, axes, axis, verbose, **kwargs) as movie:

            with movie.saving():
                for f in movie.range(N):
                    if time_step is None:
                        t = np.mean(segments[f])
                        __map = _map[f]
                    else:
                        t = tmin + f * time_step

                        seg_scale = np.median([ _s[1] - _s[0] for _s in segment ]) * .5
                        segs = []
                        weights = []
                        for s, seg in enumerate(segments):
                            if seg[0] <= t and t <= seg[1]:
                                segs.append(s)
                                seg_center = (seg[0] + seg[1]) / 2.
                                w = 1. - np.abs(t - seg_center) / seg_scale
                                weights.append(w)
                        assert bool(segs)

                        weights = np.array(weights)
                        if np.all(weights == 0):
                            weights[...] = 1. / numel(weights)
                        else:
                            weights /= np.sum(weights)
                        weights = list(weights)

                        __map = _map[segs.pop()] * weights.pop()
                        for s, w in zip(segs, weights):
                            if w == 0:
                                continue
                            __map = __map + _map[s] * w

                    _render(cells, __map, clim=clim, figure=movie.figure, axes=movie.axes,
                            colorbar=colorbar and f==0, colormap=colormap)
                    if bounding_box is not None:
                        movie.axes.update_datalim_bounds(bounding_box)
                    #
                    if time_unit:
                        movie.axes.set_title(title_pattern.format(t))
                    #
                    movie.grab_frame()
                    movie.axes.clear()

            if play:
                movie.play()

    except Aborted:
        pass


__all__ = ['animate_map_2d']

