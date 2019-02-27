
import matplotlib
#matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

from tramway.core.analyses import *
from tramway.core.analyses.base import Analyses
from tramway.tessellation.base import CellStats
from tramway.tessellation.time import TimeLattice
from tramway.inference.base import Maps
from tramway.plot.map import *
from tramway.plot.movie import *

import os.path
import numpy as np
import pandas as pd

try:
    input = raw_input # Py2
except NameError:
    pass


def animate_map_2d_helper(input_data, output_file, label=None, variable=None, **kwargs):
    """
    Animate 2D maps.

    Arguments:

        input_data (str or tramway.core.analyses.base.Analyses): path to rwa file or analysis tree.

        output_file (str): path to .mp4 file.

        label (str or list): path to the maps as a (comma-separated) list of labels.

        variable (str): name of the mapped variable to be rendered.

    The other keyword arguments are passed to :class:`animate_map_2d`.
    """
    if isinstance(input_data, Analyses):
        analyses = input_data
    else:
        from tramway.core.hdf5 import load_rwa
        input_file = os.path.expanduser(input_data)
        if not os.path.isfile(input_file):
            raise "file '{}' not found".format(input_file)
        analyses = load_rwa(input_file, lazy=True)

    cells, maps = find_artefacts(analyses, (CellStats, Maps), label)

    if variable is None:
        if maps.variables[1:]:
            raise ValueError('multiple mapped variables found: {}'.format(maps.variables))
        variable = maps.variables[0]
    _map = maps[variable]

    animate_map_2d(_map, cells, output_file, **kwargs)


def animate_map_2d(_map, cells, output_file,
        frame_rate=1, bit_rate=None, dots_per_inch=200, play=False,
        time_step=None, time_unit='s', figure=None, axes=None,
        bounding_box=None, colormap=None, colorbar=True, axis=True,
        verbose=True, time_precision=2, **kwargs):
    """
    Animate 2D maps.

    Arguments:

        _map (pandas.DataFrame): scalar or vector map.

        cells (tramway.tessellation.base.CellStats): spatial-temporal binning.

        output_file (str): path to .mp4 file.

        frame_rate (float): number of frames per second.

        bit_rate (int): movie bitrate; see also :class:`~matplotlib.animation.FFMpegWriter`.

        dots_per_inch (int): dpi argument passed to :met:`~matplotlib.animation.FFMpegWriter.saving`.

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
    output_file = os.path.expanduser(output_file)
    if verbose and os.path.exists(output_file):
        answer = input("overwrite file '{}': [N/y] ".format(output_file))
        if not (answer and answer[0].lower() == 'y'):
            return

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
        cells = CellStats(tessellation=_mesh, location_count=np.ones(_mesh.number_of_cells))
    else:
        _map = [_map]
        t = analyses.xyt['t']
        tmin, tmax = t.min(), t.max()
        segments = [[tmin, tmax]]

    trange = range
    if verbose:
        try:
            import tqdm
        except ImportError:
            pass
        else:
            trange = tqdm.trange

    if time_step is None:
        N = len(segments) # len(_map)
    else:
        N = round((tmax - tmin) / time_step) + 1
        if frame_rate is None:
            frame_rate = 1. / time_step # assume dt is in seconds

    if 'fps' not in kwargs:
        kwargs['fps'] = frame_rate
    if 'bitrate' not in kwargs and bit_rate is not None:
        kwargs['bitrate'] = bit_rate
    grab = FFMpegWriter(**{kw: arg for kw, arg in kwargs.items() if arg is not None})

    if figure is None:
        if axes is None:
            import matplotlib.pyplot as plt
            figure, axes = plt.subplots()
        else:
            figure = axes.get_figure()
    elif axes is None:
        axes = figure.gca()
    if axis in (False, 'off'):
        axes.set_axis_off()

    if time_unit:
        title_pattern = "time = {{:.{:d}f}} {}".format(time_precision, time_unit)

    with grab.saving(figure, output_file, dots_per_inch):
        for f in trange(int(N)):
            if time_step is None:
                t = np.mean(segments[f])
                __map = _map[f]
            else:
                t = tmin + f * time_step

                segs = []
                weights = []
                for s, seg in enumerate(segments):
                    if seg[0] <= t and t <= seg[1]:
                        segs.append(s)
                        seg_center = (seg[0] + seg[1]) / 2.
                        w = np.abs(t - seg_center)
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

            _render(cells, __map, clim=clim, figure=figure, axes=axes,
                    colorbar=colorbar, colormap=colormap)
            if bounding_box is not None:
                axes.update_datalim_bounds(bounding_box)
            #
            if time_unit:
                axes.set_title(title_pattern.format(t))
            #
            grab.grab_frame()
            axes.clear()
            #
    if play:
        from tramway.plot.movie import Video
        movie = Video(output_file, fps=frame_rate)
        movie.play()


__all__ = ['animate_map_2d', 'animate_map_2d_helper']

