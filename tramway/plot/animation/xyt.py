# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.plot.animation import *
import numpy as np


def animate_trajectories_2d(xyt, output_file=None,
        frame_rate=None, bit_rate=None, dots_per_inch=200, play=False,
        time_step=None, time_unit='s', color=None,
        line_width=1, marker_style='o', marker_size=4,
        figure=None, axes=None, bounding_box=None, axis=True,
        verbose=True, time_precision=2, **kwargs):
    """
    Animate 2D trajectories.

    Arguments:

        xyt (pandas.DataFrame): nxyt data.

        output_file (str): path to .mp4 file.

        frame_rate (float): number of frames per second.

        bit_rate (int): movie bitrate; see also :class:`~matplotlib.animation.FFMpegWriter`.

        dots_per_inch (int): dpi argument passed to :meth:`~matplotlib.animation.FFMpegWriter.saving`.

        play (bool): play the movie once generated.

        time_step (float): time step between successive frames.

        time_unit (str): time unit for time display; if None, no title is displayed.

        color (list or numpy.ndarray or callable): trajectory colour.

        line_width (float): translocation line width.

        marker_style (str): location marker style.

        marker_size (float): location marker size.

        figure (matplotlib.figure.Figure): figure handle.

        axes (matplotlib.axes.Axes): axes handle.

        bounding_box (matplotlib.transforms.Bbox): axis bounding box.

        axis (bool or str): make the axes visible.

        verbose (bool): ask for confirmation if the output file is to be overwritten
            and draw a progress bar (module :mod:`tqdm` must be available).

        time_precision (int): number of decimals for time display.

    Extra keyword arguments are passed to :class:`~matplotlib.animation.FFMpegWriter`.

    """

    if marker_style is None:
        base_style = '-'
    else:
        base_style = marker_style + '-'
    _style = dict(linewidth=line_width, markersize=marker_size)

    if color is None:
        def _color(n):
            while True:
                _c = np.random.rand(3)
                if np.any(.1 <= _c): # not too light (background is white)
                    break
            return _c
    elif callable(color):
        _color = color
    else:
        ncolors = len(color)
        def _color(n):
            return color[np.mod(n, ncolors)]
    def style(n):
        _style['color'] = _color(n)
        return _style

    if time_unit:
        title_pattern = "time = {{:.{:d}f}} {}".format(time_precision, time_unit)

    if time_step is None:
        dt = xyt['t'].diff()[xyt['n'].diff()==0].quantile(.5)
        if verbose:
            print("selected time step: {}".format(dt))
    else:
        dt = time_step
    if frame_rate is None:
        frame_rate = 1. / dt # assume dt is in seconds
    if bounding_box is None:
        import matplotlib.transforms
        xmin, ymin = xyt['x'].min(), xyt['y'].min()
        xmax, ymax = xyt['x'].max(), xyt['y'].max()
        bounding_box = matplotlib.transforms.Bbox.from_bounds(xmin, ymin, xmax-xmin, ymax-ymin)

    t0 = np.round(xyt['t'].min() / dt) * dt
    N = np.round((xyt['t'].max() - t0) / dt) + 1

    pending = xyt.copy()
    active = dict()
    old_active = set()

    try:
        with VideoWriterReader(output_file, frame_rate, bit_rate, dots_per_inch,
                figure, axes, axis, verbose, **kwargs) as movie:

            with movie.saving():
                for f in movie.range(N):
                    t = t0 + f * dt
                    xt = pending[np.abs(pending['t'] - t) < .5 * dt]
                    new_active = set()
                    for i, row in xt.iterrows():
                        n = row['n']
                        x, y = row['x'], row['y']
                        new_active.add(n)
                        try:
                            line = active[n]
                        except KeyError:
                            line, = movie.axes.plot(x, y, base_style, **style(n))
                            active[n] = line
                        else:
                            _x, _y = line.get_data()
                            line.set_data(np.r_[_x, x], np.r_[_y, y])
                    for n in old_active - new_active:
                        movie.axes.lines.remove(active.pop(n))
                    #
                    if time_unit:
                        movie.axes.set_title(title_pattern.format(t))
                    if bounding_box is not None:
                        movie.axes.dataLim.set(bounding_box)
                    #
                    movie.grab_frame()
                    #
                    old_active = new_active
                    pending.drop(xt.index, inplace=True)

            if play:
                movie.play()

    except Aborted:
        pass


__all__ = ['animate_trajectories_2d']

