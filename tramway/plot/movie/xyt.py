
import matplotlib
#matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

import os.path
from tramway.core.xyt import load_xyt
from tramway.core.analyses.base import Analyses
import numpy as np
import pandas as pd

try:
    input = raw_input # Py2
except NameError:
    pass


def animate_trajectories_2d(input_data, output_file,
        frame_rate=None, bit_rate=None, dots_per_inch=200, play=False,
        time_step=None, time_unit='s', color=None,
        line_width=1, marker_style='o', marker_size=4,
        figure=None, axes=None, bounding_box=None, axis=True,
        verbose=True, time_precision=2, columns=None, **kwargs):
    """
    Animate 2D trajectories.

    Arguments:

        input_data (str or pandas.DataFrame): path to xyt file or xyt data.

        output_file (str): path to .mp4 file.

        frame_rate (float): number of frames per second.

        bit_rate (int): movie bitrate; see also :class:`~matplotlib.animation.FFMpegWriter`.

        dots_per_inch (int): dpi argument passed to :met:`~matplotlib.animation.FFMpegWriter.saving`.

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

        columns (str or list): (comma-separated) list of column names if input data
            are to be loaded; see also :func:`~tramway.core.xyt.load_xyt`.

    Extra keyword arguments are passed to :class:`~matplotlib.animation.FFMpegWriter`.

    """

    if isinstance(input_data, pd.DataFrame):
        xyt = input_data
    elif isinstance(input_data, Analyses):
        xyt = input_data.locations
    else:
        input_file = os.path.expanduser(input_data)
        if not os.path.isfile(input_file):
            raise "file '{}' not found".format(input_file)
        load_kwargs = {}
        if columns is not None:
            if isinstance(columns, str):
                columns = columns.split(',')
            load_kwargs['columns'] = columns
        xyt = load_xyt(input_file, **load_kwargs)

    output_file = os.path.expanduser(output_file)
    if verbose and os.path.exists(output_file):
        answer = input("overwrite file '{}': [N/y] ".format(output_file))
        if not (answer and answer[0].lower() == 'y'):
            return

    if marker_style is None:
        base_style = '-'
    else:
        base_style = marker_style + '-'
    _style = {'LineWidth': line_width, 'MarkerSize': marker_size}

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
        _style['Color'] = _color(n)
        return _style

    trange = range
    if verbose:
        try:
            import tqdm
        except ImportError:
            pass
        else:
            trange = tqdm.trange

    if time_step is None:
        dt = xyt['t'].diff()[xyt['n'].diff()==0].quantile(.5)
        if verbose:
            print("selected time step: {}".format(dt))
    else:
        dt = time_step
    t0 = np.round(xyt['t'].min() / dt) * dt
    N = np.round((xyt['t'].max() - t0) / dt) + 1
    pending = xyt.copy()
    active = dict()

    if frame_rate is None:
        frame_rate = 1. / dt # assume dt is in seconds
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
        axes.set_title(title_pattern.format(t0))

    with grab.saving(figure, output_file, dots_per_inch):
        old_active = set()
        for f in trange(int(N)):
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
                    line, = axes.plot(x, y, base_style, **style(n))
                    active[n] = line
                else:
                    _x, _y = line.get_data()
                    line.set_data(np.r_[_x, x], np.r_[_y, y])
            for n in old_active - new_active:
                axes.lines.remove(active.pop(n))
            #
            if time_unit:
                axes.get_title().set_text(title_pattern.format(t))
            #
            grab.grab_frame()
            #
            old_active = new_active
            pending.drop(xt.index, inplace=True)
    if play:
        from tramway.plot.movie import Video
        movie = Video(output_file, fps=frame_rate)
        movie.play()

__all__ = ['animate_trajectories_2d']

