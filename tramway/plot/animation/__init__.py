# -*- coding: utf-8 -*-

# Copyright © 2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

"""
This package is no longer maintained and is superseded by the *mpl*
functionalities of :class:`~tramway.analyzer.RWAnalyzer`.

See for example :meth:`~tramway.analyzer.images._RawImage.to_color_movie`,
:meth:`~tramway.analyzer.spt_data.mpl.Mpl.animate` and
:meth:`~tramway.analyzer.mapper.mpl.Mpl.animate`.
"""

try:
    input = raw_input # Py2
except NameError:
    pass

import os
#import matplotlib
#matplotlib.use('Agg')
from matplotlib.animation import FFMpegWriter

try:
    import cv2
except:
    try: # Windows
        os.startfile
    except:
        class VideoReader(object):
            def __init__(self, *args, **kwargs):
                import cv2 # cannot read the movie
    else:
        class VideoReader(object):
            def __init__(self, filepath, *args, **kwargs):
                self.path = filepath
            def play(self, **kwargs):
                os.startfile(self.path)
else:
    # mostly borrowed from:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#playing-video-from-file
    class VideoReader(object):
        def __init__(self, filepath, *args, **kwargs):
            self.path = filepath
            self.cap = cv2.VideoCapture(self.path)
        def play(self, fps=20, **kwargs):
            frame_duration = int(1000./float(fps))
            if not self.cap.isOpened():
                self.cap.open(self.path)
                if not self.cap.isOpened():
                    raise RuntimeError('cannot open the movie')
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok: break
                cv2.imshow('frame', frame)
                # wait and listen key events
                key = cv2.waitKey(frame_duration)
                if key & 0xFF == ord('q'): # quit on 'q'
                    break
                elif key == 27: # quit on ESC
                    break
                # check that window is still opened
                if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
                    # in old opencv-python versions, use cv2.WND_PROP_AUTOSIZE instead
                    break
            self.cap.release()
            cv2.destroyAllWindows()


class Aborted(Exception):
    pass


class VideoWriter(object):
    def __init__(self, frame_rate, bit_rate=None, dots_per_inch=200, figure=None, axes=None,
            axis=True, verbose=True, **kwargs):
        self.fps = kwargs.get('fps', None)
        if self.fps is None:
            self.fps = frame_rate
        kwargs['fps'] = self.fps
        if 'bitrate' not in kwargs:
            kwargs['bitrate'] = bit_rate
        self.grab = FFMpegWriter(**{kw: arg for kw, arg in kwargs.items() if arg is not None})

        self.finally_close = False
        if figure is None:
            if axes is None:
                # always import pyplot as late as possible to allow for backend selection
                import matplotlib.pyplot as plt
                figure, axes = plt.subplots()
                figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
                self.finally_close = True
            else:
                figure = axes.get_figure()
        elif axes is None:
            axes = figure.gca()
        if axis in (False, 'off'):
            axes.set_axis_off()
        self.dpi = dots_per_inch
        self.figure = figure
        self.axes = axes
        self.verbose = verbose

    def saving(self, output_file):
        output_file = os.path.expanduser(output_file)
        if self.verbose and os.path.exists(output_file) and 0 < os.path.getsize(output_file):
            answer = input("overwrite file '{}': [N/y] ".format(output_file))
            if not (answer and answer[0].lower() == 'y'):
                raise Aborted
        return self.grab.saving(self.figure, output_file, self.dpi)

    def range(self, N):
        trange = range
        if self.verbose:
            try:
                import tqdm
            except ImportError:
                pass
            else:
                trange = tqdm.trange
        return trange(int(N))

    def grab_frame(self, *args, **kwargs):
        self.grab.grab_frame(*args, **kwargs)


class VideoWriterReader(object):
    def __init__(self, filepath, *args, **kwargs):
        self.path = filepath
        self.unlink_on_exit = False
        self.writer = VideoWriter(*args, **kwargs)

    @property
    def fps(self):
        return self.writer.fps
    @fps.setter
    def fps(self, f):
        self.writer.fps = f

    @property
    def figure(self):
        return self.writer.figure
    @figure.setter
    def figure(self, f):
        self.writer.figure = f

    @property
    def axes(self):
        return self.writer.axes
    @axes.setter
    def axes(self, a):
        self.writer.axes = a

    @property
    def verbose(self):
        return self.writer.verbose
    @verbose.setter
    def verbose(self, v):
        self.writer.verbose = v

    def __enter__(self):
        if self.path is None:
            import tempfile
            fd, self.path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
            self.unlink_on_exit = True
        return self

    def __exit__(self, *args):
        if self.writer.finally_close:
            # ideally this should happen in VideoWriter
            import matplotlib.pyplot as plt
            plt.close(self.figure)
        if self.unlink_on_exit:
            os.unlink(self.path)

    def saving(self):
        return self.writer.saving(self.path)

    def play(self, fps=None, **kwargs):
        VideoReader(self.path).play(fps=self.fps if fps is None else fps)

    def range(self, N):
        return self.writer.range(N)

    def grab_frame(self, *args, **kwargs):
        return self.writer.grab_frame(*args, **kwargs)


__all__ = [ 'VideoReader', 'VideoWriter', 'VideoWriterReader', 'Aborted' ]

