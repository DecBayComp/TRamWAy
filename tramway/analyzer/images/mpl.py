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
import numpy as np
import pandas as pd


class Mpl(AnalyzerNode):
    """
    Matplotlib plotting utilities for 2D data; no time support.
    """
    __slots__ = ()
    def plot(self, x, *args, origin=None, frame=None, **kwargs):
        """
        Converts localization data from micrometers to pixels and plots them.

        Can overlay locations onto a frame image as plotted with `imshow` and
        default parameters (``origin='upper'``).
        """
        if isinstance(x, pd.DataFrame):
            x, y = x['x'], x['y']
        else:
            y = args[0]
            args = args[1:]
        try:
            axes = kwargs.pop('axes')
        except KeyError:
            import matplotlib.pyplot as axes
        img = self._parent
        if frame is None:
            height = img.height
        else:
            #axes.imshow(frame)
            height = frame.shape[0]
        x, y = np.array(x), np.array(y)
        if origin is not None:
            x -= origin[0]
            y -= origin[1]
        x /= img.pixel_size
        y /= img.pixel_size
        if origin is None and img.loc_offset is not None:
            x -= img.loc_offset[0]
            y -= img.loc_offset[1]
        return axes.plot(x, height-1-y, *args, **kwargs)

    def roi(self, frame, origin, *args, axes=None, colormap='gray', **kwargs):
        """
        Plots a frame corresponding to a region of interest.

        `origin` should be provided by :meth:`cropping_bounds`,
        if `frame` is given by :meth:`crop_frames`.

        Extra input arguments are passed to the :func:`imshow` function.

        This method works in combination with :meth:`spt_data.mpl.Mpl.plot`.
        """
        m, n = frame.shape
        px = self._parent.pixel_size
        x0, y0 = origin[0] - .5 * px, origin[1] - .5 * px
        x1, y1 = origin[0] + (m - .5) * px, origin[1] + (n - .5) * px
        try:
            colormap = kwargs.pop('cmap')
        except KeyError:
            pass
        for arg in ('origin', 'extent'):
            try:
                kwargs.pop(arg)
            except KeyError:
                pass
            else:
                self.logger.warning(f"argument '{arg}' ignored")
        if axes is None:
            import matplotlib.pyplot as axes
        return axes.imshow(frame, cmap=colormap, extent=[x0, x1, y0, y1],
                **kwargs)


__all__ = ['Mpl']

