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
    def plot(self, x, *args, **kwargs):
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
        x = np.asarray(x) / img.pixel_size
        y = np.asarray(y) / img.pixel_size
        if img.loc_offset is not None:
            x -= img.loc_offset[0]
            y -= img.loc_offset[1]
        return axes.plot(x, img.height-y, *args, **kwargs)


__all__ = ['Mpl']

