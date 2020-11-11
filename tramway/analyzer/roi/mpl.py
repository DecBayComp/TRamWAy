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
from tramway.core.xyt import translocations_to_trajectories


class Mpl(AnalyzerNode):
    """
    See also :class:`~tramway.analyzer.spt_data.mpl.Mpl`.
    """
    __slots__ = ()
    def animate(self, fig, trajs=None, axes=None, xlim='auto', ylim=None, aspect='equal', **kwargs):
        """
        See also :meth:`~tramway.analyzer.spt_data.mpl.Mpl.animate`.
        """
        roi = self._parent
        if trajs is None:
            trajs = translocations_to_trajectories(roi.crop())
        if xlim=='auto' and ylim is None:
            _min, _max = roi.bounding_box
            xlim, ylim = (_min[0],_max[0]), (_min[1],_max[1])
        return roi._spt_data.mpl.animate(fig, trajs, axes, xlim, ylim, aspect, **kwargs)


__all__ = ['Mpl']

