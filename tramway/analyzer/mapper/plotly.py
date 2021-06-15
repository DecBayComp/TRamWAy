# -*- coding: utf-8 -*-

# Copyright © 2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import AnalyzerNode, single
from ..artefact import Analysis
import numpy as np
import pandas as pd
import plotly # missing dependencies: plotly nbformat
import plotly.graph_objects as go

"""
Plotly interfaces currently are highly experimental
and may suffer from numerous bugs.

The `plotly` dependencies are not enforced and an
:class:`ImportError` is raised if the library cannot
be found.
"""

class Plotly(AnalyzerNode):
    """
    Plotly interface for maps.

    It does not access other attributes of the :class:`RWAnalyzer`,
    and thus can be safely used from any :class:`RWAnalyzer` object:

        from tramway.analyzer import RWAnalyzer

        RWAnalyzer().mapper.plotly.plot_surface(...)

    or:

        from tramway.analyzer.mapper.plotly import Plotly

        Plotly.plot_surface(...)

    """

    def plot_surface(self, maps, feature, sampling=None, fig=None,
            row=None, col=None, colormap='viridis', title=None,
            colorbar=None, resolution=200, **kwargs):
        """
        Plot a 2D map as a colored 3D surface.
        """
        surface_kwargs = kwargs
        surface_kwargs['colorscale'] = kwargs.pop('colorscale', colormap)

        figure_kwargs = {}
        for kw in ('scene',):
            try:
                arg = kwargs.pop(kw)
            except KeyError:
                pass
            else:
                figure_kwargs[kw] = arg

        if fig is not None:
            if not (row is None and col is None):
                # subplots
                figure_kwargs.update(dict(
                    row=1 if row is None else row+1,
                    col=1 if col is None else col+1,
                    ))

        if colorbar is not None:
            surface_kwargs['colorbar'] = colorbar

        if isinstance(maps, Analysis):
            if sampling is None:
                sampling = maps.get_parent()
            maps = maps.data
        if isinstance(sampling, Analysis):
            sampling = sampling.data

        tessellation = sampling.tessellation
        try:
            tessellation = tessellation.spatial_mesh
        except AttributeError:
            pass

        xlim, ylim = sampling.bounding_box[['x', 'y']].values.T
        step = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / resolution
        x = np.arange(xlim[0], xlim[1], step)
        y = np.arange(ylim[0], ylim[1], step)
        x, y = np.meshgrid(x, y)

        pts = np.stack((x.ravel(), y.ravel()), axis=1)
        partition_kwargs = dict(sampling.param.get('partition', {}))
        for kw in list(partition_kwargs.keys()):
            if kw.startswith('time_'):
                del partition_kwargs[kw]
            elif kw in ('min_location_count', 'knn'):
                del partition_kwargs[kw]
        cell_ix = tessellation.cell_index(
                pd.DataFrame(pts, columns=['x', 'y']),
                **partition_kwargs)

        map_ = maps[feature]
        if 1<map_.shape[1]:
            map_ = map_.pow(2).sum(1).apply(np.sqrt)
        else:
            map_ = map_[feature]

        z = np.zeros(len(pts), dtype=map_.dtype)

        if isinstance(cell_ix, tuple):
            pts_, cells = cell_ix
            assert np.all(np.unique(pts_) == np.arange(len(pts)))
            # brute force accum array
            n = np.zeros(z.shape, dtype=int)
            for i in map_.index:
                k = pts_[cells==i]
                if k.size:
                    z[k] += map_[i]
                    n[k] += 1
            ok = 0<n
            z[1<n] /= n[1<n]
        else:
            ok = 0<=cell_ix
            map__ = np.full(tessellation.number_of_cells,
                    np.nan, dtype=map_.dtype)
            map__[map_.index] = map_.values
            z[ok] = map__[cell_ix[ok]]
        z[~ok] = np.nan

        z = z.reshape(x.shape)
        surface = go.Surface(x=x, y=y, z=z, **surface_kwargs)

        if fig is None:
            fig = go.Figure(data=[surface], **figure_kwargs)
        else:
            fig.add_trace(surface, **figure_kwargs)

__all__ = [ 'Plotly' ]

