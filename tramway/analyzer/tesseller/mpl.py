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
from tramway.tessellation.base import Partition, Voronoi
from tramway.tessellation.time import TimeLattice


class Mpl(AnalyzerNode):
    """
    Matplotlib plotting utilities for 2D data; no time support.
    """
    __slots__ = ()
    def plot(self, tessellation, locations=None, axes=None,
            voronoi_options=dict(), delaunay_options=None,
            location_options=dict(), **kwargs):
        from tramway.helper.tessellation import cell_plot
        try:
            voronoi_options = dict(centroid_style=None, color='rrrr') | voronoi_options
            location_options = dict(alpha=.2, color='k') | location_options
            kwargs = dict(aspect='equal', show=False) | kwargs
        except TypeError: # Python < 3.9
            voronoi_options, _options = dict(centroid_style=None, color='rrrr'), voronoi_options
            voronoi_options.update(_options)
            location_options, _options = dict(alpha=.2, color='k'), location_options
            location_options.update(_options)
            kwargs, _kwargs = dict(aspect='equal', show=False), kwargs
            kwargs.update(_kwargs)
        if isinstance(tessellation, Analysis):
            tessellation = tessellation.data
        if isinstance(tessellation, Partition):
            sampling = tessellation
            tessellation = sampling.tessellation
            if locations is None:
                locations = sampling.points
            else:
                sampling = None
        else:
            sampling = None
        if isinstance(tessellation, TimeLattice):
            if tessellation.spatial_mesh is None:
                raise TypeError('no tessellation found; only time segments')
            tessellation = tessellation.spatial_mesh
        if isinstance(tessellation, Voronoi):
            if sampling is None:
                sampling = Partition(locations, tessellation)
        else:
            raise TypeError('tessellation type not supported: {}'.format(type(tessellation)))
        cell_plot(sampling, axes=axes, voronoi=voronoi_options, locations=location_options,
                delaunay=delaunay_options, **kwargs)


__all__ = ['Mpl']

