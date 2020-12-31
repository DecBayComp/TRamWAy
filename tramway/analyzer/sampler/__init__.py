# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import *
from ..artefact import analysis
from .abc import *
from tramway.tessellation import Partition


class BaseSampler(AnalyzerNode):
    """ wraps the `Partition.cell_index` logics that readily implements
    all the possibilities mentioned in this module.
    """
    __slots__ = ('_min_location_count',)
    def __init__(self, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._min_location_count = 0
    @property
    def min_location_count(self):
        """
        *int*: Minimum number of assigned (trans-)locations, as given
            by the default Voronoi sampling;
            any microdomain with too few assigned (trans-)locations
            are marked as disabled and are excluded from further analyses
        """
        return self._min_location_count
    @min_location_count.setter
    def min_location_count(self, n):
        self._min_location_count = n
    @analysis(2, 'segmentation')
    def sample(self, spt_dataframe, segmentation=None, **kwargs):
        df = spt_dataframe
        if self.min_location_count is not None:
            kwargs['min_location_count'] = self.min_location_count
        if segmentation is None:
            if self.tesseller.initialized:
                segmentation = self.tesseller.tessellate(spt_dataframe)
            if self.time.initialized:
                segmentation = self.time.segment(spt_dataframe, segmentation)
        if segmentation is None:
            cell_index = None
        else:
            cell_index = segmentation.cell_index(df, **kwargs)
        sample = Partition(df, segmentation, cell_index)
        try:
            self.tesseller.bc_update_params(sample.param)
        except AttributeError:
            pass
        if kwargs:
            sample.param['partition'] = kwargs
        return sample
    @property
    def tesseller(self):
        return self._parent.tesseller
    @property
    def time(self):
        return self._parent.time

Sampler.register(BaseSampler)

class VoronoiSampler(BaseSampler):
    """ nearest neighbor assignment.
    
    Each data point is assigned to exactly one cell/microdomain."""
    __slots__ = ()
    def sample(self, df, segmentation=None):
        return BaseSampler.sample(self, df, segmentation)

#Sampler.register(VoronoiSampler)


class SamplerInitializer(Initializer):
    """ Initializer class for the :class:`~tramway.analyzer.RWAnalyzer`
    :attr:`~tramway.analyzer.RWAnalyzer.sampler` main attribute.

    The :attr:`~tramway.analyzer.RWAnalyzer.sampler` attribute self-modifies
    on calling any of the *from_...* methods.

    The default not-initialized :attr:`~tramway.analyzer.RWAnalyzer.sampler`
    attribute behaves like:

    .. code-block:: python

        a = RWAnalyzer()
        a.sampler.from_voronoi()

    """
    __slots__ = ()
    def from_voronoi(self):
        """ default sampler.

        The data points are assigned to the nearest cell/microdomain.
        
        See also :class:`VoronoiSampler`."""
        self.specialize( VoronoiSampler )
    def from_spheres(self, radius):
        """ nearest neighbor assignment by distance.

        Voronoi assignment is applied first and then, for each microdomain,
        the maximum distance of the assigned points to the center of the cell is considered.
        The `radius` argument defines a lower and/or upper bound(s) on the distance
        from the cell center.
        If the most distant assigned point is out of the specified bounds,
        additional points are selected (or assigned points are discarded)
        in order of distance from the corresponding cell center.
        
        See also :class:`SphericalSampler`."""
        self.specialize( SphericalSampler, radius )
    def from_nearest_neighbors(self, knn):
        """ nearest neighbor assignment by count.

        The Voronoi assignment is applied first.
        The `knn` argument defines a lower and/or upper bound(s) on the number of neighbors.
        For each microdomain, if the number that results from Voronoi assignment is
        out of the specified bounds, additional points are selected (or assigned points are
        discarded) in order of distance from the corresponding cell center.
        
        See also :class:`Knn`."""
        self.specialize( Knn, knn )
    def from_nearest_time_neighbors(self, knn):
        """ similar to `from_nearest_neighbors` but cell/microdomain size is adjusted in time
        instead of space.
        
        See also :class:`TimeKnn`."""
        self.specialize( TimeKnn, knn )
    def sample(self, spt_data, segmentation=None):
        """ main processing method. """
        self.from_voronoi()
        return self._sampler.sample(spt_data, segmentation)
    @property
    def _sampler(self):
        return self._parent.sampler


class SphericalSampler(BaseSampler):
    """ nearest neighbor selection by distance."""
    __slots__ = ('_radius',)
    def __init__(self, radius, **kwargs):
        BaseSampler.__init__(self, **kwargs)
        self._radius = radius
    @property
    def radius(self):
        """
        *float* or *(float, float)* or *callable*:
            Lower bound or (lower bound, upper bound) pair on the distance
            from a cell/microdomain center;
            alternatively, a function can define these bounds for each
            microdomain (takes the microdomain index as input argument)
        """
        return self._radius
    @radius.setter
    def radius(self, r):
        self._radius = r
    def sample(self, spt_data, segmentation=None):
        return BaseSampler.sample(self, spt_data, segmentation, radius=self.radius)


class Knn(BaseSampler):
    """ nearest neighbor selection by count."""
    __slots__ = ('_knn',)
    def __init__(self, knn, **kwargs):
        BaseSampler.__init__(self, **kwargs)
        self._knn = knn
    @property
    def knn(self):
        """ 
        *int* or *(int, int)* or *callable*:
            Lower bound or (lower bound, upper bound) pair on the number of data points
            which, if Voronoi assignment yields a number out of these bounds,
            triggers a procedure that picks extra points (or discards assigned points)
            wrt distance from the Voronoi cell center.
            This applies independently for each microdomain with insufficient or too
            many points.
            Alternatively, bounds can be defined on a per-microdomain basis
            with a function that takes the microdomain index as an `int` and
            returns either an `int` or a pair of `int` or :const:`None`
        """
        return self._knn
    @knn.setter
    def knn(self, r):
        self._knn = r
    def sample(self, spt_data, segmentation=None):
        return BaseSampler.sample(self, spt_data, segmentation, knn=self.knn)


class TimeKnn(BaseSampler):
    """ nearest neighbor selection by count across time."""
    __slots__ = ('_knn',)
    def __init__(self, knn, **kwargs):
        BaseSampler.__init__(self, **kwargs)
        self._knn = knn
    @property
    def knn(self):
        """
        *int* or *(int, int)* or *callable*:
            Lower bound or (lower bound, upper bound) pair on the number of data points
            which, if Voronoi assignment yields a number out of these bounds,
            triggers the shrinking or widening of the time window,
            centered on the original middle point of the time segment.
            This applies independently for each microdomain with insufficient or too
            many points.
            microdomain with insufficient or too many points.
            Alternatively, bounds can be defined on a per-microdomain basis
            with a function that takes the microdomain index as an `int` and
            returns either an `int` or a pair of `int` or :const:`None`
        """
        return self._knn
    @knn.setter
    def knn(self, r):
        self._knn = r
    def sample(self, spt_data, segmentation=None):
        return BaseSampler.sample(self, spt_data, segmentation, time_knn=self.knn)


__all__ = ['Sampler', 'SamplerInitializer', 'BaseSampler', 'VoronoiSampler',
        'SphericalSampler', 'Knn', 'TimeKnn']

