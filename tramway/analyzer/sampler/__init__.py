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
    __slots__ = ('_min_location_count',)
    def __init__(self, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._min_location_count = None
    @property
    def min_location_count(self):
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
    __slots__ = ()
    def sample(self, df, segmentation=None):
        return BaseSampler.sample(self, df, segmentation)

#Sampler.register(VoronoiSampler)


class SamplerInitializer(Initializer):
    __slots__ = ()
    def from_voronoi(self):
        self.specialize( VoronoiSampler )
    def from_spheres(self, radius):
        self.specialize( SphericalSampler, radius )
    def from_nearest_neighbors(self, knn):
        self.specialize( Knn, knn )
    def from_nearest_time_neighbors(self, knn):
        self.specialize( TimeKnn, knn )
    def sample(self, spt_data, segmentation=None):
        self.from_voronoi()
        return self._sampler.sample(spt_data, segmentation)
    @property
    def _sampler(self):
        return self._parent.sampler


class SphericalSampler(BaseSampler):
    __slots__ = ('_radius',)
    def __init__(self, radius, **kwargs):
        BaseSampler.__init__(self, **kwargs)
        self._radius = radius
    @property
    def radius(self):
        return self._radius
    @radius.setter
    def radius(self, r):
        self._radius = r
    def sample(self, spt_data, segmentation=None):
        return BaseSampler.sample(self, spt_data, segmentation, radius=self.radius)


class Knn(BaseSampler):
    __slots__ = ('_knn',)
    def __init__(self, knn, **kwargs):
        BaseSampler.__init__(self, **kwargs)
        self._knn = knn
    @property
    def knn(self):
        return self._knn
    @knn.setter
    def knn(self, r):
        self._knn = r
    def sample(self, spt_data, segmentation=None):
        return BaseSampler.sample(self, spt_data, segmentation, knn=self.knn)


class TimeKnn(BaseSampler):
    __slots__ = ('_knn',)
    def __init__(self, knn, **kwargs):
        BaseSampler.__init__(self, **kwargs)
        self._knn = knn
    @property
    def knn(self):
        return self._knn
    @knn.setter
    def knn(self, r):
        self._knn = r
    def sample(self, spt_data, segmentation=None):
        return BaseSampler.sample(self, spt_data, segmentation, time_knn=self.knn)

