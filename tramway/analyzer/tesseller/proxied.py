# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .proxy import *


from tramway.tessellation.grid import RegularMesh

class Squares(TessellerProxy):
    """
    Wraps :class:`tramway.tessellation.grid.RegularMesh`.
    """
    __slots__ = ()
    def __init__(self, **kwargs):
        TessellerProxy.__init__(self, RegularMesh, **kwargs)
        self.alg_name = 'grid'
    def _reset_kwargs(self):
        TessellerProxy._reset_kwargs(self)
        self._init_kwargs.update(dict(
                min_probability = None,
                avg_probability = None,
                max_probability = None,
                min_distance = None,
                avg_distance = None,
                avg_location_count = None,
                ))

    #scaler          = proxy_property('scaler', 'attr')
    lower_bound     = proxy_property('lower_bound', 'attr')
    upper_bound     = proxy_property('upper_bound', 'attr')
    count_per_dim   = proxy_property('count_per_dim',   'attr')
    min_probability = proxy_property('min_probability', 'attr')
    avg_probability = proxy_property('avg_probability', 'attr')
    max_probability = proxy_property('max_probability', 'attr')
    min_distance    = proxy_property('min_distance',    'attr')
    avg_distance    = proxy_property('avg_distance',    'attr')


from tramway.tessellation.hexagon import HexagonalMesh

class Hexagons(TessellerProxy):
    """
    Wraps :class:`tramway.tessellation.hexagon.HexagonalMesh`.
    """
    __slots__ = ()
    def __init__(self, **kwargs):
        TessellerProxy.__init__(self, HexagonalMesh, **kwargs)
        self.alg_name = 'hexagon'
    def _reset_kwargs(self):
        TessellerProxy._reset_kwargs(self)
        self._tessellate_kwargs = dict(
                lower_bound = None,
                upper_bound = None,
                )
        self._init_kwargs.update(dict(
                tilt = 0.,
                min_probability = None,
                avg_probability = None,
                max_probability = None,
                min_distance = None,
                avg_distance = None,
                avg_location_count = None,
                ))

    #scaler          = proxy_property('scaler', 'attr')
    tilt            = proxy_property('tilt',    'attr')
    min_probability = proxy_property('min_probability', 'attr')
    avg_probability = proxy_property('avg_probability', 'attr')
    max_probability = proxy_property('max_probability', 'attr')
    min_distance    = proxy_property('min_distance',    'attr')
    avg_distance    = proxy_property('avg_distance',    'attr')
    lower_bound     = proxy_property('lower_bound', 'tessellate')
    upper_bound     = proxy_property('upper_bound', 'tessellate')


from tramway.tessellation.kmeans import KMeansMesh

class KMeans(TessellerProxy):
    """
    Wraps :class:`tramway.tessellation.kmeans.KMeansMesh`.
    """
    __slots__ = ()
    def __init__(self, **kwargs):
        TessellerProxy.__init__(self, KMeansMesh, **kwargs)
        self.alg_name = 'kmeans'
    def _reset_kwargs(self):
        TessellerProxy._reset_kwargs(self)
        self._tessellate_kwargs = dict(
                tol          = 1e-7,
                prune        = False, # differs from default
                plot         = False,
                avg_distance = None,
                )
        self._init_kwargs.update(dict(
                min_distance = None,
                avg_probability = None,
                avg_location_count = None,
                ))

    #scaler          = proxy_property('scaler', 'attr')
    initial         = proxy_property('initial', 'attr')
    avg_probability = proxy_property('avg_probability', 'attr')
    _min_distance   = proxy_property('_min_distance',   'attr')
    min_distance    = proxy_property('min_distance',    '__init__')
    tol             = proxy_property('tol', 'tessellate')
    prune           = proxy_property('prune',   'tessellate')
    plot            = proxy_property('plot',    'tessellate')
    avg_distance    = proxy_property('avg_distance',    'tessellate')


from tramway.tessellation.gwr import GasMesh

class GWR(TessellerProxy):
    """
    Wraps :class:`tramway.tessellation.gwr.GasMesh`.
    """
    __slots__ = ()
    def __init__(self, **kwargs):
        TessellerProxy.__init__(self, GasMesh, **kwargs)
        self.alg_name = 'gwr'
    def _reset_kwargs(self):
        TessellerProxy._reset_kwargs(self)
        self._tessellate_kwargs = dict(
                batch_size         = 10000,
                tau                = 333.,
                trust              = 1.,
                lifetime           = .05, # differs from default
                pass_count         = (),
                residual_factor    = .7,
                error_count_tol    = 5e-3,
                min_growth         = 1e-4,
                collapse_tol       = .01,
                stopping_criterion = 0,
                verbose            = False,
                alpha_risk         = 1e-15,
                complete_delaunay  = True, # differs from default
                )
        self._init_kwargs.update(dict(
                min_distance = None,
                avg_distance = None,
                max_distance = None,
                min_probability = None,
                min_location_count = None,
                ))

    #scaler             = proxy_property('scaler',  'attr')
    min_probability    = proxy_property('min_probability',  'attr')
    _min_distance      = proxy_property('_min_distance',    'attr')
    _avg_distance      = proxy_property('_avg_distance',    'attr')
    _max_distance      = proxy_property('_max_distance',    'attr')
    batch_size         = proxy_property('batch_size',   'tessellate')
    tau                = proxy_property('tau',  'tessellate')
    trust              = proxy_property('trust',    'tessellate')
    lifetime           = proxy_property('lifetime', 'tessellate')
    pass_count         = proxy_property('pass_count',   'tessellate')
    residual_factor    = proxy_property('residual_factor',  'tessellate')
    error_count_tol    = proxy_property('error_count_tol',  'tessellate')
    min_growth         = proxy_property('min_growth',   'tessellate')
    collapse_tol       = proxy_property('collapse_tol', 'tessellate')
    stopping_criterion = proxy_property('stopping_criterion',   'tessellate')
    verbose            = proxy_property('verbose',  'tessellate')
    alpha_risk         = proxy_property('alpha_risk',   'tessellate')
    complete_delaunay  = proxy_property('complete_delaunay',    'tessellate')


__all__ = ['Squares', 'Hexagons', 'KMeans', 'GWR']

