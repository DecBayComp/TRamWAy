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
# for overloading :meth:`tessellate`
from ..artefact import analysis
from copy import deepcopy


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
    """ *ndarray* or *Series*: Scaled lower space bounds """
    upper_bound     = proxy_property('upper_bound', 'attr')
    """ *ndarray* or *Series*: Scaled upper space bounds """
    count_per_dim   = proxy_property('count_per_dim',   'attr')
    """ *int*: Number of squares per space dimension """
    #min_probability = proxy_property('min_probability', 'attr')
    #""" *float*: (not used) """
    avg_probability = proxy_property('avg_probability', 'attr')
    """ *float*: Average probability for a point to fall within any given square """
    #max_probability = proxy_property('max_probability', 'attr')
    #""" *float*: (not used) """
    min_distance    = proxy_property('min_distance',    'attr')
    """
    *float*:
        Minimum distance between adjacent square centers or, equivalently, minimum square size;
        ignored if :attr:`avg_distance` is defined
    """
    avg_distance    = proxy_property('avg_distance',    'attr')
    """
    *float*:
        Average distance between adjacent square centers or, equivalently, average square size;
        ignored if :attr:`avg_probability` is defined
    """


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
    r"""
    *float*:
        Angle of the "main" axis, in :math:`\frac{\pi}{6}` radians;
        0= the main axis is the x-axis; 1= the main axis is the y-axis
    """
    #min_probability = proxy_property('min_probability', 'attr')
    #""" *float*: (not used) """
    avg_probability = proxy_property('avg_probability', 'attr')
    """ *float*: Average probability for a point to fall within any given hexagon """
    #max_probability = proxy_property('max_probability', 'attr')
    #""" *float*: (not used) """
    min_distance    = proxy_property('min_distance',    'attr')
    """
    *float*:
        Minimum distance between adjacent hexagon centers or, equivalently,
        minimum diameter of the inscribed circle;
        ignored if :attr:`avg_distance` is defined
    """
    avg_distance    = proxy_property('avg_distance',    'attr')
    """
    *float*:
        Average distance between adjacent hexagon centers or, equivalently,
        average diameter of the inscribed circle;
        ignored if :attr:`avg_probability` is defined
    """
    lower_bound     = proxy_property('lower_bound', 'tessellate')
    """ *ndarray* or *Series*: Scaled lower space bounds """
    upper_bound     = proxy_property('upper_bound', 'tessellate')
    """ *ndarray* or *Series*: Scaled upper space bounds """


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
    """
    *str*:
        Any of :const:`'grid'` (default), :const:`'random'` and :const:`'center'`;
        :const:`'grid'` uses :class:`Squares` to initialize the cell centers;
        :const:`'random'` randomly samples the initial cell centers;
        :const:`'center'` is similar to :const:`'random'` but confines the random cell
        centers within a tiny area in the middle
    """
    avg_probability = proxy_property('avg_probability', 'attr')
    """ *float*: Average probability for a point to fall within any given cell """
    tol             = proxy_property('tol', 'tessellate')
    """ *float*: Error tolerance; passed as `thresh` to :func:`scipy.cluster.vq.kmeans` """
    prune           = proxy_property('prune',   'tessellate')
    """
    *bool* or *float*:
        Maximum edge length for the Voronoi graph, relative to the median edge length;
        if :const:`True`, defaults to :const:`2.5`;
        the edges that exceed the derived threshold value are pruned
    """
    #plot            = proxy_property('plot',    'tessellate')
    #_min_distance   = proxy_property('_min_distance',   'attr')
    #""" *float*: Scaled minimum distance between adjacent cell centers """
    min_distance    = proxy_property('min_distance',    '__init__')
    """
    *float*:
        Minimum distance between adjacent cell centers;
        affects the :const:`'grid'` initialization only
    """
    avg_distance    = proxy_property('avg_distance',    'tessellate')
    """
    *float*:
        Average distance between adjacent cell centers;
        affects the :const:`'grid'` initialization only
    """


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
                topology           = None, # differs from default
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
    """ *float*: Minimum probability for a point to fall within any given cell """
    _min_distance      = proxy_property('_min_distance',    'attr')
    _avg_distance      = proxy_property('_avg_distance',    'attr')
    _max_distance      = proxy_property('_max_distance',    'attr')
    batch_size         = proxy_property('batch_size',   'tessellate')
    """
    *int*:
        Number of points per epoch;
        this merely affects the node merging mechanism triggered between epochs
    """
    tau                = proxy_property('tau',  'tessellate')
    """
    *float* or *(float, float)*:
        Habituation scaling factor, in number of iterations,
        for the nearest and second nearest nodes respectively
    """
    trust              = proxy_property('trust',    'tessellate')
    r"""
    *float*:
        Confidence in outstanding points to become new nodes;
        the new node is set as :math:`\frac{(1-\rm{trust})*w + (1+\rm{trust})*\eta}{2}`;
        the original GWR algorithm corresponds to ``trust = 0``
    """
    lifetime           = proxy_property('lifetime', 'tessellate')
    """
    *float*:
        Edge lifetime in number of iterations;
        taken as absolute, if greater than 1;
        otherwise relative to the number of nodes in the graph
    """
    pass_count         = proxy_property('pass_count',   'tessellate')
    """
    *float* or *(float, float)*:
        Number of points to sample (with replacement), as a multiple of the
        total size of the data;
        a pair of values denotes the lower and upper bounds on the number
        of points;
        a single value is interpreted as the lower bound,
        and the upper bound is then set equal to ``2 * pass_count``
    """
    residual_factor    = proxy_property('residual_factor',  'tessellate')
    """
    *float*:
        Maximum residual for each sample point, relative to :attr:`_max_distance`
    """
    error_count_tol    = proxy_property('error_count_tol',  'tessellate')
    """
    *float*:
        Maximum rate of errors as defined by :attr:`residual_factor` and the batch size;
        as long as the rate does not fall below this threshold value, :meth:`tessellate`
        keeps on sampling batches
    """
    min_growth         = proxy_property('min_growth',   'tessellate')
    """
    *float*:
        Minimum relative increase in the number of nodes
    """
    collapse_tol       = proxy_property('collapse_tol', 'tessellate')
    """
    *float*:
        Maximum ratio of the number of collapsed nodes over the total
        number of nodes
    """
    #stopping_criterion = proxy_property('stopping_criterion',   'tessellate')
    #""" *int*: (deprecated) """
    verbose            = proxy_property('verbose',  'tessellate')
    """ *bool*: Verbose mode; default is :const:`False` """
    alpha_risk         = proxy_property('alpha_risk',   'tessellate')
    """
    *float*:
        Threshold p-value in the comparison between distance distributions
        to determine whether the edges that are not in the Delaunay graph
        are still valid
    """
    complete_delaunay  = proxy_property('complete_delaunay',    'tessellate')
    """
    *bool*:
        Use the Delaunay graph instead, for determining the cell adjacency;
        default is :const:`True`
    """
    topology           = proxy_property('topology', 'tessellate')
    """
    *str*:
        Any of :const:`'approximate density'` and :const:`'displacement length'`
        (default for trajectory/translocation data)
    """

    @analysis
    def tessellate(self, spt_dataframe):
        if not isinstance(self._tesseller, tuple):
            self.calibrate(spt_dataframe)
        tesseller = deepcopy(self.tesseller)
        # modified part below
        delta_columns = [ 'd'+col for col in self.colnames ]
        if all([ col in spt_dataframe.columns for col in delta_columns ]):
            data = (spt_dataframe[self.colnames], spt_dataframe[delta_columns])
            if self.topology is None:
                self.topology = 'displacement length'
        else:
            data = spt_dataframe[self.colnames]
        tesseller.tessellate(data, **self._tessellate_kwargs)
        # modified part above
        if self.post_processing.initialized:
            tesseller = self.post_processing.post_process(tesseller, spt_dataframe[self.colnames])
        #
        return tesseller


__all__ = ['Squares', 'Hexagons', 'KMeans', 'GWR']

