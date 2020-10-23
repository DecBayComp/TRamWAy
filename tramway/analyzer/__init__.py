# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core.exceptions import MisplacedAttributeWarning
from warnings   import warn

def report_misplaced_attribute(attr_name, proper_parent_name):
    warn('`{}` is an attribute of the initialized `{}` attribute; this warning message can safely be silenced'.format(attr_name, proper_parent_name), MisplacedAttributeWarning)
def proper_parent_name(attr_name):
    parent_name = None
    get_conditions, set_conditions = {}, {}
    set_conditions['initialized'] = True
    if attr_name in ('dt', 'time_step', 'localization_error', 'localization_precision', 'columns'):
        parent_name = 'spt_data'
    elif attr_name in ('scaler', 'resolution'):
        parent_name = 'tesseller'
    return parent_name, get_conditions, set_conditions

from .attribute import *
from .artefact  import *
from .spt_data  import *
from .roi       import *
from .time      import *
from .tesseller import *
from .sampler   import *
from .mapper    import *
from .env       import *
from .pipeline  import *
from .browser   import *
from .images    import *


class BasicLogger(object):
    """
    emulates the most basic functionalities of `logging.Logger`
    without the need for additional configuration.
    """
    __slots__ = ('_level',)
    def __init__(self, level=0):
        self.level = level
    def _print(self, msg, lvl):
        if self.level <= lvl:
            print(msg)
    def debug(self, msg):
        self._print(msg, 10)
    def info(self, msg):
        self._print(msg, 20)
    def warning(self, msg):
        self._print(msg, 30)
    def error(self, msg):
        self._print(msg, 40)
    def critical(self, msg):
        self._print(msg, 50)
    @property
    def level(self):
        return self._level
    @level.setter
    def level(self, lvl):
        if isinstance(lvl, str):
            lvl = lvl.lower()
            lvl = dict(
                    notset = 0,
                    debug = 10,
                    info = 20,
                    warning = 30,
                    error = 40,
                    critical = 50,
                ).get(lvl, 0)
        elif not isinstance(lvl, int):
            try:
                lvl = int(lvl)
            except:
                lvl = 0
        self._level = lvl
    def setLevel(self, lvl):
        self.level = lvl


class RWAnalyzer(object):
    """
    A `RWAnalyzer` object gathers the paremeters of all the processing steps
    of a standard processing chain, from SPT data loading/generation to
    inferring model parameters at microdomains.

    The supported steps are defined in a declarative way with spatial attributes;
    these steps and corresponding attributes are as follows:

    * `spt_data`: SPT data loading or generation
    * `roi`: regions of interest
    * `time`: temporal segmentation of the tracking data
    * `tesseller`: spatial segmentation
    * `sampler`: assignment of SPT data points to microdomains
    * `mapper`: estimation of model parameters at each microdomains

    Most attributes are self-morphing, i.e. they first are initializers and exhibit
    *from_...* methods (for example `from_dataframe` and `from_ascii_file`
    for `spt_data`) and then, once any such initializer method is called, they specialize
    into a new attribute and exhibit specific attributes depending on the chosen
    initializer.

    Specialized attributes themselves can exhibit self-morphing attributes.
    For example, regions of interest can be defined globally using the main
    `roi` attribute, or on a per-SPT-dataset basis:

    .. code-block:: python

        a = RWAnalyzer()
        a.spt_data.from_ascii_files('my_data_repository/*.txt')
        a.roi.from_squares(roi_centers, square_size)

    or

    .. code-block:: python

        a = RWAnalyzer()
        a.spt_data.from_ascii_files('my_data_repository/*.txt')
        for spt_file in a.spt_data:
            spt_file.roi.from_squares(roi_centers, square_size)

    In the above example, per-dataset ROI definition is useful when multiple
    datasets are loaded and the ROI may differ between datasets.
    The main `roi` attribute is still convenient as it allows to iterate
    over all the defined ROI, omitting the `spt_data` loop (continues any of
    the code blocks above):

    .. code-block:: python

        for roi in a.roi.as_support_regions():
            roi_spt_data = roi.crop()

    See the documentation of the `roi` attribute for more information about
    the various iterators available.

    While the `spt_data` and `roi` attributes act as data providers,
    the `time`, `tesseller`, `sampler` and `mapper` attributes do not feature
    direct access to the data and require the SPT data to be passed as input
    argument to their main processing methods.
    For example:

    .. code-block:: python

        a.tesseller.from_plugin('kmeans')
        for roi in a.roi.as_support_regions():
            roi_spt_data = roi.crop()
            tessellation = a.tesseller.tessellate(roi_spt_data)


    Other attributes drive the execution of the processing chain.
    The `run` method launches the processing chain, which is operated by the
    `pipeline` attribute.
    
    Various parallelization schemes are available, and the platform-specific
    implementation of these schemes are provided by the `env` attribute.

    Last but not least, the `RWAnalyzer` allows to plot the inferred parameter maps
    from a Jupyter notebook, or calling the ``bokeh serve`` command:

    .. code-block:: python

        from tramway.analyzer import *

        a = RWAnalyzer()

        # load the rwa files available in the current directory:
        a.spt_data.from_rwa_files('*.rwa')

        # help the analyzer to localize this piece of code:
        try:
            a.script = __file__
        except NameError: # in a notebook
            a.script = 'MyNotebook.ipynb' # this notebook's name (please adapt)

        a.browser.show_maps()

    See also :class:`~tramway.analyzer.browser.Browser` for additional information
    on how to export data and figures while browsing the inferred parameter maps.

    """
    __slots__ = ( '_logger', '_spt_data', '_roi', '_time', '_tesseller', '_sampler', '_mapper',
            '_env', '_pipeline', '_browser', '_images' )

    @property
    def logger(self):
        if self._logger is None:
            self._logger = BasicLogger()
            #import logging
            #self._logger = logging.getLogger(__name__)
            #self._logger.setLevel(logging.DEBUG)
        return self._logger
    @logger.setter
    def logger(self, logger):
        self._logger = logger

    def _get_spt_data(self):
        """
        SPT data accessor.

        See :class:`~spt_data.SPTDataInitializer`.
        """
        return self._spt_data
    def _set_spt_data(self, data):
        self._spt_data = data
    spt_data = selfinitializing_property('spt_data', _get_spt_data, _set_spt_data, SPTData)

    def _get_roi(self):
        """
        ROI accessor.

        See :class:`~roi.ROIInitializer`.
        """
        return self._roi
    def _set_roi(self, roi):
        self._roi = roi
    roi = selfinitializing_property('roi', _get_roi, _set_roi, ROI)

    def _get_time(self):
        """
        Time segmentation procedure.

        See :class:`~time.TimeInitializer`.
        """
        return self._time
    def _set_time(self, time):
        self._time = time
    time = selfinitializing_property('time', _get_time, _set_time, Time)

    def _get_tesseller(self):
        """
        Tessellation procedure.

        See :class:`~tesseller.TessellerInitializer`.
        """
        return self._tesseller
    def _set_tesseller(self, tesseller):
        self._tesseller = tesseller
    tesseller = selfinitializing_property('tesseller', _get_tesseller, _set_tesseller, Tesseller)

    def _get_sampler(self):
        """
        Sampling procedure.

        See :class:`~sampler.SamplerInitializer`.
        """
        return self._sampler
    def _set_sampler(self, sampler):
        self._sampler = sampler
    sampler = selfinitializing_property('sampler', _get_sampler, _set_sampler, Sampler)

    def _get_mapper(self):
        """
        Inference procedure.

        See :class:`~mapper.MapperInitializer`.
        """
        return self._mapper
    def _set_mapper(self, mapper):
        self._mapper = mapper
    mapper = selfinitializing_property('mapper', _get_mapper, _set_mapper, Mapper)

    def _get_env(self):
        """
        Environment backend for operating the pipeline.

        If not set, the pipeline will run locally in the current interpreter.

        See :mod:`~env.environments`.
        """
        return self._env
    def _set_env(self, env):
        self._env = env
    env = selfinitializing_property('env', _get_env, _set_env, Environment)

    def _get_images(self):
        """
        Single molecule microscopy image stacks.

        See :class:`~images.ImagesInitializer`.
        """
        return self._images
    def _set_images(self, images):
        self._images = images
    images = selfinitializing_property('images', _get_images, _set_images, Images)

    def __init__(self):
        self._logger = \
                self._spt_data = \
                self._roi = \
                self._tesseller = \
                self._sampler = \
                self._mapper = \
                self._env = \
                self._images = None
        self.spt_data  = SPTDataInitializer
        self.roi       = ROIInitializer
        self.time      = TimeInitializer
        self.tesseller = TessellerInitializer
        self.sampler   = SamplerInitializer
        self.mapper    = MapperInitializer
        self.env       = EnvironmentInitializer
        self.images    = ImagesInitializer
        self._pipeline = Pipeline(self)
        self._browser  = Browser(self)

    @property
    def pipeline(self):
        """
        Parallelization scheme.

        See `env`.
        """
        return self._pipeline

    def run(self):
        """
        launches the pipeline.
        """
        return self.pipeline.run()

    def add_collectible(self, collectible):
        """
        designates a file generated at the worker side to be transferred back to the submit side.

        See :meth:`Pipeline.add_collectible`
        """
        self.pipeline.add_collectible(collectible)

    @property
    def browser(self):
        """
        Visualize data artefacts.

        See also :class:`Browser`.
        """
        return self._browser

    @property
    def script(self):
        """
        Path to the *__main__* file in which the analyzer is defined.

        Designating a script is required for parallelizing computations,
        or visualizing maps without explicitly calling the ``bokeh serve`` command
        (e.g. from a Jupyter notebook).
        """
        return self.env.script
    @script.setter
    def script(self, filename):
        self.env.script = filename

    def __setattr__(self, attrname, obj):
        if attrname[0] == '_' or (isinstance(obj, type) and issubclass(obj, Initializer)) or\
                attrname in ('script',):
            object.__setattr__(self, attrname, obj)
        elif callable(obj):
            attr = getattr(self, attrname)
            try:
                attr.from_callable(obj)
            except AttributeError:
                raise AttributeError('attribute is read-only')
        else:
            parent_name, _, set_conditions = proper_parent_name(attrname)
            if parent_name is None:
                getattr(self, attrname) # raises AttributeError if no such attribute is found
                raise AttributeError('attribute `{}` is read-only'.format(attrname))
            else:
                report_misplaced_attribute(attrname, parent_name)
                proper_parent = getattr(self, parent_name)
                if set_conditions.get('initialized', False) and isinstance(proper_parent, Initializer):
                    raise AttributeError('`{}` cannot be set as long as `{}` is not initialized'.format(attrname, parent_name))
                setattr(proper_parent, attrname, obj)

    def __getattr__(self, attrname):
        parent_name, _, _ = proper_parent_name(attrname)
        if parent_name is None:
            raise AttributeError('RWAnalyzer has no attribute `{}`'.format(attrname))
        else:
            proper_parent = getattr(self, parent_name)
            return getattr(proper_parent, attrname)


__all__ = ['RWAnalyzer',
        'tessellers', 'cell_mergers',
        'Analysis', 'commit_as_analysis',
        'environments']

