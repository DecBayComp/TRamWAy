# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from tramway.core.exceptions import MisplacedAttributeWarning, SideEffectWarning
import warnings
import logging

def report_misplaced_attribute(attr_name, proper_parent_name):
    warnings.warn('`{}` is an attribute of the initialized `{}` attribute; this warning message can safely be silenced'.format(attr_name, proper_parent_name), MisplacedAttributeWarning)
def proper_parent_name(attr_name):
    parent_name = None
    get_conditions, set_conditions = {}, {}
    set_conditions['initialized'] = True
    if attr_name in ('dt', 'time_step', 'frame_interval',
            'localization_error', 'localization_precision', 'columns'):
        parent_name = 'spt_data'
    elif attr_name in ('scaler', 'resolution'):
        parent_name = 'tesseller'
    return parent_name, get_conditions, set_conditions

warnings.filterwarnings('error', category=SideEffectWarning)


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
from .localizer import *
from .tracker   import *


import importlib
class AttributeSubPackage(object):
    __slots__ = ('__attrname__', '_module')
    def __init__(self, attrname):
        self.__attrname__ = attrname
        self.module = None
    @property
    def module(self):
        if self._module is None:
            self._module = importlib.import_module(
                    '.{}.allsymbols'.format(self.__attrname__),
                    package='tramway.analyzer')
        return self._module
    @module.setter
    def module(self, mod):
        self._module = mod
    def __getattr__(self, attrname):
        return getattr(self.module, attrname)

spt_data  = AttributeSubPackage('spt_data' )
roi       = AttributeSubPackage('roi'      )
time      = AttributeSubPackage('time'     )
tesseller = AttributeSubPackage('tesseller')
sampler   = AttributeSubPackage('sampler'  )
mapper    = AttributeSubPackage('mapper'   )
images    = AttributeSubPackage('images'   )
tracker   = AttributeSubPackage('tracker'  )


class BasicLogger(object):
    """
    Emulates the most basic functionalities of :class:`logging.Logger`
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
    A :class:`RWAnalyzer` object gathers the parameters of all the processing steps
    of a standard processing chain, from SPT data loading/generation to
    inferring model parameters at microdomains.

    The supported steps are defined in a declarative way with special attributes;
    these steps and corresponding attributes are as follows:

    * :attr:`~tramway.analyzer.RWAnalyzer.images`: microscopy images
    * :attr:`~tramway.analyzer.RWAnalyzer.localizer`: single molecule localization
    * :attr:`~tramway.analyzer.RWAnalyzer.tracker`: single particle tracking
    * :attr:`~tramway.analyzer.RWAnalyzer.spt_data`: SPT data loading or generation
    * :attr:`~tramway.analyzer.RWAnalyzer.roi`: regions of interest
    * :attr:`~tramway.analyzer.RWAnalyzer.tesseller`: spatial segmentation
    * :attr:`~tramway.analyzer.RWAnalyzer.time`: temporal segmentation of the tracking data
    * :attr:`~tramway.analyzer.RWAnalyzer.sampler`: assignment of SPT data points to microdomains
    * :attr:`~tramway.analyzer.RWAnalyzer.mapper`: estimation of model parameters at each microdomains

    Most attributes are self-morphing, i.e. they first are initializers and exhibit
    *from_...* methods (for example
    :meth:`~.spt_data.SPTDataInitializer.from_dataframe` and
    :meth:`~.spt_data.SPTDataInitializer.from_ascii_file` for
    :attr:`~tramway.analyzer.RWAnalyzer.spt_data`) and then,
    once any such initializer method is called,
    they specialize into a new attribute and exhibit specific attributes depending
    on the chosen initializer.

    Specialized attributes can also exhibit self-morphing attributes.
    For example, regions of interest can be defined globally using the main
    :attr:`~tramway.analyzer.RWAnalyzer.roi` attribute, or on a per-SPT-datafile basis:

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
    The main :attr:`~tramway.analyzer.RWAnalyzer.roi` attribute is still convenient
    as it allows to iterate over all the defined ROI,
    omitting the :attr:`~tramway.analyzer.RWAnalyzer.spt_data` loop (continues any
    of the code blocks above):

    .. code-block:: python

        for roi in a.roi.as_support_regions():
            roi_spt_data = roi.crop()

    See the documentation for the :attr:`~tramway.analyzer.RWAnalyzer.roi` attribute
    for more information about the available iterators.

    While the :attr:`~tramway.analyzer.RWAnalyzer.spt_data` and
    :attr:`~tramway.analyzer.RWAnalyzer.roi` attributes act as data providers,
    the :attr:`~tramway.analyzer.RWAnalyzer.time`,
    :attr:`~tramway.analyzer.RWAnalyzer.tesseller`,
    :attr:`~tramway.analyzer.RWAnalyzer.sampler` and
    :attr:`~tramway.analyzer.RWAnalyzer.mapper` attributes
    do not feature direct access to the data and require the SPT data to be passed as
    input argument to their main processing methods.
    For example:

    .. code-block:: python

        a.tesseller.from_plugin('kmeans')
        for roi in a.roi.as_support_regions():
            roi_spt_data = roi.crop()
            tessellation = a.tesseller.tessellate(roi_spt_data)

    Similarly, the :attr:`~tramway.analyzer.RWAnalyzer.images` attribute defines data location,
    while the :attr:`~tramway.analyzer.RWAnalyzer.localizer` and
    :attr:`~tramway.analyzer.RWAnalyzer.tracker` attributes define processing steps
    on these data.

    Other attributes drive the execution of the processing chain.
    The :meth:`run` method launches the processing chain, which is operated by the
    :attr:`~tramway.analyzer.RWAnalyzer.pipeline` attribute.
    
    Various parallelization schemes are available, and the platform-specific
    implementation of these schemes are provided by the
    :attr:`~tramway.analyzer.RWAnalyzer.env` attribute.

    Last but not least, the :class:`RWAnalyzer` features plotting utilities.
    Some of them are available through the *mpl* sub-attribute of some
    main :class:`RWAnalyzer` attributes or items
    (for example :attr:`~.images._RawImage.mpl`, :attr:`~.spt_data._SPTDataFrame.mpl`,
    :attr:`~.tesseller.TessellerInitializer.mpl`, :attr:`~.mapper.MapperInitializer.mpl`).
    In addition, the :attr:`~tramway.analyzer.RWAnalyzer.browser` attribute
    can plot the inferred parameter maps from a Jupyter notebook,
    or calling the ``bokeh serve`` command:

    .. code-block:: python

        from tramway.analyzer import *

        a = RWAnalyzer()

        # load the rwa files available in the current directory:
        a.spt_data.from_rwa_files('*.rwa')

        # help the analyzer locate this piece of code:
        try:
            a.script = __file__
        except NameError: # in a notebook
            a.script = 'MyNotebook.ipynb' # this notebook's name (please adapt)

        a.browser.show_maps()

    See also :class:`~tramway.analyzer.browser.Browser` for additional information
    on how to export data and figures while browsing the inferred parameter maps.

    """
    __slots__ = ( '_logger', '_spt_data', '_roi', '_time', '_tesseller', '_sampler', '_mapper',
            '_env', '_pipeline', '_browser', '_images', '_localizer', '_tracker' )

    @property
    def logger(self):
        """
        """
        if self._logger is None:
            #self._logger = BasicLogger()
            import logging
            self._logger = logging.getLogger(__name__)
            if not self._logger.hasHandlers():
                self._logger.setLevel(logging.INFO)
                self._logger.addHandler(logging.StreamHandler())
        return self._logger
    @logger.setter
    def logger(self, logger):
        self._logger = logger

    def _get_spt_data(self):
        """
        SPT data accessor.

        See :class:`~tramway.analyzer.spt_data.SPTDataInitializer`
        and :class:`~tramway.analyzer.spt_data.SPTData`.
        """
        return self._spt_data
    def _set_spt_data(self, data):
        self._spt_data = data
    spt_data = selfinitializing_property('spt_data', _get_spt_data, _set_spt_data, SPTData)

    def _get_roi(self):
        """
        ROI accessor.

        See :class:`~tramway.analyzer.roi.ROIInitializer`
        and :class:`~tramway.analyzer.roi.ROI`.
        """
        return self._roi
    def _set_roi(self, roi):
        self._roi = roi
    roi = selfinitializing_property('roi', _get_roi, _set_roi, ROI)

    def _get_time(self):
        """
        Time segmentation procedure.

        See :class:`~tramway.analyzer.time.TimeInitializer`
        and :class:`~tramway.analyzer.time.Time`.
        """
        return self._time
    def _set_time(self, time):
        self._time = time
    time = selfinitializing_property('time', _get_time, _set_time, Time)

    def _get_tesseller(self):
        """
        Tessellation procedure.

        See :class:`~tramway.analyzer.tesseller.TessellerInitializer`
        and :class:`~tramway.analyzer.tesseller.Tesseller`.
        """
        return self._tesseller
    def _set_tesseller(self, tesseller):
        self._tesseller = tesseller
    tesseller = selfinitializing_property('tesseller', _get_tesseller, _set_tesseller, Tesseller)

    def _get_sampler(self):
        """
        Sampling procedure.

        See :class:`~tramway.analyzer.sampler.SamplerInitializer`
        and :class:`~tramway.analyzer.sampler.Sampler`.
        """
        return self._sampler
    def _set_sampler(self, sampler):
        self._sampler = sampler
    sampler = selfinitializing_property('sampler', _get_sampler, _set_sampler, Sampler)

    def _get_mapper(self):
        """
        Inference procedure.

        See :class:`~tramway.analyzer.mapper.MapperInitializer`
        and :class:`~tramway.analyzer.mapper.Mapper`.
        """
        return self._mapper
    def _set_mapper(self, mapper):
        self._mapper = mapper
    mapper = selfinitializing_property('mapper', _get_mapper, _set_mapper, Mapper)

    def _get_env(self):
        """
        Environment backend for operating the pipeline.

        If not set, the pipeline will run locally in the current interpreter.

        See :mod:`~tramway.analyzer.env.environments`.
        """
        return self._env
    def _set_env(self, env):
        self._env = env
    env = selfinitializing_property('env', _get_env, _set_env, Environment)

    def _get_images(self):
        """
        Microscopy image stacks.

        See :class:`~tramway.analyzer.images.ImagesInitializer`
        and :class:`~tramway.analyzer.images.Images`.
        """
        return self._images
    def _set_images(self, images):
        self._images = images
    images = selfinitializing_property('images', _get_images, _set_images, Images)

    def _get_localizer(self):
        """
        Single molecule localization procedure.

        See :class:`~tramway.analyzer.localizer.LocalizerInitializer`
        and :class:`~tramway.analyzer.localizer.Localizer`.
        """
        return self._localizer
    def _set_localizer(self, localizer):
        self._localizer = localizer
    localizer = selfinitializing_property('localizer', _get_localizer, _set_localizer, Localizer)

    def _get_tracker(self):
        """
        Single particle tracking procedure.

        See :class:`~tramway.analyzer.tracker.TrackerInitializer`
        and :class:`~tramway.analyzer.tracker.Tracker`.
        """
        return self._tracker
    def _set_tracker(self, tracker):
        self._tracker = tracker
    tracker = selfinitializing_property('tracker', _get_tracker, _set_tracker, Tracker)

    def __init__(self):
        self._logger = \
                self._spt_data = \
                self._roi = \
                self._tesseller = \
                self._sampler = \
                self._mapper = \
                self._env = \
                self._images = \
                self._localizer = \
                self._tracker = None
        self.spt_data  = SPTDataInitializer
        self.roi       = ROIInitializer
        self.time      = TimeInitializer
        self.tesseller = TessellerInitializer
        self.sampler   = SamplerInitializer
        self.mapper    = MapperInitializer
        self.images    = ImagesInitializer
        self.localizer = LocalizerInitializer
        self.tracker   = TrackerInitializer
        self._pipeline = Pipeline(self)
        self.env       = EnvironmentInitializer
        self._browser  = Browser(self)

    @property
    def pipeline(self):
        """
        Parallelization scheme.

        See :class:`~tramway.analyzer.pipeline.Pipeline`.
        """
        return self._pipeline

    def run(self):
        """
        Launches the pipeline.

        Alias for :attr:`~tramway.analyzer.RWAnalyzer.pipeline` :meth:`~.pipeline.Pipeline.run`.
        """
        return self.pipeline.run()

    def add_collectible(self, collectible):
        """
        Designates a file generated on the worker side to be transferred back to the submit side.

        Alias for :attr:`~tramway.analyzer.RWAnalyzer.pipeline`
        :meth:`~.pipeline.Pipeline.add_collectible`.
        """
        self.pipeline.add_collectible(collectible)

    @property
    def browser(self):
        """
        Data visualization and export.

        See :class:`~.browser.Browser`.
        """
        return self._browser

    @property
    def script(self):
        """
        Path to the *__main__* file in which the analyzer is defined.

        Designating a script is required for parallelizing computations,
        or visualizing maps without explicitly calling the ``bokeh serve`` command
        (e.g. from a Jupyter notebook).

        Alias for :attr:`~tramway.analyzer.RWAnalyzer.env` :attr:`~.env.Environment.script`.
        """
        return self.env.script
    @script.setter
    def script(self, filename):
        self.env.script = filename

    def __del__(self):
        if self.spt_data.initialized:
            try:
                for f in self.spt_data:
                    try:
                        f._analyses.terminate()
                    except AttributeError:
                        pass
            except ValueError:
                pass

    def __setattr__(self, attrname, obj):
        if attrname[0] == '_' or (isinstance(obj, type) and issubclass(obj, Initializer)) or\
                attrname in ('script',):
            object.__setattr__(self, attrname, obj)
        elif callable(obj):
            if attrname[0] != '_' and isinstance(obj, InitializerMethod):
                attr = getattr(self, attrname)
                if isinstance(attr, Initializer):
                    obj.assign( self )
                else:
                    warnings.warn(
                            "attribute '{}' is already initialized; side effects may occur".format(
                                attrname),
                            SideEffectWarning)
                    obj.reassign( self )
            else:
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
        'spt_data', 'roi',
        'time', 'tesseller', 'sampler',
        'mapper',
        'images', 'tracker',
        'tessellers', 'cell_mergers',
        'Analysis', 'commit_as_analysis',
        'environments',
        'first', 'single',
        'SideEffectWarning',
        'stages',
        ]

