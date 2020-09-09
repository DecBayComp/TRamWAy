
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


class BasicLogger(object):
    def debug(self, msg):
        print(msg)
    def info(self, msg):
        print(msg)
    def warning(self, msg):
        print(msg)
    def error(self, msg):
        print(msg)
    def critical(self, msg):
        print(msg)
    def setLevel(self, lvl):
        pass


class RWAnalyzer(object):
    __slots__ = ( '_logger', '_spt_data', '_roi', '_time', '_tesseller', '_sampler', '_mapper',
            '_env', '_pipeline' )

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
        return self._spt_data
    def _set_spt_data(self, data):
        self._spt_data = data
    spt_data = selfinitializing_property('spt_data', _get_spt_data, _set_spt_data, SPTData)

    def _get_roi(self):
        return self._roi
    def _set_roi(self, roi):
        self._roi = roi
    roi = selfinitializing_property('roi', _get_roi, _set_roi, ROI)

    def _get_time(self):
        return self._time
    def _set_time(self, time):
        self._time = time
    time = selfinitializing_property('time', _get_time, _set_time, Time)

    def _get_tesseller(self):
        return self._tesseller
    def _set_tesseller(self, tesseller):
        self._tesseller = tesseller
    tesseller = selfinitializing_property('tesseller', _get_tesseller, _set_tesseller, Tesseller)

    def _get_sampler(self):
        return self._sampler
    def _set_sampler(self, sampler):
        self._sampler = sampler
    sampler = selfinitializing_property('sampler', _get_sampler, _set_sampler, Sampler)

    def _get_mapper(self):
        return self._mapper
    def _set_mapper(self, mapper):
        self._mapper = mapper
    mapper = selfinitializing_property('mapper', _get_mapper, _set_mapper, Mapper)

    def _get_env(self):
        return self._env
    def _set_env(self, env):
        self._env = env
    env = selfinitializing_property('env', _get_env, _set_env, Environment)

    def __init__(self):
        self._logger = \
                self._spt_data = \
                self._roi = \
                self._tesseller = \
                self._sampler = \
                self._mapper = \
                self._env = None
        self.spt_data  = SPTDataInitializer
        self.roi       = ROIInitializer
        self.time      = TimeInitializer
        self.tesseller = TessellerInitializer
        self.sampler   = SamplerInitializer
        self.mapper    = MapperInitializer
        self.env       = EnvironmentInitializer
        self._pipeline = Pipeline(self)

    @property
    def pipeline(self):
        return self._pipeline

    def run(self):
        return self.pipeline.run()

    def __setattr__(self, attrname, obj):
        if attrname[0] == '_' or (isinstance(obj, type) and issubclass(obj, Initializer)):
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


__all__ = ['RWAnalyzer', 'tessellers', 'Analysis', 'commit_as_analysis', 'environments']

