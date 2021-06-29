
from .plugin import MapperPlugin
import numpy as np

class DV(MapperPlugin):
    __slots__ = ()
    def __init__(self, variant='standard', start=None, **kwargs):

        err = ValueError(f"not sure what to do with variant='{variant}' and start='{start}'")
        if variant in ('naive', 'original'):
            if start is not None:
                raise NotImplementedError(f"variant='standard' and start='{start}'")
            plugin = 'dv'
        elif variant.startswith('semi') and variant[5:] == 'stochastic':
            if not (start is None or start == 'stochastic'):
                raise NotImplementedError(f"variant='semi-stochastic' and start='{start}'")
            plugin = 'semi.stochastic.dv'
            start = 'stochastic'
        elif variant == 'stochastic':
            plugin = 'stochastic.dv'
            start = None
        elif variant == 'standard':
            if start == 'stochastic':
                plugin = 'semi.stochastic.dv'
            elif start is None:
                plugin = 'stochastic.dv'
            else:
                raise err
        else:
            raise err

        MapperPlugin.__init__(self, plugin, **kwargs)

        if variant == 'standard' and start is None:
            self.stochastic = False
            self.allow_negative_potential = True
        elif variant == 'naive':
            self.rgrad = 'delta0'
        elif plugin.endswith('stochastic.dv'):
            # as long as superlocal is True:
            self.ls_step_max = np.r_[.025, .5]
            self.ftol = 1e-3
            self.low_df_rate = 1
            self.allow_negative_potential = True
            if start == 'stochastic':
                self.max_iter = 20
        self.verbose = False

    @property
    def _rgrad_is_delta0(self):
        if self.rgrad is None:
            return self._name in ('stochastic.dv', 'semi.stochastic.dv')
        else:
            return self.rgrad == 'delta0'

    #@property
    #def diffusivity_prior(self):
    #    lambda_ = self.get_plugin_arg('diffusivity_prior')
    #    if self._rgrad_is_delta0 and lambda_ is not None:
    #        lambda_ = 2 * lambda_
    #    return lambda_

    #@diffusivity_prior.setter
    #def diffusivity_prior(self, lambda_):
    #    if self._rgrad_is_delta0 and lambda_ is not None:
    #        lambda_ = lambda_ / 2
    #    self.set_plugin_arg('diffusivity_prior', lambda_)

    #@property
    #def diffusivity_spatial_prior(self):
    #    lambda_ = self.get_plugin_arg('diffusivity_spatial_prior')
    #    if self._rgrad_is_delta0 and lambda_ is not None:
    #        lambda_ = 2 * lambda_
    #    return lambda_

    #@diffusivity_spatial_prior.setter
    #def diffusivity_spatial_prior(self, lambda_):
    #    if self._rgrad_is_delta0 and lambda_ is not None:
    #        lambda_ = lambda_ / 2
    #    self.set_plugin_arg('diffusivity_spatial_prior', lambda_)

    #@property
    #def potential_prior(self):
    #    lambda_ = self.get_plugin_arg('potential_prior')
    #    if self._rgrad_is_delta0 and lambda_ is not None:
    #        lambda_ = 2 * lambda_
    #    return lambda_

    #@potential_prior.setter
    #def potential_prior(self, lambda_):
    #    if self._rgrad_is_delta0 and lambda_ is not None:
    #        lambda_ = lambda_ / 2
    #    self.set_plugin_arg('potential_prior', lambda_)

    #@property
    #def potential_spatial_prior(self):
    #    lambda_ = self.get_plugin_arg('potential_spatial_prior')
    #    if self._rgrad_is_delta0 and lambda_ is not None:
    #        lambda_ = 2 * lambda_
    #    return lambda_

    #@potential_spatial_prior.setter
    #def potential_spatial_prior(self, lambda_):
    #    if self._rgrad_is_delta0 and lambda_ is not None:
    #        lambda_ = lambda_ / 2
    #    self.set_plugin_arg('potential_spatial_prior', lambda_)

    #@property
    #def verbose(self):
    #    return self.get_plugin_arg('verbose')

    #@verbose.setter
    #def verbose(self, v):
    #    import logging
    #    from tramway.inference.optimization import module_logger
    #    if v:
    #        lvl = logging.DEBUG
    #    else:
    #        v = True
    #        lvl = logging.WARNING
    #    module_logger.setLevel(lvl)
    #    self.set_plugin_arg('verbose', v)
