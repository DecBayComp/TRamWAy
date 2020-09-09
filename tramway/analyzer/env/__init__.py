
from ..attribute import *
from .abc import *
from . import environments


class EnvironmentInitializer(Initializer):
    __slots__ = ()
    def from_callable(self, env):
        if issubclass(env, Environment):
            self.specialize( env )
        else:
            raise TypeError('env is not an Environment')


__all__ = ['Environment', 'EnvironmentInitializer', 'environments']

