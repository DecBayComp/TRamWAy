
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
    @property
    def script(self):
        return None
    @script.setter
    def script(self, filename):
        pass


__all__ = ['Environment', 'EnvironmentInitializer', 'environments']

