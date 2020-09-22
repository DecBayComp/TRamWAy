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

