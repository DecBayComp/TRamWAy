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
    """
    Can be left not-initialized.

    For local multi-processing:

    .. code:: python

        a.env = environments.LocalHost

    For *sbatch* job scheduling on an SSH-reachable server:

    .. code:: python

        a.env = environments.SlurmOverSSH

    See :class:`~.environments.LocalHost` and :class:`~.environments.SlurmOverSSH`.
    """
    __slots__ = ('_script',)
    def __init__(self, attribute_setter, parent=None):
        Initializer.__init__(self, attribute_setter, parent=parent)
        self._script = None
    def from_callable(self, env):
        if issubclass(env, Environment):
            self.specialize( env )
        else:
            raise TypeError('env is not an Environment')
    @property
    def script(self):
        """
        *str*: Path to the local script; this compatibility attribute
            is actually used by the :attr:`~tramway.analyzer.RWAnalyzer.browser`
        """
        #return None
        return self._script
    @script.setter
    def script(self, filename):
        #pass
        self._script = filename
    @property
    def collectibles(self):
        """ set: Compatilibity attribute; not used """
        return set()
    @property
    def submit_side(self):
        return False
    @property
    def worker_side(self):
        return False


__all__ = ['Environment', 'EnvironmentInitializer', 'environments']

