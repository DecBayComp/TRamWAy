# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from abc import *

class Analyses(metaclass=ABCMeta):
    @property
    @abstractmethod
    def data(self):
        pass
    @data.setter
    @abstractmethod
    def data(self, d):
        pass
    @property
    @abstractmethod
    def artefact(self):
        pass
    @artefact.setter
    @abstractmethod
    def artefact(self, a):
        pass
    @property
    @abstractmethod
    def metadata(self):
        pass
    @metadata.setter
    @abstractmethod
    def metadata(self, d):
        pass
    @property
    @abstractmethod
    def instances(self):
        pass
    @property
    @abstractmethod
    def comments(self):
        pass
    @property
    @abstractmethod
    def labels(self):
        pass
    @abstractmethod
    def autoindex(self, pattern=None):
        pass
    @abstractmethod
    def add(self, analysis, label=None, comment=None, raw=False):
        pass

__all__ = ['Analyses']

