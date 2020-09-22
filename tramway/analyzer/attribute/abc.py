# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from abc import ABCMeta, abstractmethod

class GenericAttribute(metaclass=ABCMeta):
    @property
    @abstractmethod
    def initialized(self):
        """most sub-attributes can be set only once the attribute object is initialized."""
        pass
    @property
    @abstractmethod
    def reified(self):
        """some sub-attributes can be set only as long as the attribute object has not been reified."""
        pass

class Attribute(GenericAttribute):
    @property
    def initialized(self):
        return True

