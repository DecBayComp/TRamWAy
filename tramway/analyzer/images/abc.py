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
from ..attribute.abc import Attribute, abstractmethod

class Images(Attribute):
    def as_frames(self, index=None, return_time=False):
        """ see for example :meth:`~tramway.analyzer.images._RawImage.as_frames`. """
        pass
    def crop_frames(self, bounding_box, index=None, return_time=False):
        """ see for example :meth:`~tramway.analyzer.images._RawImage.crop_frames`. """
        pass

class Image(metaclass=ABCMeta):
    pass

__all__ = ['Image', 'Images']

