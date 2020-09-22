# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute.abc import Attribute, abstractmethod

class ROI(Attribute):
    @abstractmethod
    def __iter__(self):
        return self.as_individual_roi()
    @abstractmethod
    def as_individual_roi(self, index=None, collection=None):
        """ 
        If `collection` is defined, `index` is the index in the collection,
        otherwise, `index` is a linearized index across all the collections.

        In the case a single collection is defined, the two indexing should
        be equivalent.
        """
        pass
    @abstractmethod
    def as_support_regions(self, index=None):
        pass

