
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

