
from ..attribute.abc import Attribute, abstractmethod

class Sampler(Attribute):
    @abstractmethod
    def sample(self, spt_dataframe, segmentation=None):
        pass

