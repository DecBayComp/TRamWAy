
from ..attribute.abc import Attribute, abstractmethod

class Mapper(Attribute):
    @abstractmethod
    def infer(self, sampling):
        pass

