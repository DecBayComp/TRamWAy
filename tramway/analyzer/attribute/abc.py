
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

