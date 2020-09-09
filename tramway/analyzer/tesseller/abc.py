
from ..attribute.abc import Attribute, abstractmethod

class Tesseller(Attribute):
    @abstractmethod
    def tessellate(self, spt_dataframe):
        pass
    @property
    @abstractmethod
    def resolution(self):
        pass
    @resolution.setter
    @abstractmethod
    def resolution(self, res):
        pass

