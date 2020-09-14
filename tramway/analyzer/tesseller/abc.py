
from ..attribute.abc import Attribute, abstractmethod
# for relative imports in sub-packages:
from ..attribute import AnalyzerNode, Initializer

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

class TessellationPostProcessing(Attribute):
    @abstractmethod
    def post_process(self, tessellation, spt_dataframe):
        pass

