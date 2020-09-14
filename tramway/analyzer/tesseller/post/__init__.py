
from ..abc import Initializer
from . import merger as cell_mergers

class TessellationPostProcessingInitializer(Initializer):
    __slots__ = ()
    def from_callable(self, cls):
        """
        Argument:

            cls (callable): no-argument callable object (usually a class)
                that returns a :class:`TessellationPostProcessing` object.

        """
        self.specialize( cls )

