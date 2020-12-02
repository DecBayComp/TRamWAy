# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..abc import Initializer
from . import merger as cell_mergers

class TessellationPostProcessingInitializer(Initializer):
    """
    Initializer class for attribute :attr:`~tramway.analyzer.RWAnalyzer.tesseller`
    :attr:`~tramway.analyzer.tesseller.proxy.TessellerProxy.post`.

    For now, a single use case is available:

    .. code-block:: python

        a = RWAnalyzer()
        # initialize attribute `tesseller`
        a.tesseller = tessellers.KMeans
        # .. so that attribute `post` is exposed
        a.tesseller.post = cell_mergers.ByTranslocationCount
        # .. and now `tesseller.post` is initialized
        a.tesseller.post.count_threshold = 20

    Note that :mod:`~tramway.analyzer.cell_mergers` is exported by the
    :mod:`tramway.analyzer` module, just like :mod:`~tramway.analyzer.tesseller.tessellers`.

    See also :class:`~tramway.analyzer.tesseller.post.merger.ByTranslocationCount`.
    """
    __slots__ = ()
    def from_callable(self, cls):
        """
        Argument:

            cls (callable):
                no-argument callable object (usually a class) that returns a
                :class:`~tramway.analyzer.tesseller.TessellationPostProcessing` object.

        """
        self.specialize( cls )

