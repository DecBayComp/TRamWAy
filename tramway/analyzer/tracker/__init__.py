# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import *
from .abc import *


class NonTrackingTracker(AnalyzerNode):
    """ Non-tracking tracker.
    """
    __slots__ = ()
    def __init__(self, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
    def track(self, locations):
        raise NotImplementedError

Tracker.register(NonTrackingTracker)


class TrackerInitializer(Initializer):
    """ initializer class for the `RWAnalyzer.tracker` main analyzer attribute.

    The `RWAnalyzer.tacker` attribute self-modifies on calling *from_...* methods.

    """
    __slots__ = ()
    def from_non_tracking(self):
        """ *Non-tracking* tracker.
        
        See also :class:`NonTrackingTracker`."""
        self.specialize( NonTrackingTracker )


__all__ = [ 'Tracker', 'TrackerInitializer', 'NonTrackingTracker' ]

