# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from abc import *
from ..attribute.abc import Attribute, abstractmethod

class SPTData(Attribute):
    @property
    @abstractmethod
    def columns(self):
        pass
    @property
    @abstractmethod
    def localization_error(self):
        pass
    @abstractmethod
    def reset_origin(self, columns=None, same_origin=False):
        pass
    @abstractmethod
    def discard_static_trajectories(self, dataframe=None, min_msd=None):
        """ the minimum mean-square-displacement is set to the localization error per default. 
        
        If an input dataframe is given, then `discard_static_trajectories` returns the processed
        dataframe and `self` does not maintain any copy of it.
        
        Otherwise, `discard_static_trajectories` applies inplace to the internal dataframe
        and does not return any dataframe.
        Note that it may set a flag instead of actually processing the internal dataframe,
        for the parcimonious processing in regions of interest only."""
        pass
    @abstractmethod
    def __iter__(self):
        """returns an iterator on the :class:`~.abc.SPTDataItem` objects as natively stored. """
        pass
    @abstractmethod
    def as_dataframes(self, source=None):
        """returns an iterator that yields :class:`pandas.DataFrame` objects.
        
        `source` can be a source name (filepath) or a boolean function
        that takes a source string as input argument."""
        if source is None:
            for f in self:
                with accessor(f, provides='dataframe') as a:
                    yield a.dataframe
        else:
            if callable(source):
                filter = source
            else:
                filter = lambda s: s == source
            for f in self:
                with accessor(f, provides=('source', 'dataframe')) as a:
                    if filter(a.source):
                        yield a.dataframe
    @property
    def bounds(self):
        pass


class SPTDataItem(metaclass=ABCMeta):
    @property
    @abstractmethod
    def dataframe(self):
        pass
    @property
    @abstractmethod
    def source(self):
        pass
    @property
    @abstractmethod
    def columns(self):
        pass
    @property
    @abstractmethod
    def bounds(self):
        pass
    @property
    @abstractmethod
    def analyses(self):
        pass

__all__ = [ 'SPTData', 'SPTDataItem' ]

