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
    """
    Abstract base class for the :attr:`~tramway.analyzer.RWAnalyzer.spt_data` attribute
    of an :class:`~tramway.analyzer.RWAnalyzer` object.
    """
    @property
    @abstractmethod
    def columns(self):
        """
        *list* of *str*: Data column names
        """
        pass
    @property
    @abstractmethod
    def localization_error(self):
        r"""
        *float*: Localization error in :math:`\mu m^2`
        """
        pass
    @property
    @abstractmethod
    def frame_interval(self):
        """
        *float*: Time interval between successive frames, in seconds
        """
        pass
    @abstractmethod
    def reset_origin(self, columns=None, same_origin=False):
        """ Translates the lower bound of the spatial data to zero.
        """
        pass
    @abstractmethod
    def discard_static_trajectories(self, dataframe=None, min_msd=None):
        """ The minimum mean-square-displacement is set to the localization error per default. 
        
        If an input dataframe is given, then :meth:`discard_static_trajectories` returns the processed
        dataframe and `self` does not maintain any copy of it.
        
        Otherwise, :meth:`discard_static_trajectories` applies inplace to the internal dataframe
        and does not return any dataframe.
        Note that it may set a flag instead of actually processing the internal dataframe,
        for the parcimonious processing in regions of interest only."""
        pass
    @abstractmethod
    def __iter__(self):
        """ Returns an iterator on the :class:`~tramway.analyzer.spt_data.abc.SPTDataItem`
        objects as natively stored. """
        pass
    @abstractmethod
    def as_dataframes(self, source=None):
        """ Generator function; yields :class:`pandas.DataFrame` objects.
        
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
        """
        *pandas.DataFrame*: Lower (index *min*) and upper (index *max*) bounds for the SPT data
        """
        pass


class SPTDataItem(metaclass=ABCMeta):
    """
    Abstract base class for SPT data blocks an :class:`SPTData` object contains.
    """
    @property
    @abstractmethod
    def dataframe(self):
        """
        *pandas.DataFrame*: Raw SPT data
        """
        pass
    @property
    @abstractmethod
    def source(self):
        """
        *str*: File path or identifier
        """
        pass
    @property
    @abstractmethod
    def columns(self):
        pass
    columns.__doc__ = SPTData.columns.__doc__
    @property
    @abstractmethod
    def localization_error(self):
        pass
    localization_error.__doc__ = SPTData.localization_error.__doc__
    @property
    @abstractmethod
    def frame_interval(self):
        pass
    frame_interval.__doc__ = SPTData.frame_interval.__doc__
    @property
    @abstractmethod
    def bounds(self):
        pass
    bounds.__doc__ = SPTData.bounds.__doc__
    @property
    @abstractmethod
    def analyses(self):
        """
        *Analyses*: Analysis tree
        """
        pass

__all__ = [ 'SPTData', 'SPTDataItem' ]

