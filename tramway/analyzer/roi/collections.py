
import tramway.helper.roi as helper
from tramway.helper.roi import IgnoredInputWarning, UnitRegions

"""
This transition module wraps some functionalities from the :mod:`tramway.helper.roi` module.
"""

def add_metadata(self, analysis, pkg_version=[]):
    return helper.Helper().add_metadata(analysis, pkg_version)

class Collections(helper.RoiCollections):
    def __init__(self, group_overlapping_roi=False, metadata=add_metadata, verbose=False):
        helper.RoiCollections.__init__(self, group_overlapping_roi, metadata=metadata, verbose=verbose)
        self._numeric_format = helper.RoiCollections.numeric_format.fget(self)
    @property
    def numeric_format(self):
        return self._numeric_format
    @numeric_format.setter
    def numeric_format(self, fmt):
        if isinstance(fmt, int):
            fmt = '{{:0>{:d}d}}'.format(fmt)
        self._numeric_format = fmt

