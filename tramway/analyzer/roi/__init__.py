# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from .abc import *
from ..attribute import *
from tramway.core.xyt import crop
from tramway.helper.base import HelperBase
import tramway.helper.roi as helper
import warnings
import numpy as np
from collections.abc import Sequence, Set


class BaseRegion(AnalyzerNode):
    __slots__ = ('_spt_data','_label')
    def __init__(self, spt_data, label=None, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._spt_data = spt_data
        self._label = label
    @property
    def label(self):
        if callable(self._label):
            self._label = self._label()
        return self._label
    @property
    def analyses(self):
        return self._spt_data.analyses[self.label]
    def add_sampling(self, sampling, label=None, comment=None):
        if callable(label):
            label = label(self.label)
        elif label is None:
            label = self.label
        return self._spt_data.add_sampling(sampling, label, comment)
    def autosaving(self, *args, **kwargs):
        return self._spt_data.autosaving(*args, **kwargs)
    def discard_static_trajectories(self, df):
        return self._spt_data.discard_static_trajectories(df)

class IndividualROI(BaseRegion):
    """ for typing only """
    __slots__ = ()
    pass

class BoundingBox(IndividualROI):
    """ See :class:`BoundingBoxes`. """
    __slots__ = ('_bounding_box',)
    def __init__(self, bb, label, spt_data, **kwargs):
        IndividualROI.__init__(self, spt_data, label, **kwargs)
        self._bounding_box = bb
    def crop(self, df=None):
        _min,_max = self._bounding_box
        if df is None:
            df = self._spt_data.dataframe
        return crop(df, np.r_[_min, _max-_min])
    @property
    def bounding_box(self):
        return self._bounding_box

class SupportRegion(BaseRegion):
    """
    union of overlapping ROI.
    """
    __slots__ = ('_sr_index','_support_regions')
    def __init__(self, r, regions, spt_data, **kwargs):
        BaseRegion.__init__(self,
                spt_data,
                r if isinstance(r, str) else regions.region_label(r),
                **kwargs)
        self._sr_index = r
        self._support_regions = regions
    def crop(self, df=None):
        if df is None:
            df = self._spt_data.dataframe
        return self._support_regions.crop(self._sr_index, df)
    @property
    def bounding_box(self):
        if isinstance(self._support_regions, helper.UnitRegions):
            return self._support_regions[self._sr_index]
        else:
            minima, maxima = zip(*[ self._support_regions.unit_region[u] \
                for u in self._support_regions[self._sr_index] ])
            return np.min(np.stack(minima, axis=0), axis=0), np.max(np.stack(maxima, axis=0), axis=0)

class FullRegion(BaseRegion):
    """
    wraps the full dataset; does not actually crop.

    A `FullRegion` can be both an individual ROI and a support region.
    """
    __slots__ = ()
    def crop(self, df=None):
        return self._spt_data.dataframe if df is None else df


class DecentralizedROIManager(AnalyzerNode):
    """
    This class allows to iterate over the ROI defined at the level of
    each SPT data item.
    """
    __slots__ = ('_records',)
    def __init__(self, first_record=None, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._records = set()
        if first_record is not None:
            self._register_decentralized_roi(first_record)
    def _register_decentralized_roi(self, has_roi):
        self._records.add(has_roi)
        has_roi.roi._global = self
    def _update_decentralized_roi(self, known_record, new_record):
        self._records.remove(known_record)
        self._register_decentralized_roi(new_record)
    def reset(self):
        self._records = set()
    def self_update(self, op):
        raise NotImplementedError('why for?')
        self._parent._roi = op(self)
    def as_individual_roi(self, index=None, collection=None, source=None, **kwargs):
        """ returns a generator to loop over all the individual roi.

        Filtering is delegated to the individual *.spt_data.roi* attributes.

        A *callable* filter takes a single key (*int* for indices, *str* for labels and paths)
        and returns a *bool*.
        
        Arguments:
            
            index (*int*, *sequence* of *int*s, or *callable*):
                individual ROI index filter; indices apply within a collection
                
            collection (*str*, *sequence* of *str*s, or *callable*):
                collection label filter
            
            source (*str*, *sequence* of *str*s, or *callable*):
                SPT data source filter
                
        Returns:
            
            generator: iterator for individual ROI
            
        """
        if source is None:
            for rec in self._records:
                for roi in rec.roi.as_individual_roi(index, collection, **kwargs):
                    yield roi
        else:
            if callable(source):
                sfilter = source
            else:
                sfilter = lambda s: s==source
            for rec in self._records:
                if sfilter(rec.source):
                    for roi in rec.roi.as_individual_roi(index, collection, **kwargs):
                        yield roi
    def as_support_regions(self, index=None, source=None, **kwargs):
        """ returns a generator to loop over all the support regions.

        Support regions are equivalent to individual ROI if *group_overlapping_roi*
        was set to ``False``.

        Filtering is delegated to the individual *.spt_data.roi* attributes.

        A *callable* filter takes a single key (*int* for indices, *str* for paths)
        and returns a *bool*.
        
        Arguments:
            
            index (*int*, *sequence* of *int*s, or *callable*):
                support region index filter
            
            source (*str*, *sequence* of *str*s, or *callable*):
                SPT data source filter
                
        Returns:
            
            generator: iterator for support regions
            
        """
        if source is None:
            for rec in self._records:
                for reg in rec.roi.as_support_regions(index, **kwargs):
                    yield reg
        else:
            if callable(source):
                sfilter = source
            else:
                sfilter = lambda s: s==source
            for rec in self._records:
                if sfilter(rec.source):
                    for reg in rec.roi.as_support_regions(index, **kwargs):
                        yield reg

ROI.register(DecentralizedROIManager)


class ROIInitializer(Initializer):
    """
    initial value for the `RWAnalyzer.roi` attribute.

    `from_...` methods alters the parent attribute which specializes
    into an initialized :class:`.abc.ROI` object.
    """
    __slots__ = ()
    def specialize(self, cls, *args, **kwargs):
        Initializer.specialize(self, cls, *args, **kwargs)
        if self._parent is self._eldest_parent and not issubclass(cls, DecentralizedROIManager):
            # replace all individual-SPT-item-level roi initializers by mirrors
            spt_data, roi = self._parent.spt_data, self._parent.roi
            if not spt_data.initialized:
                raise RuntimeError('cannot define ROI as long as the `spt_data` attribute is not initialized')
            for f in spt_data:
                assert isinstance(f, HasROI)
                if f.roi.initialized:
                    raise RuntimeError('ROI already defined at the individual SPT data item level')
                f.roi._from_common_roi(roi)
    ## initializers
    def _from_common_roi(self, roi):
        # special `specialize`
        spt_data = self._parent
        spt_data._roi = spt_data._bear_child( CommonROI, roi )
    def _register_decentralized_roi(self, roi):
        self.specialize( DecentralizedROIManager, roi )
    def from_bounding_boxes(self, bb, label=None, group_overlapping_roi=False):
        """
        Defines ROI as bounding boxes.

        Arguments:

            bb (sequence): collection of bounding boxes, each bounding boxes being
                a pair of lower and upper bounds (*numpy.ndarray*s)

            label (str): unique label for the collection

            group_overlapping_roi (bool): if ``False``, `as_support_regions` will
               behave similarly to `as_individual_roi`, otherwise support regions
               are unions of overlapping ROI

        See also :class:`BoundingBoxes`.
        """
        self.specialize( BoundingBoxes, bb, label, group_overlapping_roi )
    def from_squares(self, centers, side, label=None, group_overlapping_roi=False):
        """
        Defines ROI as centers for squares/cubes of uniform size.

        See also `from_bounding_boxes`.
        """
        bb = [ (center-.5*side, center+.5*side) for center in centers ]
        self.from_bounding_boxes(bb, label, group_overlapping_roi)
    ## in the case no ROI are defined
    def as_support_regions(self, index=None, source=None, return_index=False):
        """ returns a generator to loop over all the support regions.
        
        A `ROIInitializer` does not define any ROI,
        as a consequence a single `FullRegion` object is generated."""
        if not null_index(index):
            raise ValueError('no ROI defined; cannot seek for the ith ROI')
        if return_index:
            def bear_child(*args):
                return 0, self._bear_child(*args)
        else:
            bear_child = self._bear_child
        try:
            spt_data = self._parent.spt_data
        except AttributeError:
            # decentralized roi (single source)
            if source is not None:
                warnings.warn('ignoring argument `source`', helper.IgnoredInputWarning)
            spt_data = self._parent
            yield bear_child( FullRegion, spt_data )
        else:
            # roi manager (multiple sources)
            if isinstance(spt_data, Initializer):
                raise RuntimeError('cannot iterate not-initialized SPT data')
            if source is None:
                for d in spt_data:
                    yield bear_child( FullRegion, d )
            else:
                if callable(source):
                    sfilter = source
                else:
                    sfilter = lambda s: s==source
                for d in spt_data:
                    if sfilter(d.source):
                        yield bear_child( FullRegion, d )
    def as_individual_roi(self, index=None, collection=None, source=None, **kwargs):
        """ returns a generator to loop over all the individual roi.
        
        A `ROIInitializer` does not define any ROI,
        as a consequence a single `FullRegion` object is generated."""
        if collection is not None:
            warnings.warn('ignoring argument `collection`', helper.IgnoredInputWarning)
        return self.as_support_regions(index, source, **kwargs)

ROI.register(ROIInitializer)


class CommonROI(AnalyzerNode):
    """
    Mirrors the global `RWAnalyzer.roi` attribute.

    The individual `.spt_data.roi` attributes become `CommonROI` objects
    as soon as the global `.roi` attribute is specialized,
    so that `as_support_regions` and `as_individual_roi` iterators delegate
    to the global attribute.
    """
    __slots__ = ('_global',)
    def __init__(self, roi, parent=None):
        AnalyzerNode.__init__(self, parent)
        self._global = roi
    def as_support_regions(self, index=None, source=None, return_index=False):
        spt_data = self._parent
        if not spt_data.compatible_source(source):
            warnings.warn('ignoring argument `source`', helper.IgnoredInputWarning)
            return
        yield from self._global.as_support_regions(index, spt_data.source, return_index)
    def self_update(self, op):
        raise RuntimeError('cannot alter a mirror attribute')
    def as_individual_roi(self, index=None, collection=None, source=None, return_index=False):
        spt_data = self._parent
        if not spt_data.compatible_source(source):
            warnings.warn('ignoring argument `source`', helper.IgnoredInputWarning)
            return
        yield from self._global.as_individual_roi(index, collection, spt_data.source, return_index)


class SpecializedROI(AnalyzerNode):
    """
    Base class for initialized *roi* attributes.
    """
    __slots__ = ('_global','_collections')
    def __init__(self, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._global = None
        self._collections = None
    def self_update(self, op):
        self._parent._roi = op(self)
        assert self._global is not None
        if self._global is not None:
            # parent spt_data object should still be registered
            assert self._parent in self._global._records
    def as_support_regions(self, index=None, source=None, return_index=False):
        if return_index:
            def bear_child(cls, r, *args):
                i, r = r
                return i, self._bear_child(cls, r, *args)
            kwargs = dict(return_index=return_index)
        else:
            bear_child = self._bear_child
            kwargs = {}
        try:
            spt_data = self._parent.spt_data
        except AttributeError:
            # decentralized roi (single source)
            if source is not None:
                warnings.warn('ignoring argument `source`', helper.IgnoredInputWarning)
            spt_data = self._parent
            for r in indexer(index, self._collections.regions, **kwargs):
                yield bear_child( SupportRegion, r, self._collections.regions, spt_data )
        else:
            # roi manager (one set of regions, multiple sources)
            if isinstance(spt_data, Initializer):
                raise RuntimeError('cannot iterate not-initialized SPT data')
            if source is None:
                for d in spt_data:
                    for r in indexer(index, self._collections.regions, **kwargs):
                        yield bear_child( SupportRegion, r, self._collections.regions, d )
            else:
                if callable(source):
                    sfilter = source
                else:
                    sfilter = lambda s: s==source
                for d in spt_data:
                    if sfilter(d.source):
                        for r in indexer(index, self._collections.regions, **kwargs):
                            yield bear_child( SupportRegion, r, self._collections.regions, d )

class BoundingBoxes(SpecializedROI):
    """
    Bounding boxes are a list of pairs of NumPy row arrays (1xD).

    The first array specifies the lower bounds, the second array the upper bounds.
    """
    __slots__ = ('_bounding_boxes',)
    def __init__(self, bb, label=None, group_overlapping_roi=False, **kwargs):
        SpecializedROI.__init__(self, **kwargs)
        self._collections = helper.RoiCollections(
                metadata=helper.Helper().add_metadata,
                group_overlapping_roi=group_overlapping_roi,
                verbose=False)
        if label is None:
            label = ''
        self._collections[label] = bb
        self._bounding_boxes = {label: bb}
    @property
    def bounding_boxes(self):
        return self._bounding_boxes
    def as_individual_roi(self, index=None, collection=None, source=None, return_index=False):
        if return_index:
            def bear_child(i, *args):
                return i, self._bear_child(*args)
        else:
            def bear_child(i, *args):
                return self._bear_child(*args)
        try:
            spt_data = self._parent.spt_data
        except AttributeError:
            # decentralized roi (single source)
            if source is not None:
                warnings.warn('ignoring argument `source`', helper.IgnoredInputWarning)
            spt_data = self._parent
            for label in indexer(collection, self.bounding_boxes):
                for i, bb in indexer(index, self.bounding_boxes[label], return_index=True):
                    roi_label = self._collections[label].roi_label(i)
                    yield bear_child(i, BoundingBox, bb, roi_label, spt_data )
        else:
            for d in spt_data:
                for label in indexer(collection, self.bounding_boxes):
                    for i, bb in indexer(index, self.bounding_boxes[label], return_index=True):
                        roi_label = self._collections[label].roi_label(i)
                        yield bear_child(i, BoundingBox, bb, roi_label, d )

ROI.register(BoundingBoxes)


class HasROI(AnalyzerNode):
    """ Class to be inherited from by SPT data item classes.
    
    Maintains a self-modifying *roi* attribute."""
    __slots__ = ('_roi',)
    def _get_roi(self):
        return self._roi
    def _set_roi(self, roi):
        self._roi = roi
        global_roi_attr = self._eldest_parent.roi
        if global_roi_attr.initialized:
            assert isinstance(global_roi_attr, DecentralizedROIManager)
        global_roi_attr._register_decentralized_roi(self)
    roi = selfinitializing_property('roi', _get_roi, _set_roi, ROI)

    def __init__(self, roi=ROIInitializer, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._roi = roi(self._set_roi, parent=self)
    def compatible_source(self, source):
        """
        returns ``True`` if filter *source* matches with `self.source`.
        """
        if source is None:
            return True
        elif callable(source):
            return source(spt_data.source)
        elif isinstance(source, str):
            return source == spt_data.source
        elif isinstance(source, (Set, Sequence)):
            return spt_data.source in source
        else:
            raise NotImplementedError

