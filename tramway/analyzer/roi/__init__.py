
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
    __slots__ = ()
    def crop(self, df=None):
        return self._spt_data.dataframe if df is None else df


class DecentralizedROIManager(AnalyzerNode):
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
        self.specialize( BoundingBoxes, bb, label, group_overlapping_roi )
    def from_squares(self, centers, side, label=None, group_overlapping_roi=False):
        bb = [ (center-.5*side, center+.5*side) for center in centers ]
        self.from_bounding_boxes(bb, label, group_overlapping_roi)
    ## in the case no ROI are defined
    def as_support_regions(self, index=None, source=None, return_index=False):
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
        if collection is not None:
            warnings.warn('ignoring argument `collection`', helper.IgnoredInputWarning)
        return self.as_support_regions(index, source, **kwargs)

ROI.register(ROIInitializer)


class CommonROI(AnalyzerNode):
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

