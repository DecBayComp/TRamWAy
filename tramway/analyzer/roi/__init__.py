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
from . import collections as helper
import warnings
import numpy as np
import pandas as pd
from collections.abc import Sequence, Set
from tramway.tessellation.base import Partition
from rwa.lazy import lazytype


class BaseRegion(AnalyzerNode):
    """
    This class should not be directly instanciated.
    It brings basic functionalities to all the available representations
    of regions of interest, merely labelling and analysis registration.
    """
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
    def source(self):
        return self._spt_data.source
    def get_sampling(self, label=None, unique=True, _type=Partition):
        if callable(label):
            label = label(self.label)
        elif label is None:
            label = self.label
        if isinstance(label, str):
            labels = [label]
        else:
            labels = list(label)
        analyses = [ self._spt_data.get_sampling(_label) for _label in labels ]
        if _type:
            analyses = [ a for a in analyses if issubclass(lazytype(a._data), _type) ]
        if unique:
            if analyses[1:]:
                raise ValueError('label is not specific enough; multiple samplings match')
            elif not analyses:
                raise KeyError("could not find label '{}'".format(label))
            return analyses[0]
        else:
            return analyses
    def add_sampling(self, sampling, label=None, comment=None):
        if callable(label):
            label = label(self.label)
        elif label is None:
            label = self.label
        return self._spt_data.add_sampling(sampling, label, comment)
    def autosaving(self, *args, **kwargs):
        return self._spt_data.autosaving(*args, **kwargs)
    def discard_static_trajectories(self, df, **kwargs):
        return self._spt_data.discard_static_trajectories(df, **kwargs)
    @property
    def _mpl_impl(self):
        from .mpl import Mpl
        return Mpl
    @property
    def mpl(self):
        return self._mpl_impl(self)

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
        n_space_cols = len([ col for col in 'xyz' if col in df.columns ])
        if n_space_cols < _min.size:
            assert _min.size == n_space_cols + 1
            df = df[(_min[-1] <= df['t']) & (df['t'] <= _max[-1])]
            _min, _max = _min[:-1], _max[:-1]
        df = crop(df, np.r_[_min, _max-_min])
        return df
    @property
    def bounding_box(self):
        return self._bounding_box
    def crop_frames(self, **kwargs):
        """
        Iterates and crops the image frames.

        `kwargs` are passed to the :class:`~tramway.analyzer.images.Images`
        :meth:`~tramway.analyzer.images.Images.crop_frames` method.
        """
        yield from self._spt_data.get_image().crop_frames(self.bounding_box, **kwargs)

class SupportRegion(BaseRegion):
    """
    Union of overlapping ROI.
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
        df = self._support_regions.crop(self._sr_index, df)
        return df
    @property
    def bounding_box(self):
        if isinstance(self._support_regions, helper.UnitRegions):
            return self._support_regions[self._sr_index]
        else:
            minima, maxima = zip(*[ self._support_regions.unit_region[u] \
                for u in self._support_regions[self._sr_index] ])
            return np.min(np.stack(minima, axis=0), axis=0), np.max(np.stack(maxima, axis=0), axis=0)
    def crop_frames(self, **kwargs):
        """
        Iterates and crops the image frames, based on :attr:`bounding_box`.

        `kwargs` are passed to the :class:`~tramway.analyzer.images.Images`
        :meth:`~tramway.analyzer.images.Images.crop_frames` method.
        """
        yield from BoundingBox.crop_frames(self, **kwargs)

class FullRegion(BaseRegion):
    """
    Wraps the full dataset; does not actually crop.

    A `FullRegion` can be both an individual ROI and a support region.
    """
    __slots__ = ()
    def crop(self, df=None):
        if df is None:
            df = self._spt_data.dataframe
        return df
    def crop_frames(self, **kwargs):
        """
        .. note::

            Time cropping is not supported yet.

        """
        if self.time_support is not None:
            self._eldest_parent.logger.warning('time cropping is not supported yet')
        yield from self._spt_data.as_frames(**kwargs)


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
        if isinstance(has_roi.roi, ROIInitializer):
            self._parent.logger.warning('cannot register an uninitialized ROI attribute')
            return
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
        """ Generator function; loops over all the individual roi.

        Filtering is delegated to the individual *SPTDataItem.roi* attributes.

        A *callable* filter takes a single key (*int* for indices, *str* for labels and paths)
        and returns a *bool*.
        
        Arguments:
            
            index (*int*, *set* of *int*, *sequence* of *int*, or *callable*):
                individual ROI index filter; indices apply within a collection
                
            collection (*str*, *set* of *str*, *sequence* of *str*, or *callable*):
                collection label filter
            
            source (*str*, *set* of *str*, *sequence* of *str*, or *callable*):
                SPT data source filter
            
        """
        if source is None:
            for rec in self._records:
                yield from rec.roi.as_individual_roi(index, collection, **kwargs)
        else:
            if callable(source):
                sfilter = source
            else:
                sfilter = lambda s: s==source
            for rec in self._records:
                if sfilter(rec.source):
                    yield from rec.roi.as_individual_roi(index, collection, **kwargs)
    def as_support_regions(self, index=None, source=None, **kwargs):
        """ Generator function; loops over all the support regions.

        Support regions are equivalent to individual ROI if *group_overlapping_roi*
        was set to :const:`False`.

        Filtering is delegated to the individual SPT data block
        :attr:`~tramway.analyzer.spt_data.SPTDataItem.roi` attributes.

        A *callable* filter takes a single key (*int* for indices, *str* for paths)
        and returns a *bool*.
        
        Arguments:
            
            index (*int*, *set* of *int*, *sequence* of *int*, or *callable*):
                support region index filter
            
            source (*str*, *set* of *str*, *sequence* of *str*, or *callable*):
                SPT data source filter
                
        """
        if source is None:
            for rec in self._records:
                yield from rec.roi.as_support_regions(index, **kwargs)
        else:
            if callable(source):
                sfilter = source
            else:
                sfilter = lambda s: s==source
            for rec in self._records:
                if sfilter(rec.source):
                    yield from rec.roi.as_support_regions(index, **kwargs)
    def __iter__(self):
        raise AttributeError(type(self).__name__+' object is not iterable; call methods as_support_regions() or as_individual_roi()')

ROI.register(DecentralizedROIManager)


class ROIInitializer(Initializer):
    """
    Initial value for the :class:`~tramway.analyzer.RWAnalyzer`
    :attr:`~tramway.analyzer.RWAnalyzer.roi` attribute.

    *from_...* methods alters the parent attribute which specializes
    into an initialized :class:`ROI` object.
    """
    __slots__ = ()
    def specialize(self, cls, *args, **kwargs):
        Initializer.specialize(self, cls, *args, **kwargs)
        if self._parent is self._eldest_parent:
            if issubclass(cls, DecentralizedROIManager):
                if cls is not DecentralizedROIManager:
                    # post-register decentralized roi again if necessary
                    spt_data, roi = self._parent.spt_data, self._parent.roi
                    for f in spt_data:
                        if isinstance(f, HasROI) and f.roi.initialized:
                            roi._register_decentralized_roi(f)
            else:
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
                a pair of lower and upper bounds (*numpy.ndarray*)

            label (str): unique label for the collection

            group_overlapping_roi (bool): if :const:`False`, :meth:`as_support_regions`
                will behave similarly to :meth:`as_individual_roi`, otherwise support
                regions are unions of overlapping ROI

        See also :class:`BoundingBoxes`.
        """
        self.specialize( BoundingBoxes, bb, label, group_overlapping_roi )
    def from_squares(self, centers, side, label=None, group_overlapping_roi=False):
        """
        Defines ROI as centers for squares/cubes of uniform size.

        See also :meth:`from_bounding_boxes`.
        """
        bb = [ (center-.5*side, center+.5*side) for center in centers ]
        self.from_bounding_boxes(bb, label, group_overlapping_roi)
    def from_ascii_file(self, filepath, size=None,
            label=None, group_overlapping_roi=False):
        """
        Reads the ROI centers or bounds from a text file.

        .. note::

            This initializer can only be called from the decentralized
            :attr:`~HasROI.roi` attribute of an
            :class:`~tramway.analyzer.spt_data.SPTDataItem` item of the
            :class:`~tramway.analyzer.RWAnalyzer`
            :attr:`~tramway.analyzer.RWAnalyzer.spt_data` main attribute.

        See also :class:`ROIAsciiFile`.
        """
        if isinstance(self._parent, HasROI):
            self.specialize( ROIAsciiFile, filepath, size, label, group_overlapping_roi )
        else:
            raise AttributeError('from_ascii_file called for the main roi attribute')
    def from_ascii_files(self, suffix='roi', extension='.txt', size=None,
            label=None, group_overlapping_roi=False, skip_missing=False):
        """
        Reads the ROI centers or bounds from text files -- one file per item in
        :attr:`~tramway.analyzer.RWAnalyzer.spt_data`.

        The file paths are inferred from the SPT data source files (`source` attribute).
        Files are looked for in the same directories as the corresponding SPT data file,
        and are expected to be named with the same basename, plus a suffix and the *.txt*
        extension.

        A hyphen or underscore character is automatically appended left of the suffix if
        necessary.

        If the `source` attribute of the :class:`~tramway.analyzer.spt_data.SPTDataItem`
        items is not defined, a :class:`ValueError` exception is raised.

        .. note::

            Calling this initializer method from a satellite `roi` attribute
            (nested in an :class:`~tramway.analyzer.spt_data.SPTDataItem` data block)
            is equivalent to calling the same initializer from the
            :class:`~tramway.analyzer.RWAnalyzer`
            :attr:`~tramway.analyzer.RWAnalyzer.roi` main attribute.

        See also :class:`ROIAsciiFiles`.
        """
        self._eldest_parent.roi.specialize( ROIAsciiFiles, suffix, extension,
                size, label, group_overlapping_roi, skip_missing )
    def from_dedicated_rwa_record(self, label=None, version=None, _impl=None):
        """
        See also :class:`v1_ROIRecord`.
        """
        if isinstance(self._parent, HasROI):
            if version is None:
                self._parent.logger.info('set version=1 to ensure constant behavior in the future')
                version = 1
            elif version != 1:
                raise ValueError('version {} not supported'.format(version))
            self.specialize( v1_ROIRecord, label, _impl )
        else:
            self.from_dedicated_rwa_records(label, version, _impl=_impl)
    def from_dedicated_rwa_records(self, label=None, version=None, _impl=None):
        """
        See also :class:`ROIRecords`.
        """
        self._eldest_parent.roi.specialize( ROIRecords, label, version, _impl )
    ## in the case no ROI are defined
    def as_support_regions(self, index=None, source=None, return_index=False):
        """ Generator function; loops over all the support regions.
        
        A :class:`ROIInitializer` attribute object does not define any ROI,
        as a consequence a single :class:`FullRegion` object is yielded."""
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
        """ Generator function; loops over all the individual ROI.
        
        A :class:`ROIInitializer` does not define any ROI,
        as a consequence a single :class:`FullRegion` object is yielded."""
        if collection is not None:
            warnings.warn('ignoring argument `collection`', helper.IgnoredInputWarning)
        return self.as_support_regions(index, source, **kwargs)
    def __iter__(self):
        raise AttributeError(type(self).__name__+' object is not iterable; call methods as_support_regions() or as_individual_roi()')

ROI.register(ROIInitializer)


class CommonROI(AnalyzerNode):
    """
    Mirrors the global :attr:`~tramway.analyzer.RWAnalyzer.roi` attribute.

    The individual *SPTDataItem.roi* attributes become :class:`CommonROI` objects
    as soon as the global :attr:`~tramway.analyzer.RWAnalyzer.roi` attribute is
    specialized, so that :meth:`as_support_regions` and :meth:`as_individual_roi`
    iterators delegate to the global attribute.
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
    def __iter__(self):
        raise AttributeError(type(self).__name__+' object is not iterable; call methods as_support_regions() or as_individual_roi()')


class SpecializedROI(AnalyzerNode):
    """
    Basis for initialized :class:`ROI` classes.
    """
    __slots__ = ('_global','_collections')
    def __init__(self, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._global = None
        self._collections = None
    def self_update(self, op):
        self._parent._roi = op(self)
        if self._global is not None:
            # parent spt_data object should still be registered
            assert self._parent in self._global._records
    def as_support_regions(self, index=None, source=None, return_index=False):
        if return_index:
            def bear_child(cls, r, *args):
                return r, self._bear_child(cls, r, *args)
        else:
            bear_child = self._bear_child
        try:
            spt_data = self._parent.spt_data
        except AttributeError:
            # decentralized roi (single source)
            if source is not None:
                warnings.warn('ignoring argument `source`', helper.IgnoredInputWarning)
            spt_data = self._parent
            for r, _ in indexer(index, self._collections.regions, has_keys=True, return_index=True):
                yield bear_child( SupportRegion, r, self._collections.regions, spt_data )
        else:
            # roi manager (one set of regions, multiple sources)
            if isinstance(spt_data, Initializer):
                raise RuntimeError('cannot iterate not-initialized SPT data')
            if source is None:
                for d in spt_data:
                    for r, _ in indexer(index, self._collections.regions, has_keys=True, return_index=True):
                        yield bear_child( SupportRegion, r, self._collections.regions, d )
            else:
                if callable(source):
                    sfilter = source
                else:
                    sfilter = lambda s: s==source
                for d in spt_data:
                    if sfilter(d.source):
                        for r, _ in indexer(index, self._collections.regions, has_keys=True, return_index=True):
                            yield bear_child( SupportRegion, r, self._collections.regions, d )
    as_support_regions.__doc__ = ROI.as_support_regions.__doc__
    def __iter__(self):
        raise AttributeError(type(self).__name__+' object is not iterable; call methods as_support_regions() or as_individual_roi()')
    def get_support_region(self, index, collection=None):
        """
        Returns the :class:`SupportRegion` object corresponding to an individual ROI.

        .. note::

            When using individual ROI indices and the parallelization capability of
            :attr:`~tramway.analyzer.RWAnalyzer.pipeline`,
            beware that :meth:`~ROI.as_individual_roi` is not controlled by the proper
            filters, unlike :meth:`~ROI.as_support_regions`, in all the possible
            implementations.
            :meth:`get_support_region` may still be called on workers assigned to
            processing other regions and consequently raise a :class:`RuntimeError`
            exception.

        """
        sr_index = self._collections.regions.unit_to_region(index, collection)
        sr = single(self.as_support_regions(index=sr_index))
        assert sr._sr_index == sr_index
        return sr


class BoundingBoxes(SpecializedROI):
    """
    Bounding boxes are a list of pairs of NumPy row arrays (1xD).

    The first array specifies the lower bounds, the second array the upper bounds.
    """
    __slots__ = ('_bounding_boxes',)
    def __init__(self, bb, label=None, group_overlapping_roi=False, **kwargs):
        SpecializedROI.__init__(self, **kwargs)
        self._collections = helper.Collections(group_overlapping_roi)
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
    as_individual_roi.__doc__ = ROI.as_individual_roi.__doc__
    @property
    def index_format(self):
        """
        *str*: Format of the numeric part of the label
        """
        return self._collections.numeric_format
    @index_format.setter
    def index_format(self, fmt):
        self._collections.numeric_format = fmt
    def set_num_digits(self, n):
        """
        Sets the number of digits in the numeric part of the label.
        """
        if not isinstance(n, int):
            raise TypeError('num_digits is not an int')
        self.index_format = n
    def to_ascii_file(self, filepath, collection=None, columns=None, header=True,
            float_format='%.4f', **kwargs):
        """
        Exports the bounding boxes to text file.

        Arguments:

            filepath (str): output filepath.

            collection (str): roi collection label.

            columns (sequence of *str*): column names; default is:
                ['x', 'y'] for 2D, ['x', 'y', 't'] for 3D
                and ['x', 'y', 'z', 't'] for 4D.

            header (bool): print column names on the first line.

            float_format (str): see also :meth:`pandas.DataFrame.to_csv`.

        Additional keyword arguments are passed to :meth:`pandas.DataFrame.to_csv`.
        """
        if collection is None:
            collection = ''
        example_bound, _ = first(self.bounding_boxes[collection])
        if columns is None:
            columns = {
                    2: list('xy'),
                    3: list('xyt'),
                    4: list('xyzt'),
                }[len(example_bound)]
        bounds = np.stack([ np.r_[bounds] for bounds in self.bounding_boxes[collection] ], axis=0)
        actual_columns = []
        for col in columns:
            actual_columns.append(col+' min')
        for col in columns:
            actual_columns.append(col+' max')
        bounds = pd.DataFrame(bounds, columns=actual_columns)
        for arg in ('sep', 'index'):
            try:
                kwargs.pop(arg)
            except KeyError:
                pass
            else:
                self._eldest_parent.logger.warning("ignoring argument '{}'".format(arg))
        bounds.to_csv(filepath, sep='\t', index=False, header=header, float_format=float_format,
                **kwargs)

ROI.register(BoundingBoxes)


class ROIAsciiFile(BoundingBoxes):
    """
    :class:`ROI` class for the decentralized :attr:`HasROI.roi` attributes
    to be loaded from text files.

    A ROI file contains tab-separated columns with a header line.

    The columns represent either centers or bounds.

    ROI center information is formed by :const:`'x'` and :const:`'y'` columns
    (and optionally :const:`'z'` but not :const:`'t'`).
    
    ROI bounds are defined by two columns for each coordinate, for example
    labelled :const:`'x min'` and :const:`'x max'` for coordinate *x*.
    Time can also be represented this way.

    Note that combining spatial center information and time bounds is allowed,
    i.e. :const:`'x'`, :const:`'y'`, :const:`'t min'` and :const:`'t max'`.

    The center information from a ROI file must be complemented with
    the `size` argument/attribute.
    """
    __slots__ = ('_path', '_size')
    def __init__(self, path, size=None, label=None,
            group_overlapping_roi=False, **kwargs):
        SpecializedROI.__init__(self, **kwargs) # not BoundingBoxes.__init__
        self._collections = helper.Collections(group_overlapping_roi)
        if label is None:
            label = ''
        self._bounding_boxes = {label: None}
        self._path = path
        self._size = size
    @property
    def reified(self):
        """ *bool*: :const:`True` if the files have been loaded """
        return not all([ bb is None for bb in self._bounding_boxes.values() ])
    @property
    def filepath(self):
        """ *str*: Path of the ROI file """
        return self._path
    @property
    def size(self):
        """ *float*: ROI size for space components; apply solely to center-defined ROI """
        return self._size
    @size.setter
    def size(self, sz):
        if self.reified:
            raise AttributeError("file '{}' has already been loaded; cannot modify the ROI size anymore".format(self.filepath.split('/')[-1]))
        else:
            self._size = sz
    @property
    def bounding_boxes(self):
        # this is enough to make `as_individual_roi` properly work
        if not self.reified:
            self.load()
        return self._bounding_boxes
    def as_support_regions(self, *args, **kwargs):
        if not self.reified:
            self.load()
        yield from BoundingBoxes.as_support_regions(self, *args, **kwargs)
    def load(self):
        roi = pd.read_csv(self.filepath, sep='\t')
        center_cols = [ col for col in 'xyzt' if col in roi.columns ]
        bound_cols = [ col for col in 'xyzt' if col+' min' in roi.columns and col+' max' in roi.columns ]
        if 't' in center_cols:
            raise ValueError('cannot interprete a ROI center time')
        if bound_cols:
            if center_cols:
                if 't' in bound_cols and not any([ col in bound_cols for col in 'xyz' ]):
                    # only case allowed
                    coords = [ col in center_cols for col in 'xyz' ]
                    if coords:
                        lower_bounds = (roi[coords] - .5 * self.size).join(roi[['t min']]).values
                        upper_bounds = (roi[coords] + .5 * self.size).join(roi[['t max']]).values
                    else:
                        lower_bounds = roi[['t min']].values
                        upper_bounds = roi[['t max']].values
                        if self._size is not None:
                            self._eldest_parent.logger.debug('ROI size is defined but no spatial coordinates were found')
                else:
                    raise ValueError('center and bound information both available at the same time')
            else:
                coords = [ col in bound_cols for col in 'xyz' ]
                # ordering from bound_cols matters
                lower_bounds = roi[[ col+' min' for col in bound_cols ]].values
                upper_bounds = roi[[ col+' max' for col in bound_cols ]].values
                if self._size is not None:
                    self._eldest_parent.logger.debug('ROI size does not apply to bounds-defined regions')
        else:
            coords = center_cols
            lower_bounds = roi[coords].values - .5 * self.size
            upper_bounds = roi[coords].values + .5 * self.size
        # last check
        if 'x' in coords:
            if 'y' not in coords:
                raise ValueError('x coordinate found but could not find y coordinate')
        elif 'y' in coords:
            raise ValueError('y coordinate found but could not find x coordinate')
        #
        bounds = list(zip(lower_bounds, upper_bounds))
        #
        for label in self._bounding_boxes:
            if self._bounding_boxes[label] is None:
                self._collections[label] = bounds
                self._bounding_boxes[label] = bounds
                #break?

ROI.register(ROIAsciiFile)


class ROIAsciiFiles(DecentralizedROIManager):
    """
    :class:`ROI` class for multiple ROI text files.

    The filepaths are inferred from the :attr:`~..spt_data.SPTDataItem.source`
    attribute of each :class:`~..spt_data.SPTDataItem` in the main
    :attr:`~tramway.analyzer.RWAnalyzer.spt_data` attribute.
    The SPT file extensions are replaced by a suffix (usually *-roi*) plus
    the *.txt* extension.

    See also :class:`ROIAsciiFile` for more information on the format.
    """
    __slots__ = ('_suffix',)
    def __init__(self, suffix='roi', extension='.txt', side=None, label=None,
            group_overlapping_roi=False, skip_missing=False, **kwargs):
        DecentralizedROIManager.__init__(self, **kwargs)
        if not isinstance(suffix, str):
            raise TypeError('suffix is not an str')
        self._suffix = suffix+extension
        # TODO: initialize the decentralized ROI so that _list_files is triggered later
        self._list_files(side, label, group_overlapping_roi, skip_missing)
    def _list_files(self, *args):
        args, skip_missing = args[:-1], args[-1]
        import os.path
        first = True
        for f in self._parent.spt_data:
            if f.source is None or not os.path.isfile(f.source):
                self._parent.logger.warning('cannot identify or find SPT data source: {}'.format(f.source))
                continue
            filepath, _ = os.path.splitext(f.source)
            if first:
                found = False
                for join_with in ('', '-', '_'):
                    candidate_filepath = join_with.join((filepath, self._suffix))
                    if os.path.isfile(os.path.expanduser(candidate_filepath)):
                        found = True
                        filepath = candidate_filepath
                        self._suffix = join_with + self._suffix
                        break
                if not found:
                    if skip_missing:
                        self._parent.logger.info('skipping roi file for source: '+str(f.source))
                        continue
                    raise FileNotFoundError('{}{}{}'.format(
                            os.path.basename(filepath),
                            '' if self._suffix[0] in '-_' else '[-_]',
                            self._suffix,
                        ))
                first = False
            else:
                filepath = filepath + self._suffix
                if not os.path.isfile(os.path.expanduser(filepath)):
                    if skip_missing:
                        self._parent.logger.info('skipping roi file for source: '+str(f.source))
                        continue
                    raise FileNotFoundError('{}{}{}'.format(
                            os.path.basename(filepath),
                            '' if self._suffix[0] in '-_' else '[-_]',
                            self._suffix,
                        ))

            #
            f.roi.from_ascii_file(filepath, *args)
        #
        if first:
            raise FileNotFoundError('not any roi file found')

ROI.register(ROIAsciiFiles)


class v1_ROIRecord(BoundingBoxes):
    """
    :class:`ROI` class for the individual special partitions in an analysis tree.

    This storing strategy is likely to be marked as deprecated.
    """
    __slots__ = ('_helper',)
    def __init__(self, label=None, _impl=None, **kwargs):
        SpecializedROI.__init__(self, **kwargs) # not BoundingBoxes.__init__
        if _impl is None:
            _impl = helper.helper.RoiHelper
        analyses = self._parent.analyses
        self._helper = _impl(analyses.analyses.statefree(), autosave=False, verbose=False)
        if label is None:
            label = ''
        self._bounding_boxes = {label: None}
    @property
    def _collections(self):
        return self._helper.collections
    @_collections.setter
    def _collections(self, roi):
        if roi is not None:
            raise AttributeError("can't set attribute")
    @property
    def reified(self):
        """ *bool*: :const:`True` if the files have been loaded """
        return not all([ bb is None for bb in self._bounding_boxes.values() ])
    @property
    def bounding_boxes(self):
        # this is enough to make `as_individual_roi` properly work
        if not self.reified:
            self.load()
        return self._bounding_boxes
    def as_support_regions(self, *args, **kwargs):
        if not self.reified:
            self.load()
        yield from BoundingBoxes.as_support_regions(self, *args, **kwargs)
    def load(self):
        for coll_label, meta_label in self._helper.get_meta_labels().items():
            self._bounding_boxes[coll_label] = \
                    self._helper.collections[coll_label] = \
                            self._helper.get_bounding_boxes(meta_label=meta_label)


class ROIRecords(DecentralizedROIManager):
    """
    :class:`ROI` class for the main :attr:`~tramway.analyzer.RWAnalyzer.roi` attribute.
    """
    __slots__ = ()
    def __init__(self, label=None, version=None, _impl=None, **kwargs):
        DecentralizedROIManager.__init__(self, **kwargs)
        if version is None:
            self._parent.logger.info('set version=1 to ensure constant behavior in the future')
            version = 1
        for f in self._parent.spt_data:
            f.roi.from_dedicated_rwa_record(label, version, _impl=_impl)


class HasROI(AnalyzerNode):
    """ Class to be inherited from by SPT data item classes.
    
    Maintains a self-modifying :attr:`roi` attribute."""
    __slots__ = ('_roi',)
    def _get_roi(self):
        """
        *ROI*: Regions of interest for the parent data block
        """
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
        Returns :const:`True` if filter *source* matches with `self.source`.

        .. note::

            Does not check against the alias.

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


__all__ = [ 'ROI', 'ROIInitializer', 'SpecializedROI', 'BoundingBoxes', 'DecentralizedROIManager',
        'BaseRegion', 'FullRegion', 'IndividualROI', 'BoundingBox', 'SupportRegion',
        'CommonROI', 'HasROI', 'ROIAsciiFile', 'ROIAsciiFiles', 'v1_ROIRecord', 'ROIRecords' ]

