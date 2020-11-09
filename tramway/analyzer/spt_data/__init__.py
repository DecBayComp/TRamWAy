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
from ..artefact import *
from ..roi import HasROI
from .abc import *
from collections.abc import Sequence, Set
import os.path
from tramway.core.xyt import load_xyt, load_mat, discard_static_trajectories
from tramway.core.analyses.auto import Analyses, AutosaveCapable
from tramway.core.hdf5.store import load_rwa
from tramway.core.exceptions import RWAFileException
from math import sqrt
import numpy as np
import pandas as pd


class SPTParameters(object):
    """ children classes should define the `_dt` and `_localization_error` attributes
        or implement the `dt` and `localization_error` properties.
        Default values should be None."""
    __slots__ = ()
    def __init__(self, localization_precision=None, localization_error=None):
        if localization_precision is None:
            if localization_error is not None:
                self.localization_error = localization_error
        elif localization_error is None:
            self.localization_precision = localization_precision
        else:
            raise ValueError('both error and precision are defined; choose either one')
    @classmethod
    def __parse__(cls, kwargs):
        return (
            kwargs.pop('localization_precision', None),
            kwargs.pop('localization_error', None),
            )
    @property
    def localization_error(self):
        return self._localization_error
    @localization_error.setter
    def localization_error(self, err):
        if err is None:
            err = 0.
        self._localization_error = err
    @property
    def localization_precision(self):
        return sqrt(self.localization_error)
    @localization_precision.setter
    def localization_precision(self, pr):
        err = pr if pr is None else pr*pr
        self.localization_error = err
    def discard_static_trajectories(self, dataframe, min_msd=None, **kwargs):
        if min_msd is None:
            min_msd = self.localization_error
        return discard_static_trajectories(dataframe, min_msd, **kwargs)
    @property
    def dt(self):
        if self._dt is None:
            t = self.dataframe['t']
            self._dt = np.median(t.diff())
        return self._dt
    @dt.setter
    def dt(self, dt):
        self._dt = dt
    @property
    def time_step(self):
        return self.dt
    @time_step.setter
    def time_step(self, dt):
        self.dt = dt
    @property
    def logger(self):
        return self._eldest_parent.logger


def normalize(p):
    return os.path.expanduser(os.path.normpath(p))


class SPTDataIterator(AnalyzerNode, SPTParameters):
    """ partial implementation for multiple SPT data items.

    Children classes must implement the `__iter__` method. """
    __slots__ = ('_bounds',)
    def __init__(self, **kwargs):
        prms = SPTParameters.__parse__(kwargs)
        AnalyzerNode.__init__(self, **kwargs)
        self._bounds = None
        SPTParameters.__init__(self, *prms)
    @property
    def localization_error(self):
        it = iter(self)
        sigma2 = next(it).localization_error
        while True:
            try:
                _s2 = next(it).localization_error
            except StopIteration:
                break
            else:
                if _s2 != sigma2:
                    raise AttributeError('not all the data blocks share the same localization error')
        return sigma2
    @localization_error.setter
    def localization_error(self, sigma2):
        for f in self:
            f.localization_error = sigma2
    @property
    def dt(self):
        it = iter(self)
        dt = next(it).dt
        while True:
            try:
                _dt = next(it).dt
            except StopIteration:
                break
            else:
                _delta = dt - _dt
                if 1e-12 < _delta*_delta:
                    raise AttributeError('not all the data blocks share the same time step (dt)')
        return dt
    @dt.setter
    def dt(self, dt):
        for f in self:
            f.dt = dt
    def as_dataframes(self, source=None, return_index=False):
        """returns an iterator.
        
        `source` can be a source name (filepath) or a boolean function
        that takes a source string as input argument."""
        for f in self.filter_by_source(source, return_index):
            yield f.dataframe
    def filter_by_source(self, source_filter, return_index=False):
        if return_index:
            _out = lambda i, f: i, f
        else:
            _out = lambda _, f: f
        if source_filter is None:
            if return_index:
                yield from enumerate(self)
            else:
                yield from self
        elif callable(source_filter):
            for i, f in enumerate(self):
                if source_filter(f.source):
                    yield _out(i, f)
        else:
            if isinstance(source_filter, str):
                sources = [normalize(source_filter)]
            elif isinstance(source_filter, Sequence):
                visited = dict()
                yielded = set()
                for _p in source_filter:
                    p = normalize(_p)
                    i, it = -1, iter(self)
                    try:
                        while True:
                            i += 1
                            f = next(it)
                            try:
                                s = visited[i]
                            except KeyError:
                                visited[i] = s = normalize(f.source)
                            if s == p:
                                if i in yielded:
                                    raise ValueError('duplicate source: {}'.format(_p))
                                yield _out(i, f)
                                yielded.add(i)
                                break
                    except StopIteration:
                        raise ValueError('cannot find source: {}'.format(_p))
                return
            elif isinstance(source_filter, Set):
                sources = set([ normalize(p) for p in source_filter ])
            for i, f in enumerate(self):
                p = normalize(f.source)
                if p in sources:
                    yield _out(i, f)
    def discard_static_trajectories(self, dataframe=None, min_msd=None, **kwargs):
        if dataframe is None:
            for data in self:
                data.discard_static_trajectories(min_msd=min_msd, **kwargs)
        else:
            return SPTParameters.discard_static_trajectories(self, dataframe, min_msd, **kwargs)
    @property
    def bounds(self):
        if self._bounds is None:
            for f in self:
                _bounds = f.bounds
                if self._bounds is None:
                    self._bounds = _bounds
                else:
                    self._bounds.loc['min'] = np.minimum(self._bounds.loc['min'], _bounds.loc['min'])
                    self._bounds.loc['max'] = np.minimum(self._bounds.loc['max'], _bounds.loc['max'])
        return self._bounds
    def self_update(self, new_self):
        """
        """
        if callable(new_self):
            f = new_self
            new_self = f(self)
        if new_self is not self:
            self._parent._spt_data = new_self
            try:
                roi_central = next(iter(new_self)).roi._global
                roi_central.reset()
            except AttributeError:
                pass
            else:
                for f in new_self:
                    roi_central._register_decentralized_roi(f)


class SPTDataInitializer(Initializer):
    """
    initial value for the `RWAnalyzer.spt_data` attribute.

    `from_...` methods alters the parent attribute which specializes
    into an initialized :class:`.abc.SPTData` object.
    """
    __slots__ = ()
    def from_ascii_file(self, filepath):
        """
        Sets a text file as the source of SPT data.

        Note that data loading is NOT performed while calling this method.
        Loading is postponed until the data is actually required.
        This lets additional arguments to be provided to the spt_data attribute.
        """
        self.specialize( StandaloneSPTAsciiFile, filepath )
    def from_ascii_files(self, filepattern):
        """
        Sets text files, which paths match with a pattern, as the source of SPT data.

        `filepattern` is a standard filepath with the '*' placeholder.
        For example:  `'dataset/*.txt'`

        The parts of the filename that match the placeholder are used as keys.

        Note that data loading is NOT performed while calling this method.
        Loading is postponed until the data is actually required.
        This lets additional arguments to be provided to the spt_data attribute.
        """
        self.specialize( SPTAsciiFiles, filepattern )
    def from_dataframe(self, df):
        self.specialize( StandaloneSPTDataFrame, df )
    def from_dataframes(self, dfs):
        self.specialize( SPTDataFrames, dfs )
    def from_mat_file(self, filepath):
        """
        Sets a MatLab V7 file as the source of SPT data.

        Note that data loading is NOT performed while calling this method.
        Loading is postponed until the data is actually required.
        This lets additional arguments to be provided to the spt_data attribute.
        """
        self.specialize( StandaloneSPTMatFile, filepath )
    def from_mat_files(self, filepattern):
        """
        Sets MatLab V7 files, which paths match with a pattern, as the source of SPT data.

        `filepattern` is a standard filepath with the '*' placeholder.
        For example:  `'datasets/*.txt'`

        The parts of the filename that match the placeholder are used as keys.

        Note that data loading is NOT performed while calling this method.
        Loading is postponed until the data is actually required.
        This lets additional arguments to be provided to the spt_data attribute.
        """
        self.specialize( SPTMatFiles, filepattern )
    def from_rwa_file(self, filepath):
        """
        Similar to `from_ascii_file`.
        """
        self.specialize( StandaloneRWAFile, filepath )
    def from_rwa_files(self, filepattern):
        """
        Similar to `from_ascii_files`.
        """
        self.specialize( RWAFiles, filepattern )
    def from_rw_generator(self, generator):
        """
        A generator is an object that features a `generate` method
        which input arguments are exposed as attributes.
        """
        self.specialize( RWGenerator, generator )
    def from_analysis_tree(self, analyses, copy=False):
        self.specialize( RWAnalyses, analyses, copy )
    def from_tracker(self):
        self.specialize( TrackerOutput )



class StandaloneDataItem(object):
    """
    partial implementation for single data item `RWAnalyzer.spt_data` attribute
    """
    __slots__ = ()
    def __len__(self):
        return 1
    def __iter__(self):
        yield self
    def as_dataframes(self, source=None):
        return SPTDataIterator.as_dataframes(self, source)
    def self_update(self, new_self):
        if callable(new_self):
            f = new_self
            new_self = f(self)
        if new_self is not self:
            self._parent._spt_data = new_self
            try:
                roi_central = next(iter(new_self)).roi._global
                roi_central.reset()
            except AttributeError:
                pass
            else:
                for f in new_self:
                    roi_central._register_decentralized_roi(f)



class _SPTDataFrame(HasROI, SPTParameters):
    __slots__ = ('_source','_analyses','_dt','_localization_error')
    def __init__(self, df, source=None, **kwargs):
        prms = SPTParameters.__parse__(kwargs)
        HasROI.__init__(self, **kwargs)
        self._source = source
        self.analyses = Analyses(df, standard_metadata(), autosave=True)
        self._dt = self._localization_error = None
        SPTParameters.__init__(self, *prms)
    def set_analyses(self, tree):
        assert bool(tree.metadata)
        if self.source is not None:
            tree.metadata['datafile'] = self.source
        #if not isinstance(tree, AutosaveCapable):
        #    autosaver = Analyses(tree)
        #    tree = autosaver
        self._analyses = tree
    @property
    def analyses(self):
        return self._analyses
    @analyses.setter
    def analyses(self, tree):
        self.set_analyses(tree)
    @property
    def dataframe(self):
        return self.analyses.data
    @property
    def _dataframe(self):
        return self._analyses.data
    @_dataframe.setter
    def _dataframe(self, df):
        self._analyses._data = df
    @property
    def reified(self):
        return self._dataframe is not None
    @property
    def columns(self):
        return self.dataframe.columns
    def reset_origin(self, columns=None):
        if columns is None:
            columns = [ col for col in ['x', 'y', 'z', 't'] if col in self.columns ]
        origin = self.dataframe[columns].min().values
        self.dataframe[columns] -= origin
    def discard_static_trajectories(self, dataframe=None, min_msd=None, **kwargs):
        inplace = dataframe is None
        if inplace:
            dataframe = self._dataframe
        dataframe = SPTParameters.discard_static_trajectories(self, dataframe, min_msd, **kwargs)
        if inplace:
            self._dataframe = dataframe
        else:
            return dataframe
    @property
    def source(self):
        return self._source
    @property
    def bounds(self):
        _min = self.dataframe.min(axis=0)
        _max = self.dataframe.max(axis=0)
        _bounds = np.stack((_min, _max), axis=0)
        _bounds = pd.DataFrame( _bounds, columns=_min.index, index=['min','max'])
        return _bounds
    def to_ascii_file(self, filepath, columns=None, header=True, **kwargs):
        """
        Exports the data to text file.

        Arguments:

            filepath (str): output filepath.

            columns (sequence of str): columns to be exported.

            header (bool): print column names on the first line.

        Additional keyword arguments are passed to `pandas.DataFrame.to_csv`.
        See for example `float_format`.
        """
        df = self.dataframe
        if columns:
            df = df[columns]
        for arg in ('sep', 'index'):
            try:
                kwargs.pop(arg)
            except KeyError:
                pass
            else:
                self.logger.warning("ignoring argument '{}'".format(arg))
        df.to_csv(filepath, sep='\t', index=False, header=header, **kwargs)
    def to_rwa_file(self, filepath, **kwargs):
        if self.analyses.data is None:
            raise ValueError('no data available')
        try:
            analyses = self.analyses.analyses.statefree()
        except AttributeError:
            analyses = self.analyses
        save_rwa(filepath, analyses, **kwargs)
    def add_sampling(self, sampling, label, comment=None):
        analyses = self.analyses
        label = analyses.autoindex(label)
        subtree = Analyses(sampling, standard_metadata())
        analyses.add(subtree, label, comment)
        return label
    def get_sampling(self, label):
        return Analysis.get_analysis(self.analyses, label)
    def autosaving(self, *args, **kwargs):
        assert isinstance(self.analyses, AutosaveCapable)
        if not self.analyses.rwa_file:
            if self.source:
                self.analyses.rwa_file = os.path.splitext(self.source)[0]+'.rwa'
            else:
                self.logger.warning('no output filename defined')
        return self.analyses.autosaving(*args, **kwargs)


class SPTDataFrame(_SPTDataFrame):
    __slots__ = ()

SPTDataItem.register(SPTDataFrame)


class StandaloneSPTDataFrame(SPTDataFrame, StandaloneDataItem):
    """
    `RWAnalyzer.spt_data` attribute for single dataframes.
    """
    __slots__ = ()

SPTData.register(StandaloneSPTDataFrame)


class SPTDataFrames(SPTDataIterator):
    """
    `RWAnalyzer.spt_data` attribute for multiple dataframes.
    """
    __slots__ = ('_dataframes',)
    def __init__(self, dfs, **kwargs):
        SPTDataIterator.__init__(self, **kwargs)
        self.dataframes = dfs
    @property
    def dataframes(self):
        return [ df.dataframe for df in self._dataframes ]
    @dataframes.setter
    def dataframes(self, dfs):
        if not isinstance(dfs, (tuple, list)):
            raise TypeError("the input dataframes are not in a tuple or list")
        if not dfs:
            raise ValueError("no dataframes found")
        self._dataframes = tuple([ self._bear_child( SPTDataFrame, df ) for df in  dfs ])
        if not all([ tuple(self.columns) == tuple(df.columns) for df in dfs ]):
            raise ValueError("not all the dataframes feature the same column names")
    @property
    def reified(self):
        return True
    def __len__(self):
        return len(self._dataframes)
    def __iter__(self):
        yield from self._dataframes
    @property
    def columns(self):
        return self._dataframes[0].columns
    def reset_origin(self, columns=None, same_origin=False):
        if same_origin:
            if columns is None:
                columns = [ col for col in ['x', 'y', 'z', 't'] if col in self.columns ]
            origin = None
            for df in self.dataframes:
                _origin = df[columns].min().values
                if origin is None:
                    origin = _origin
                else:
                    origin = np.minimum(origin, _origin)
            for df in self.dataframes:
                df[columns] -= origin
        else:
            for df in self._dataframes:
                df.reset_origin(columns)

SPTData.register(SPTDataFrames)


class SPTFile(_SPTDataFrame):
    __slots__ = ('_filepath','_alias','_reset_origin','_discard_static_trajectories')
    def __init__(self, filepath, dataframe=None, **kwargs):
        _SPTDataFrame.__init__(self, dataframe, filepath, **kwargs)
        self._alias = None
        self._reset_origin = False
        self._discard_static_trajectories = False
    @property
    def filepath(self):
        return self._source
    @filepath.setter
    def filepath(self, fp):
        if self.reified:
            raise AttributeError("file '{}' has already been loaded; cannot set the filepath anymore".format(self.filepath.split('/')[-1]))
        else:
            self._source = fp
    @property
    def source(self):
        return self.filepath
    @source.setter
    def source(self, fp):
        self.filepath = fp
    @property
    def alias(self):
        return self._alias
    @alias.setter
    def alias(self, name):
        if callable(name):
            self._alias = name(self.filepath)
        else:
            self._alias = name
    def get_analyses(self):
        if self._analyses.data is None:
            self.load()
            assert self._analyses.data is not None
        return self._analyses
    @property
    def analyses(self):
        return self.get_analyses()
    @analyses.setter
    def analyses(self, tree):
        self.set_analyses(tree)
    def _trigger_discard_static_trajectories(self):
        if self._discard_static_trajectories is True:
            SPTDataFrame.discard_static_trajectories(self)
        elif self._discard_static_trajectories: # not in (None, False)
            SPTDataFrame.discard_static_trajectories(self, **self._discard_static_trajectories)
    def _trigger_reset_origin(self):
        if self._reset_origin:
            SPTDataFrame.reset_origin(self, self._reset_origin)
    def reset_origin(self, columns=None, same_origin=False):
        if self.reified:
            SPTDataFrame.reset_origin(self, columns)
        else:
            self._reset_origin = True if columns is None else columns
    def discard_static_trajectories(self, dataframe=None, min_msd=None, **kwargs):
        if dataframe is not None or self.reified:
            return SPTDataFrame.discard_static_trajectories(self, dataframe, min_msd, **kwargs)
        else:
            if min_msd is not None:
                kwargs['min_msd'] = min_msd
            self._discard_static_trajectories = kwargs if kwargs else True
    def get_image(self, match=None):
        """
        Looks for the corresponding localization microscopy image in the `~RWAnalyzer.images` attribute.

        The search is based on the object's alias.
        The first item that contains the alias in its filename is returned as a match.

        If the `alias` attribute is not set or images do not have a defined `filepath`,
        an `AttributeError` exception is raised.

        The optional argument *match* is a 2-argument callable with the following signature:

        Arguments:

            alias (str): alias of these SPT data.

            filepath (str): filepath to a stack of images.

        Returns

            bool: ``True`` if *filepath* matches with *alias*.

        """
        if match is None:
            match = lambda alias, filepath: alias in os.path.basename(filepath)
        if self.alias is None:
            raise AttributeError('alias is not defined')
        analyzer = self._eldest_parent
        any_filepath_defined = False
        for image in analyzer.images:
            if image.filepath: # may raise AttributeError
                any_filepath_defined = True
                if match(self.alias, image.filepath):
                    return image
        if not any_filepath_defined:
            raise AttributeError('image filepaths are not defined')


class RawSPTFile(SPTFile):
    __slots__ = ('_columns',)
    def __init__(self, filepath, dataframe=None, **kwargs):
        SPTFile.__init__(self, filepath, dataframe, **kwargs)
        self._columns = None
    @property
    def columns(self):
        if self._columns is None:
            return self.dataframe.columns
        else:
            return self._columns
    @columns.setter
    def columns(self, cols):
        if self.reified:
            raise AttributeError('the SPT data have already been loaded; cannot set column names anymore')
        else:
            self._columns = cols

class _SPTAsciiFile(RawSPTFile):
    __slots__ = ()
    def load(self):
        self._dataframe = load_xyt(os.path.expanduser(self.filepath), self._columns, reset_origin=self._reset_origin)
        self._trigger_discard_static_trajectories()

class SPTAsciiFile(_SPTAsciiFile):
    __slots__ = ()

SPTDataItem.register(SPTAsciiFile)

class StandaloneSPTAsciiFile(_SPTAsciiFile, StandaloneDataItem):
    """
    `RWAnalyzer.spt_data` attribute for single SPT text files.
    """
    __slots__ = ()

SPTData.register(StandaloneSPTAsciiFile)


class SPTFiles(SPTDataFrames):
    __slots__ = ('_files','_filepattern')
    def __init__(self, filepattern, **kwargs):
        SPTDataIterator.__init__(self, **kwargs)
        self._files = []
        if isinstance(filepattern, str):
            self._filepattern = os.path.expanduser(filepattern)
        else:
            self._filepattern = [os.path.expanduser(pattern) for pattern in filepattern]
    @property
    def filepattern(self):
        return self._filepattern
    @filepattern.setter
    def filepattern(self, fp):
        if self._files:
            if fp != self._filepattern:
                raise AttributeError('the files have already been listed; cannot set the file pattern anymore')
        else:
            self._filepattern = fp
    @property
    def files(self):
        if not self._files:
            self.list_files()
        return self._files
    @property
    def filepaths(self):
        return [ f.filepath for f in self.files ]
    @property
    def dataframes(self): # this also hides the parent class' setter
        return [ f.dataframe for f in self.files ]
    @property
    def partially_reified(self):
        return self._files and any([ f.reified for f in self._files ])
    @property
    def fully_reified(self):
        return self._files and all([ f.reified for f in self._files ])
    @property
    def reified(self):
        return self.fully_reified
    def __len__(self):
        return len(self.files)
    def __iter__(self):
        yield from self.files
    def list_files(self):
        from glob import glob
        if isinstance(self.filepattern, str):
            self._files = glob(self.filepattern)
        else:
            self._files = []
            for pattern in self.filepattern:
                self._files += glob(pattern)
        if not self._files:
            raise ValueError("no files found")
    @property
    def columns(self):
        it = iter(self)
        col = next(it).columns
        while True:
            try:
                _col = next(it).columns
            except StopIteration:
                break
            else:
                if _col != col:
                    raise AttributeError('not all the data blocks share the same columns')
        return col
    def reset_origin(self, columns=None, same_origin=False):
        if same_origin:
            SPTDataFrames.reset_origin(self, columns, True)
        else:
            for f in self.files:
                f.reset_origin(columns)
    @property
    def alias(self):
        raise AttributeError('alias is write-only')
    @alias.setter
    def alias(self, name):
        if callable(name):
            for f in self.files:
                f.alias = name
        else:
            raise TypeError('global alias is not callable')
    @property
    def aliases(self):
        return [ f.alias for f in self.files ]
    def filter_by_source(self, source_filter, return_index=False):
        if return_index:
            _out = lambda i, f: i, f
        else:
            _out = lambda _, f: f
        # first test for aliases
        _any = False
        if source_filter is None:
            if return_index:
                yield from enumerate(self)
            else:
                yield from self
            _any = True
        elif callable(source_filter):
            for i, f in enumerate(self):
                if source_filter(f.alias):
                    yield _out(i, f)
                    _any = True
        else:
            if isinstance(source_filter, str):
                aliases = [source_filter]
            elif isinstance(source_filter, Sequence):
                yielded = set()
                for a in source_filter:
                    i, it = -1, iter(self)
                    try:
                        while True:
                            i += 1
                            f = next(it)
                            if f.alias == a:
                                if i in yielded:
                                    raise ValueError('duplicate source: {}'.format(a))
                                yield _out(i, f)
                                yielded.add(i)
                                _any = True
                                break
                    except StopIteration:
                        raise ValueError('cannot find source: {}'.format(a))
            elif isinstance(source_filter, Set):
                aliases = source_filter
            if not _any:
                for i, f in enumerate(self):
                    if f.alias in aliases:
                        yield _out(i, f)
                        _any = True
        # if no alias matched, try sources
        if not _any:
            yield from SPTDataIterator.filter_by_source(self, source_filter, return_index)

class RawSPTFiles(SPTFiles):
    __slots__ = ()
    @property
    def columns(self):
        return SPTFiles.columns.fget(self)
    @columns.setter
    def columns(self, col):
        for f in self:
            f.columns = col


class SPTAsciiFiles(RawSPTFiles):
    """
    `RWAnalyzer.spt_data` attribute for multiple SPT text files.
    """
    __slots__ = ()
    def list_files(self):
        SPTFiles.list_files(self)
        self._files = [ self._bear_child( SPTAsciiFile, filepath ) for filepath in self._files ]

SPTData.register(SPTAsciiFiles)


class _RWAFile(SPTFile):
    __slots__ = ()
    def __init__(self, filepath, **kwargs):
        SPTFile.__init__(self, filepath, None, **kwargs)
    def set_analyses(self, tree):
        self._analyses = tree
    def load(self):
        # ~ expansion is no longer necessary from rwa-python==0.8.4
        try:
            self.analyses = load_rwa(os.path.expanduser(self.filepath), lazy=True)
        except KeyError as e:
            raise RWAFileException(self.filepath, e) from None
        self._trigger_discard_static_trajectories()
        self._trigger_reset_origin()

class RWAFile(_RWAFile):
    __slots__ = ()

SPTDataItem.register(RWAFile)

class StandaloneRWAFile(_RWAFile, StandaloneDataItem):
    """
    `RWAnalyzer.spt_data` attribute for single RWA files.
    """
    __slots__ = ()

SPTData.register(StandaloneRWAFile)


class RWAFiles(SPTFiles):
    """
    `RWAnalyzer.spt_data` attribute for multiple RWA files.
    """
    __slots__ = ()
    def list_files(self):
        SPTFiles.list_files(self)
        self._files = [ self._bear_child( RWAFile, filepath ) for filepath in self._files ]

SPTData.register(RWAFiles)


class _SPTMatFile(RawSPTFile):
    __slots__ = ('_coord_scale',)
    def __init__(self, filepath, dataframe=None, **kwargs):
        RawSPTFile.__init__(self, filepath, dataframe, **kwargs)
        self._coord_scale = None
    @property
    def coord_scale(self):
        return self._coord_scale
    @coord_scale.setter
    def coord_scale(self, scale):
        if self.reified:
            raise AttributeError('the SPT data have already been loaded; cannot set the coordinate scale anymore')
        else:
            self._coord_scale = scale
    @property
    def pixel_size(self):
        self.logger.warning('attribute pixel_size is deprecated; use coord_scale instead')
        return self.coord_scale
    @pixel_size.setter
    def pixel_size(self, siz):
        self.logger.warning('attribute pixel_size is deprecated; use coord_scale instead')
        self.coord_scale = siz
    def load(self):
        try:
            self._dataframe = load_mat(os.path.expanduser(self.filepath),
                    columns=self._columns, dt=self._dt, coord_scale=self.coord_scale)
        except OSError as e:
            raise OSError('{}\nwhile loading file: {}'.format(e, self.source)) from None
        self._trigger_discard_static_trajectories()
        self._trigger_reset_origin()

class SPTMatFile(_SPTMatFile):
    __slots__ = ()

SPTDataItem.register(SPTMatFile)

class StandaloneSPTMatFile(_SPTMatFile, StandaloneDataItem):
    """
    `RWAnalyzer.spt_data` attribute for single MatLab v7 data files.
    """
    __slots__ = ()

SPTData.register(StandaloneSPTMatFile)


class SPTMatFiles(RawSPTFiles):
    """
    `RWAnalyzer.spt_data` attribute for multiple MatLab v7 data files.
    """
    __slots__ = ()
    @property
    def coord_scale(self):
        it = iter(self)
        px = next(it).coord_scale
        while True:
            try:
                _px = next(it).coord_scale
            except StopIteration:
                break
            else:
                _delta = px - _px
                if 1e-12 < _delta*_delta:
                    raise AttributeError('not all the data blocks share the same coordinate scale')
        return px
    @coord_scale.setter
    def coord_scale(self, scale):
        for f in self:
            f.coord_scale = scale
    @property
    def pixel_size(self):
        self.logger.warning('attribute pixel_size is deprecated; use coord_scale instead')
        return self.coord_scale
    @pixel_size.setter
    def pixel_size(self, px):
        self.logger.warning('attribute pixel_size is deprecated; use coord_scale instead')
        self.coord_scale = px
    def list_files(self):
        SPTFiles.list_files(self)
        self._files = [ self._bear_child( SPTMatFile, filepath ) for filepath in self._files ]

SPTData.register(RWAFiles)


class RWGenerator(SPTDataFrame):
    """ not implemented yet """
    pass


class RWAnalyses(StandaloneSPTDataFrame):
    """
    `RWAnalyzer.spt_data` attribute for single analysis trees as stored in *.rwa* files.

    .. note:

        copying (initializing with ``copy=True``) copies only the SPT data
        and NOT the subsequent analyses.

    """
    __slots__ = ()
    def __init__(self, analyses, copy=False, **kwargs):
        dataframe = analyses.data
        source = analyses.metadata.get('datafile', None)
        SPTDataFrame.__init__(self, dataframe, source, **kwargs)
        if not copy:
            self.analyses = analyses

class MultipleRWAnalyses(SPTDataFrames):
    """
    `RWAnalyzer.spt_data` attribute for multiple analysis trees as stored in *.rwa* files.

    .. note:

        copying (initializing with ``copy=True``) copies only the SPT data
        and NOT the subsequent analyses.

    """
    __slots__ = ()
    def __init__(self, analyses, copy=False, **kwargs):
        dataframes = [ a.data for a in analyses ]
        SPTDataFrames.__init__(self, dataframes, **kwargs)
        if copy:
            for f, a in zip(self._dataframes, analyses):
                f.source = a.metadata.get('datafile', None)
        else:
            for f, a in zip(self._dataframes, analyses):
                f.analyses = a
        

class _FakeSPTData(pd.DataFrame):
    __slots__ = ()
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, columns=list('nxyt'))

class TrackerOutput(SPTDataFrames):
    __slots__ = ()
    def __init__(self, **kwargs):
        SPTDataFrames.__init__(self, [_FakeSPTData()], **kwargs)
    def add_tracked_data(self, trajectories, source=None):
        df = self._bear_child( SPTDataFrame, trajectories, source )
        siblings = self._dataframes[0]
        df.dt = siblings.dt
        df.localization_error = siblings.localization_error
        if isinstance(siblings.dataframe, _FakeSPTData):
            assert not self._dataframes[1:]
            self._dataframes = [ df ]
        else:
            if tuple(df.columns) != tuple(siblings.columns):
                raise ValueError("not all the dataframes feature the same column names")
            self._dataframes.append( df )

