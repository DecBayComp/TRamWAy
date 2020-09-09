
from ..attribute import *
from ..artefact import *
from ..roi import HasROI
from .abc import *
import os.path
from tramway.core.xyt import load_xyt, discard_static_trajectories
from tramway.core.analyses.auto import Analyses, AutosaveCapable
import warnings
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


class SPTDataIterator(AnalyzerNode, SPTParameters):
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
    def as_dataframes(self, source=None):
        """returns an iterator.
        
        `source` can be a source name (filepath) or a boolean function
        that takes a source string as input argument."""
        if source is None:
            for f in self:
                yield f.dataframe
        else:
            if callable(source):
                filter = source
            else:
                filter = lambda s: s == source
            for f in self:
                if filter(f.source):
                    yield f.dataframe
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


class SPTDataInitializer(Initializer):
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
        For example:  `'datasets/*.txt'`

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



class StandaloneDataItem(object):
    __slots__ = ()
    def __len__(self):
        return 1
    def __iter__(self):
        yield self
    def as_dataframes(self, source=None):
        return SPTDataIterator.as_dataframes(self, source)



class _SPTDataFrame(HasROI, SPTParameters):
    __slots__ = ('_source','_analyses','_dt','_localization_error')
    def __init__(self, df, source=None, **kwargs):
        prms = SPTParameters.__parse__(kwargs)
        HasROI.__init__(self, **kwargs)
        self._source = source
        self.analyses = Analyses(df, standard_metadata())
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
    def to_ascii_file(self, filepath):
        self.dataframe.to_csv(filepath, sep='\t', index=False)
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
        return self.analyses.autosaving(*args, **kwargs)

class SPTDataFrame(_SPTDataFrame):
    __slots__ = ()

SPTDataItem.register(SPTDataFrame)


class StandaloneSPTDataFrame(SPTDataFrame, StandaloneDataItem):
    __slots__ = ()

SPTData.register(StandaloneSPTDataFrame)


class SPTDataFrames(SPTDataIterator):
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
        if not all([ self.columns == df.columns for df in dfs ]):
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
    __slots__ = ('_filepath','_reset_origin','_discard_static_trajectories')
    def __init__(self, filepath, dataframe=None, **kwargs):
        _SPTDataFrame.__init__(self, dataframe, filepath, **kwargs)
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
    @property
    def columns(self):
        if self._dataframe is None:
            if self._columns is None:
                self.load()
            else:
                return self._columns
        return self._dataframe.columns
    def load(self):
        # post-processing only; to be called last by children classes
        if self._discard_static_trajectories is True:
            SPTDataFrame.discard_static_trajectories(self)
        elif self._discard_static_trajectories:
            SPTDataFrame.discard_static_trajectories(self, **self._discard_static_trajectories)
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


class SPTAsciiFile(SPTFile):
    __slots__ = ('_columns',)
    def __init__(self, filepath, dataframe=None, **kwargs):
        SPTFile.__init__(self, filepath, dataframe, **kwargs)
        self._columns = None
        assert bool(self._analyses.metadata)
        assert 'datafile' in self._analyses.metadata
        assert self._analyses.metadata['datafile'] == filepath
    @property
    def columns(self):
        if self._dataframe is None:
            if self._columns is None:
                self.load()
            else:
                return self._columns
        return self._dataframe.columns
    @columns.setter
    def columns(self, cols):
        if self.reified:
            raise AttributeError('the SPT data have already been loaded; cannot set column names anymore')
        else:
            self._columns = cols
    def load(self):
        self._dataframe = load_xyt(os.path.expanduser(self.filepath), self._columns, reset_origin=self._reset_origin)
        SPTFile.load(self) # post-process

SPTDataItem.register(SPTAsciiFile)

class StandaloneSPTAsciiFile(SPTAsciiFile, StandaloneDataItem):
    __slots__ = ()

SPTData.register(StandaloneSPTAsciiFile)


class SPTFiles(SPTDataFrames):
    __slots__ = ('_files','_filepattern')
    def __init__(self, filepattern, **kwargs):
        SPTDataIterator.__init__(self, **kwargs)
        self._files = []
        self._filepattern = os.path.expanduser(filepattern)
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
    def dataframes(self):
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
        self._files = glob(self.filepattern)
        if not self._files:
            raise ValueError("no files found")
    @property
    def columns(self):
        return self.files[0].columns
    def reset_origin(self, columns=None, same_origin=False):
        if same_origin:
            SPTDataFrames.reset_origin(self, columns, True)
        else:
            for f in self.files:
                f.reset_origin(columns)


class SPTAsciiFiles(SPTFiles):
    __slots__ = ()
    def list_files(self):
        SPTFiles.list_files(self)
        self._files = [ self._bear_child( SPTAsciiFile, filepath ) for filepath in self._files ]

SPTData.register(SPTAsciiFiles)


class RWAFile(SPTFile):
    __slots__ = ()
    def __init__(self, filepath, **kwargs):
        SPTFile.__init__(self, filepath, None, **kwargs)
    def load(self):
        self.analyses = load_rwa(self.filepath, lazy=True)
        SPTFile.load(self) # post-process

SPTDataItem.register(RWAFile)

class StandaloneRWAFile(RWAFile, StandaloneDataItem):
    __slots__ = ()

SPTData.register(StandaloneRWAFile)


class RWAFiles(SPTFiles):
    __slots__ = ()
    def list_files(self):
        SPTFiles.list_files(self)
        self._files = [ self._bear_child( RWAFile, filepath ) for filepath in self._files ]

SPTData.register(RWAFiles)


class RWGenerator(SPTDataFrame):
    """ not implemented yet """
    pass


class RWAnalyses(StandaloneSPTDataFrame):
    __slots__ = ()
    def __init__(self, analyses, copy=False, **kwargs):
        dataframe = analyses.data
        source = analyses.metadata.get('datafile', None)
        SPTDataFrame.__init__(self, dataframe, source, **kwargs)
        if not copy:
            self.analyses = analyses

