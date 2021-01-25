# -*- coding: utf-8 -*-

# Copyright © 2020-2021, Institut Pasteur
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
from tramway.core.analyses import base, lazy
from tramway.core.analyses.auto import Analyses, AutosaveCapable
from tramway.core.hdf5.store import load_rwa
from rwa.lazy import islazy
from tramway.core.exceptions import RWAFileException
from math import sqrt
import numpy as np
import pandas as pd
import copy
import glob as _glob


def compute_dtypes(df, precision):
    """
    Returns a dictionnary of dtypes to align the numerical precision of
    a :class:`pandas.DataFrame` data.
    """
    if isinstance(precision, dict):
        dtypes = precision
    else:
        if precision == 'half':
            nbytes = '2'
        elif precision == 'single':
            nbytes = '4'
        elif precision == 'double':
            nbytes = '8'
        else:
            raise ValueError('precision not supported: {}'.format(precision))
        dtypes = {}
        for col, dtype in df.dtypes.items():
            dtype = np.dtype(dtype).str
            if dtype[0] == '<' and dtype[1] in 'uif' and dtype[2] != nbytes:
                dtypes[col] = np.dtype(dtype[:2]+nbytes)
    return dtypes


def glob(pattern):
    """ Same as :func:`glob.glob`, with support for tilde.

    :func:`os.path.expanduser` is applied to `pattern`,
    and then the expanded part is replaced back with
    tilde in the resulting paths."""
    if pattern.startswith('~'):
        home = os.path.expanduser('~')
        files = []
        for f in _glob.glob(os.path.expanduser(pattern)):
            if f.startswith(home):
                f = '~' + f[len(home):]
            files.append(f)
    else:
        files = _glob.glob(pattern)
    return files


class SPTParameters(object):
    """ Children classes should define the :attr:`_frame_interval` and :attr:`_localization_error`
        attributes, or implement the :attr:`frame_interval` and :attr:`localization_error` properties.

        Default values should be :const:`None`."""
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
    localization_error.__doc__ = SPTData.localization_error.__doc__
    @property
    def localization_precision(self):
        r"""
        *float*: Localization precision in :math:`\mu m`;
            :attr:`localization_error` :math:`\sigma^2` is affected by
            :attr:`localization_precision` :math:`\sigma` and vice versa
        """
        if self.localization_error is None:
            return None
        else:
            return sqrt(self.localization_error)
    @localization_precision.setter
    def localization_precision(self, pr):
        err = pr if pr is None else pr*pr
        self.localization_error = err
    def discard_static_trajectories(self, dataframe, min_msd=None, **kwargs):
        if min_msd is None:
            min_msd = self.localization_error
        return discard_static_trajectories(dataframe, min_msd, **kwargs)
    discard_static_trajectories.__doc__ = SPTData.discard_static_trajectories.__doc__
    @property
    def frame_interval(self):
        if self._frame_interval is None:
            t = self.dataframe['t']
            self._frame_interval = t.diff().median()
        return self._frame_interval
    @frame_interval.setter
    def frame_interval(self, dt):
        self._frame_interval = dt
    frame_interval.__doc__ = SPTData.frame_interval.__doc__
    @property
    def time_step(self):
        """
        *float*: Alias for the :attr:`frame_interval` property
        """
        return self.frame_interval
    @time_step.setter
    def time_step(self, dt):
        self.frame_interval = dt
    @property
    def dt(self):
        return self.frame_interval
    @dt.setter
    def dt(self, dt):
        self.frame_interval = dt
    dt.__doc__ = time_step.__doc__
    @property
    def logger(self):
        """
        Logger of the parent :class:`~tramway.analyzer.RWAnalyzer`
        """
        return self._eldest_parent.logger
    def set_precision(self, precision):
        """
        Sets the numerical precision of the raw data.

        Arguments:

            precision (*dict* or *str*): any of :const:`'half'`, :const:`'single'`, :const:`'double'`,
                or a dictionnary of dtypes with column names as keys,
                as admitted by :meth:`pandas.DataFrame.astype`.

        """
        dtypes = compute_dtypes(self.dataframe, precision)
        self.dataframe = self.dataframe.astype(dtypes)


def _normalize(p):
    return os.path.expanduser(os.path.normpath(p))


class SPTDataIterator(AnalyzerNode, SPTParameters):
    """ Partial implementation for multi-item :class:`SPTData`.

    Children classes must implement the :meth:`__iter__` method. """
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
    def frame_interval(self):
        it = iter(self)
        dt = next(it).frame_interval
        while True:
            try:
                _dt = next(it).frame_interval
            except StopIteration:
                break
            else:
                _delta = dt - _dt
                if 1e-12 < _delta*_delta:
                    raise AttributeError('not all the data blocks share the same frame interval (dt)')
        return dt
    @frame_interval.setter
    def frame_interval(self, dt):
        for f in self:
            f.frame_interval = dt
    @property
    def dt(self):
        return self.frame_interval
    @dt.setter
    def dt(self, dt):
        self.frame_interval = dt
    @property
    def time_step(self):
        return self.frame_interval
    @time_step.setter
    def time_step(self, dt):
        self.frame_interval = dt
    def as_dataframes(self, source=None, return_index=False):
        """ Generator function; yields :class:`pandas.DataFrame` objects.
        
        `source` can be a source name (filepath) or a boolean function
        that takes a source string as input argument."""
        for f in self.filter_by_source(source, return_index):
            yield f.dataframe
    def filter_by_source(self, source_filter, return_index=False):
        """ Generator function; similar to :meth:`__iter__`;
        yields :class:`SPTDataItem` objects.

        *source* can be a single `str` value, or a set of `str` values,
        or a sequence of `str` values (the order is followed),
        or a `callable` that takes a `str` value and returns a `bool` value.
        """
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
                sources = [_normalize(source_filter)]
            elif isinstance(source_filter, Sequence):
                visited = dict()
                yielded = set()
                for _p in source_filter:
                    p = _normalize(_p)
                    i, it = -1, iter(self)
                    try:
                        while True:
                            i += 1
                            f = next(it)
                            try:
                                s = visited[i]
                            except KeyError:
                                visited[i] = s = _normalize(f.source)
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
                sources = set([ _normalize(p) for p in source_filter ])
            for i, f in enumerate(self):
                p = _normalize(f.source)
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
    bounds.__doc__ = SPTData.bounds.__doc__
    def self_update(self, new_self):
        """
        """
        if callable(new_self):
            f = new_self
            new_self = f(self)
        if isinstance(self, Proxy) and not isinstance(new_self, Proxy):
            new_self = type(self)(new_self)
        if new_self is not self:
            self._parent._spt_data = new_self
            new_self._parent = self._parent
            try:
                roi_central = next(iter(self)).roi._global
                roi_central.reset()
            except AttributeError:
                pass
            else:
                for f in new_self:
                    roi_central._register_decentralized_roi(f)
    def reload_from_rwa_files(self, skip_missing=False):
        """
        Reloads the SPT data and analysis tree from the corresponding rwa files.

        The rwa file that corresponds to an SPT file should be available at the
        same path with the *.rwa* extension instead of the SPT file's extension.

        This method is known to fail with a :class:`TypeError` exception in cases
        where not any matching *.rwa* file can be found.

        .. note::

            As this operation modifies the SPT data `source` and `filepath` attributes,
            aliases should be favored when identifying or filtering SPT data items.

        """
        RWAFiles.reload_from_rwa_files(self, skip_missing=skip_missing)
    def set_precision(self, precision):
        for f in self:
            f.set_precision(precision)


class SPTDataInitializer(Initializer):
    """
    Initial value for the :class:`~tramway.analyzer.RWAnalyzer`
    :attr:`~tramway.analyzer.RWAnalyzer.spt_data` attribute.

    *from_...* methods alters the parent attribute which specializes
    into an initialized :class:`SPTData` object.
    """
    __slots__ = ()
    def from_ascii_file(self, filepath):
        """
        Sets a text file as the source of SPT data.

        Note that data loading is NOT performed while calling this method.
        Loading is postponed until the data is actually required.
        This lets additional arguments to be provided to the
        :class:`~tramway.analyzer.RWAnalyzer` :attr:`~tramway.analyzer.RWAnalyzer.spt_data`
        attribute before the data are loaded.

        See also :class:`StandaloneSPTAsciiFile`.
        """
        self.specialize( StandaloneSPTAsciiFile, filepath )
    def from_ascii_files(self, filepattern):
        """
        Sets text files, which paths match with a pattern, as the source of SPT data.

        `filepattern` is a standard filepath with the :const:`'*'` placeholder.
        For example:  `'dataset/*.txt'`

        The parts of the filename that match the placeholder are used as keys.

        Note that data loading is NOT performed while calling this method.
        Loading is postponed until the data is actually required.
        This lets additional arguments to be provided to the
        :class:`~tramway.analyzer.RWAnalyzer` :attr:`~tramway.analyzer.RWAnalyzer.spt_data`
        attribute before the data are loaded.

        See also :class:`SPTAsciiFiles`.
        """
        self.specialize( SPTAsciiFiles, filepattern )
    def from_dataframe(self, df):
        """
        See also :class:`StandaloneSPTDataFrame`.
        """
        self.specialize( StandaloneSPTDataFrame, df )
    def from_dataframes(self, dfs):
        """
        See also :class:`SPTDataFrames`.
        """
        self.specialize( SPTDataFrames, dfs )
    def from_mat_file(self, filepath):
        """
        Sets a MatLab V7 file as the source of SPT data.

        Similarly to :meth:`from_ascii_file`, data loading is lazy.

        See also :class:`StandaloneSPTMatFile`.
        """
        self.specialize( StandaloneSPTMatFile, filepath )
    def from_mat_files(self, filepattern):
        """
        Sets MatLab V7 files, which paths match with a pattern, as the source of SPT data.

        `filepattern` is a standard filepath with the :const:`'*'` placeholder.
        For example:  `'datasets/*.txt'`

        The parts of the filename that match the placeholder are used as keys.

        Similarly to :meth:`from_ascii_files`, data loading is lazy.

        See also :class:`SPTMatFiles`.
        """
        self.specialize( SPTMatFiles, filepattern )
    def from_rwa_file(self, filepath):
        """
        Similar to :meth:`from_ascii_file`, for *.rwa* files.

        See also :class:`StandaloneRWAFile`.
        """
        self.specialize( StandaloneRWAFile, filepath )
    def from_rwa_files(self, filepattern):
        """
        Similar to :meth:`from_ascii_files`, for *.rwa* files.

        See also :class:`RWAFiles`.
        """
        self.specialize( RWAFiles, filepattern )
    def from_rw_generator(self, generator):
        """
        A random walk generator features a :meth:`generate` method.

        See also :class:`RWGenerator`.
        """
        self.specialize( RWGenerator, generator )
    def from_analysis_tree(self, analyses, copy=False):
        self.specialize( RWAnalyses, analyses, copy )
    def from_tracker(self):
        """ This initializer method does not need to be called;
        The :class:`~tramway.analyzer.RWAnalyzer`
        :attr:`~tramway.analyzer.RWAnalyzer.tracker` attribute does this automatically.

        See also :class:`TrackerOutput`.
        """
        self.specialize( TrackerOutput )



class StandaloneDataItem(object):
    """
    Partial implementation for single data item
    :class:`~tramway.analyzer.spt_data.SPTData` attribute.
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
        if isinstance(self, Proxy) and not isinstance(new_self, Proxy):
            new_self = type(self)(new_self)
        if new_self is not self:
            self._parent._spt_data = new_self
            new_self._parent = self._parent
            try:
                roi_central = next(iter(self)).roi._global
                roi_central.reset()
            except AttributeError:
                pass
            else:
                for f in new_self:
                    roi_central._register_decentralized_roi(f)
    def reload_from_rwa_files(self, skip_missing=False):
        StandaloneRWAFile.reload_from_rwa_files(self, skip_missing=skip_missing)


class HasAnalysisTree(HasROI):
    """
    Partial implementation for :class:`SPTData` that complements :class:`SPTParameters`.
    """
    __slots__ = ('_analyses','_frame_interval_cache','_localization_error_cache')
    def __init__(self, df=None, **kwargs):
        self._frame_interval_cache = self._localization_error_cache = None
        HasROI.__init__(self, **kwargs)
        self.analyses = Analyses(df, standard_metadata(), autosave=True)
        self._analyses.hooks.append(lambda _: self.commit_cache(autoload=True))
    @property
    def _frame_interval(self):
        return self._frame_interval_cache
    @_frame_interval.setter
    def _frame_interval(self, dt):
        self._frame_interval_cache = dt
        if not islazy(self._dataframe) and self._dataframe is not None:
            self._dataframe.frame_interval = dt
    @property
    def _localization_error(self):
        return self._localization_error_cache
    @_localization_error.setter
    def _localization_error(self, err):
        self._localization_error_cache = err
        if not islazy(self._dataframe) and self._dataframe is not None:
            self._dataframe.localization_error = err
    def commit_cache(self, autoload=False):
        """
        Pushes the cached parameters into the :attr:`dataframe` object.
        """
        if islazy(self._dataframe) or self._dataframe is None:
            if not autoload:
                raise RuntimeError('the SPT data has not been loaded')
        self.dataframe.frame_interval = self._frame_interval_cache
        self.dataframe.localization_error = self._localization_error_cache
    def clear_cache(self):
        """
        Clears the cached values and reads from the :attr:`dataframe` object.
        """
        self._frame_interval_cache = self._localization_error_cache = None
        if not islazy(self._dataframe) and self._dataframe is not None:
            try:
                self._frame_interval_cache = self._dataframe.frame_interval
            except AttributeError:
                pass
            try:
                self._localization_error_cache = self._dataframe.localization_error
            except AttributeError:
                pass
    def check_cache(self, _raise=AttributeError):
        """
        Checks the parameter cache integrity.

        If differences are found with the values in :attr:`dataframe`,
        :meth:`check_cache` raises an exception of type `_raise`.

        If `_raise` is :const:`None` or :const:`False`, then :meth:`check_cache` returns
        a `bool` instead, that is :const:`False` if the cache is alright,
        :const:`True` otherwise.
        If `_raise` is :const:`True`, then :meth:`check_cache` returns :const:`True` if
        the cache is alright, :const:`False` otherwise.
        """
        if _raise is None:
            _raise = False
        if isinstance(_raise, bool):
            _return = _raise
            _raise = None
        elif isinstance(_raise, Exception):
            _return = None
        elif isinstance(_raise, type) and issubclass(_raise, Exception):
            _return = None
            _raise = _raise('cache integrity is compromised')
        else:
            raise TypeError('unsupported type for second argument: {}'.format(type(_raise)))
        if not islazy(self._dataframe) and self._dataframe is not None:
            try:
                ok = self._frame_interval_cache == self._dataframe.frame_interval
            except AttributeError:
                pass
            else:
                if not ok:
                    if _return is None: # _raise is not None
                        raise _raise from None
                    else:
                        return not _return
            try:
                ok = self._localization_error_cache == self._dataframe.localization_error
            except AttributeError:
                pass
            else:
                if not ok:
                    if _return is None: # _raise is not None
                        raise _raise from None
                    else:
                        return not _return
        # the cache is alright
        if _return is not None:
            return _return
    def set_analyses(self, tree):
        if isinstance(tree, AutosaveCapable):
            self._analyses = tree
            tree = tree.analyses.statefree()
        else:
            autosaver = Analyses(tree, autosave=True)
            self._analyses = autosaver
        #
        err_cache = self._localization_error_cache
        dt_cache = self._frame_interval_cache
        if islazy(tree._data):
            store = tree._data.store
            record = store.getRecord(tree._data.locator, store.store)
            try:
                err = store.peek('localization_error', record)
            except KeyError:
                err = None
            try:
                dt = store.peek('frame_interval', record)
            except KeyError:
                dt = None
        else:
            df = tree.data
            try:
                dt = df.frame_interval
            except AttributeError:
                dt = None
            try:
                err = df.localization_error
            except AttributeError:
                err = None
        if err is not None:
            if not (err_cache is None or err_cache == err):
                self.logger.warning("localization error does not match with record: {} != {}".format(err_cache, err))
            self._localization_error_cache = err
        if dt is not None:
            if not (dt_cache is None or dt_cache == dt):
                self.logger.warning("frame interval does not match with record: {} != {}".format(dt_cache, dt))
            self._frame_interval_cache = dt
    @property
    def analyses(self):
        return self._analyses
    @analyses.setter
    def analyses(self, tree):
        self.set_analyses(tree)
    @property
    def _dataframe(self):
        return self._analyses._data
    @_dataframe.setter
    def _dataframe(self, df):
        self._analyses._data = df
    @property
    def dataframe(self):
        return self.analyses.data


class _SPTDataFrame(HasAnalysisTree, SPTParameters):
    """
    Basis for all concrete :class:`SPTDataItem` classes.
    """
    __slots__ = ('_source',)
    def __init__(self, df, source=None, **kwargs):
        prms = SPTParameters.__parse__(kwargs)
        self._source = source
        HasAnalysisTree.__init__(self, df, **kwargs)
        SPTParameters.__init__(self, *prms)
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
    def set_precision(self, precision):
        dtypes = compute_dtypes(self.dataframe, precision)
        self._dataframe = self.dataframe.astype(dtypes)
    def to_ascii_file(self, filepath, columns=None, header=True, float_format='%.4f', **kwargs):
        """
        Exports the data to text file.

        Arguments:

            filepath (str): output filepath.

            columns (sequence of *str*): columns to be exported.

            header (bool): print column names on the first line.

            float_format (str): see also :meth:`pandas.DataFrame.to_csv`.

        Additional keyword arguments are passed to :meth:`pandas.DataFrame.to_csv`.
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
        df.to_csv(filepath, sep='\t', index=False, header=header, float_format=float_format, **kwargs)
    def to_rwa_file(self, filepath, **kwargs):
        """
        Exports the analysis tree to file.

        Calls :func:`~tramway.core.hdf5.store.save_rwa` with argument
        ``compress=False`` unless explicitly set.
        """
        if self.analyses.data is None:
            raise ValueError('no data available')
        try:
            analyses = self.analyses.analyses.statefree()
        except AttributeError:
            analyses = self.analyses
        if 'compress' not in kwargs:
            kwargs['compress'] = False
        save_rwa(filepath, analyses, **kwargs)
    def set_analyses(self, tree):
        assert bool(tree.metadata)
        if self.source is not None:
            tree.metadata['datafile'] = self.source
        HasAnalysisTree.set_analyses(self, tree)
    def add_sampling(self, sampling, label, comment=None):
        analyses = self.analyses
        label = analyses.autoindex(label)
        subtree = lazy.Analyses(sampling, standard_metadata())
        analyses.add(subtree, label, comment)
        return label
    def get_sampling(self, label):
        return Analysis.get_analysis(self.analyses, label)
    def autosaving(self, *args, overwrite=True, **kwargs):
        assert isinstance(self.analyses, AutosaveCapable)
        if not self.analyses.rwa_file:
            if self.source:
                self.analyses.rwa_file = os.path.splitext(self.source)[0]+'.rwa'
                if self.analyses.rwa_file == self.source:
                    if not overwrite:
                        i = 0
                        while os.path.isfile(os.path.expanduser(self.analyses.rwa_file)):
                            i += 1
                            self.analyses.rwa_file = '{}-{:d}.rwa'.format(os.path.splitext(self.source)[0],i)
            else:
                self.logger.warning('no output filename defined')
        return self.analyses.autosaving(*args, **kwargs)
    @property
    def _mpl_impl(self):
        from .mpl import Mpl
        return Mpl
    @property
    def mpl(self):
        """ tramway.analyzer.spt_data.mpl.Mpl: Matplotlib utilities """
        return self._mpl_impl(self)


class SPTDataFrame(_SPTDataFrame):
    __slots__ = ()

SPTDataItem.register(SPTDataFrame)


class StandaloneSPTDataFrame(_SPTDataFrame, StandaloneDataItem):
    """
    :class:`SPTData` attribute for single dataframes.
    """
    __slots__ = ()

SPTData.register(StandaloneSPTDataFrame)


class SPTDataFrames(SPTDataIterator):
    """
    :class:`SPTData` attribute for multiple dataframes.
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
    """
    Basis for :class:`SPTDataItem` classes for data in a file.
    """
    __slots__ = ('_alias','_reset_origin','_discard_static_trajectories')
    def __init__(self, filepath, dataframe=None, **kwargs):
        self._alias = None
        self._reset_origin = False
        self._discard_static_trajectories = False
        _SPTDataFrame.__init__(self, dataframe, filepath, **kwargs)
    @property
    def filepath(self):
        """
        *str*: Alias for the :attr:`source` property
        """
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
        """
        *str*: Identifier, shorter than :attr:`source`
        """
        return self._alias
    @alias.setter
    def alias(self, name):
        if callable(name):
            self._alias = name(self.filepath)
        else:
            self._alias = name
    def get_analyses(self):
        if self._analyses._data is None:
            self.load()
            assert self._analyses._data is not None
        return self._analyses
    @property
    def analyses(self):
        return self.get_analyses()
    @analyses.setter
    def analyses(self, tree):
        self.set_analyses(tree)
    def _trigger_discard_static_trajectories(self):
        """
        Calls :meth:`SPTDataFrame.discard_static_trajectories`;
        requires the data are already loaded.
        """
        if self._discard_static_trajectories is True:
            SPTDataFrame.discard_static_trajectories(self)
        elif self._discard_static_trajectories: # not in (None, False)
            SPTDataFrame.discard_static_trajectories(self, **self._discard_static_trajectories)
    def _trigger_reset_origin(self):
        """
        Calls :meth:`SPTDataFrame.reset_origin`;
        requires the data are already loaded.
        """
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
        Looks for the corresponding localization microscopy image in the
        :attr:`~tramway.analyzer.RWAnalyzer.images` attribute.

        The search is based on the object's alias.
        The first item that contains the alias in its filename is returned as a match.

        If the :attr:`alias` attribute is not set or images do not have a defined `filepath`,
        an :class:`AttributeError` exception is raised.

        The optional argument *match* is a 2-argument *callable* that
        takes the alias (*str*) of *self* and the filepath (*str*) of an
        :class:`~tramway.analyzer.images.Image` stack of images, and returns
        :const:`True` if the filepath and alias match.
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
    """
    Basis for :class:`SPTDataItem` classes for raw data in a file,
    possibly with non-standard column names or data units.
    """
    __slots__ = ('_columns',)
    def __init__(self, filepath, dataframe=None, **kwargs):
        self._columns = None
        SPTFile.__init__(self, filepath, dataframe, **kwargs)
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
        #self._filepattern = filepattern # should work in most cases
        if isinstance(filepattern, str):
            self._filepattern = os.path.expanduser(filepattern)
        else:
            self._filepattern = [os.path.expanduser(pattern) for pattern in filepattern]
    @property
    def filepattern(self):
        """
        *str*: Filepath glob pattern
        """
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
        """
        *list*: SPT file objects
        """
        if not self._files:
            self.list_files()
        return self._files
    @property
    def filepaths(self):
        """
        *list* of *str*: File paths (copy)
        """
        return [ f.filepath for f in self.files ]
    @property
    def dataframes(self): # this also hides the parent class' setter
        return [ f.dataframe for f in self.files ]
    @property
    def partially_reified(self):
        """
        *bool*: :const:`True` if any file has been loaded
        """
        return self._files and any([ f.reified for f in self._files ])
    @property
    def fully_reified(self):
        """
        *bool*: :const:`True` if all the files have been loaded
        """
        return self._files and all([ f.reified for f in self._files ])
    @property
    def reified(self):
        return self.fully_reified
    def __len__(self):
        return len(self.files)
    def __iter__(self):
        yield from self.files
    def list_files(self):
        """
        Interprets the filepath glob pattern and lists the matching files.
        """
        if self.filepattern is None:
            raise ValueError('filepattern is not defined')
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
        """
        *callable*: Function that extracts an alias out of filepaths (**write-only property**)
        """
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
        """
        *list* of *str*: Aliases of the SPT data items (copy)
        """
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
    :class:`SPTData` class for multiple SPT text files.
    """
    __slots__ = ()
    def list_files(self):
        SPTFiles.list_files(self)
        self._files = [ self._bear_child( SPTAsciiFile, filepath ) for filepath in self._files ]

SPTData.register(SPTAsciiFiles)


class _RWAFile(SPTFile):
    __slots__ = ('_filepath',)
    def __init__(self, filepath, **kwargs):
        self._filepath = filepath
        SPTFile.__init__(self, filepath, None, **kwargs)
    @property
    def filepath(self):
        """
        *str*: *.rwa* file path; distinct from :attr:`source`.
        """
        return self._filepath
    @filepath.setter
    def filepath(self, fp):
        if self.reified:
            raise AttributeError("file '{}' has already been loaded; cannot set the filepath anymore".format(self.filepath.split('/')[-1]))
        else:
            self._filepath = fp
    def set_analyses(self, tree):
        HasAnalysisTree.set_analyses(self, tree)
    def load(self):
        # ~ expansion is no longer necessary from rwa-python==0.8.4
        try:
            tree = load_rwa(os.path.expanduser(self.filepath),
                    lazy=True, force_load_spt_data=False)
            assert islazy(tree._data)
            self.analyses = tree
        except KeyError as e:
            raise RWAFileException(self.filepath, e) from None
        self._trigger_discard_static_trajectories()
        self._trigger_reset_origin()
    @classmethod
    def __reload__(cls, self, parent=None, filepath=None):
        """
        To be called as either:

        .. code-block: python

            RWAFile.__reload__(spt_data_item)

        or:

        .. code-block: python

            StandaloneRWAFile.__reload__(spt_data)

        In the first case, the `parent` argument may be useful if the call to :meth:`__reload__`
        is part of a general reload of the :class:`SPTData` object which `spt_data_item` is an
        :class:`SPTDataItem` item.
        `parent` should point at the new reloaded :class:`SPTData` object.

        The returned object is of the most specialized type of `cls` and `type(self)`.

        """
        if filepath is None:
            if isinstance(self, _RWAFile):
                filepath = self.filepath
            else:
                filepath = os.path.splitext(self.source)[0]+'.rwa'
        if not os.path.isfile(os.path.expanduser(filepath)):
            raise FileNotFoundError(filepath)
        # favor the most specialized type
        if isinstance(self, cls):
            cls = type(self)
        #
        reloaded = cls(filepath, parent=self._parent if parent is None else parent)
        # should be readily available unless cache integrity has been compromised
        reloaded._frame_interval = self.frame_interval
        reloaded._localization_error = self.localization_error
        #
        reloaded._alias = self.alias
        #
        reloaded._roi = copy.copy(self.roi)
        reloaded.roi._parent = reloaded
        #
        self.analyses.terminate()
        reloaded.analyses.rwa_file = self.analyses.rwa_file
        reloaded.analyses.autosave = self.analyses.autosave
        return reloaded

class RWAFile(_RWAFile):
    __slots__ = ()

SPTDataItem.register(RWAFile)

class StandaloneRWAFile(_RWAFile, StandaloneDataItem):
    """
    :class:`SPTata` class for single RWA files.
    """
    __slots__ = ()
    def reload_from_rwa_files(self, skip_missing=False):
        if skip_missing:
            self.logger.error('cannot omit the only rwa file; ignoring skip_missing')
        cls = type(self) if isinstance(self, StandaloneRWAFile) else StandaloneRWAFile
        self.self_update( cls.__reload__(self) )

SPTData.register(StandaloneRWAFile)


class RWAFiles(SPTFiles):
    """
    :class:`SPTData` class for multiple RWA files.
    """
    __slots__ = ()
    def list_files(self):
        SPTFiles.list_files(self)
        self._files = [ self._bear_child( RWAFile, filepath ) for filepath in self._files ]
    def reload_from_rwa_files(self, skip_missing=False):
        cls = type(self) if isinstance(self, RWAFiles) else RWAFiles
        assert cls is RWAFiles
        self.self_update( cls.__reload__(self, skip_missing=skip_missing) )
    @classmethod
    def __reload__(cls, self, skip_missing=False, parent=None):
        if isinstance(self, cls):
            # favor the most specialized type
            cls = type(self)
        reloaded = cls([], parent=self._parent if parent is None else parent)
        reloaded.filepattern = None
        reloaded._files = []
        for f in self:
            try:
                f = RWAFile.__reload__(f, parent=reloaded)
            except FileNotFoundError:
                if not skip_missing:
                    raise
            else:
                reloaded._files.append(f)
        if not reloaded._files:
            raise RuntimeError('could not reload any .rwa file')
        return reloaded

SPTData.register(RWAFiles)


class _SPTMatFile(RawSPTFile):
    __slots__ = ('_coord_scale',)
    def __init__(self, filepath, dataframe=None, **kwargs):
        self._coord_scale = None
        RawSPTFile.__init__(self, filepath, dataframe, **kwargs)
    @property
    def coord_scale(self):
        r"""
        *float*:
            Convertion factor for the loaded coordinates so that they read in :math:`\mu m`
        """
        return self._coord_scale
    @coord_scale.setter
    def coord_scale(self, scale):
        if self.reified:
            raise AttributeError('the SPT data have already been loaded; cannot set the coordinate scale anymore')
        else:
            self._coord_scale = scale
    @property
    def pixel_size(self):
        """ *float*: Former name for :attr:`coord_scale`; **deprecated** """
        self.logger.warning('attribute pixel_size is deprecated; use coord_scale instead')
        return self.coord_scale
    @pixel_size.setter
    def pixel_size(self, siz):
        self.logger.warning('attribute pixel_size is deprecated; use coord_scale instead')
        self.coord_scale = siz
    def load(self):
        try:
            self._dataframe = load_mat(os.path.expanduser(self.filepath),
                    columns=self._columns, dt=self._frame_interval, coord_scale=self.coord_scale)
        except OSError as e:
            raise OSError('{}\nwhile loading file: {}'.format(e, self.source)) from None
        self._trigger_discard_static_trajectories()
        self._trigger_reset_origin()

class SPTMatFile(_SPTMatFile):
    __slots__ = ()

SPTDataItem.register(SPTMatFile)

class StandaloneSPTMatFile(_SPTMatFile, StandaloneDataItem):
    """
    :class:`SPTData` class for single MatLab v7 data files.
    """
    __slots__ = ()

SPTData.register(StandaloneSPTMatFile)


class SPTMatFiles(RawSPTFiles):
    """
    :class:`SPTData` class for multiple MatLab v7 data files.
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
    coord_scale.__doc__ = SPTMatFile.coord_scale.__doc__
    @property
    def pixel_size(self):
        """ *float*: Former name for :attr:`coord_scale`; **deprecated** """
        self.logger.warning('attribute pixel_size is deprecated; use coord_scale instead')
        return self.coord_scale
    @pixel_size.setter
    def pixel_size(self, px):
        self.logger.warning('attribute pixel_size is deprecated; use coord_scale instead')
        self.coord_scale = px
    def list_files(self):
        SPTFiles.list_files(self)
        self._files = [ self._bear_child( SPTMatFile, filepath ) for filepath in self._files ]

SPTData.register(SPTMatFiles)


class RWGenerator(SPTDataFrame):
    """ not implemented yet """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class RWAnalyses(StandaloneSPTDataFrame):
    """
    :class:`SPTData` class for single analysis trees as stored in *.rwa* files.

    .. note::

        copying (initializing with ``copy=True``) copies only the SPT data
        and NOT the subsequent analyses.

    .. warning::

        not thouroughly tested.

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
    :class:`SPTData` class for multiple analysis trees as stored in *.rwa* files.

    .. note::

        copying (initializing with ``copy=True``) copies only the SPT data
        and NOT the subsequent analyses.

    .. warning::

        not tested.

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
    """
    Makes :class:`SPTDataFrames` initialization possible with empty data.

    To be wrapped in the :class:`SPTDataFrame` object for storage of SPT parameters.

    As soon as proper SPT datablocks are appended, the fake SPT datablock
    should be removed and the previously defined SPT parameters copied into
    the other datablocks.
    """
    __slots__ = () # useless as DataFrame defines __dict__
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, columns=list('nxyt'))

class LocalizationFile(SPTFile):
    __slots__ = ()

SPTDataItem.register(LocalizationFile)

class TrackerOutput(SPTDataFrames):
    """
    :class:`SPTData` class for SPT data yielded by the
    :attr:`~tramway.analyzer.RWAnalyzer.tracker` attribute.
    """
    __slots__ = ()
    def __init__(self, **kwargs):
        SPTDataFrames.__init__(self, [_FakeSPTData()], **kwargs)
    def add_tracked_data(self, trajectories, source=None, filepath=None):
        """
        .. note::

            In most cases `source` and `filepath` are supposed to represent
            the same piece of information.
            Here, `filepath` should be preferred over `source` if the localization
            data come from a file.

        """
        if filepath:
            df = self._bear_child( LocalizationFile, filepath, trajectories )
        else:
            df = self._bear_child( SPTDataFrame, trajectories, source )
        siblings = self._dataframes[0]
        df.frame_interval = siblings.frame_interval
        df.localization_error = siblings.localization_error
        if isinstance(siblings.dataframe, _FakeSPTData):
            assert not self._dataframes[1:]
            self._dataframes = [ df ]
        else:
            if tuple(df.columns) != tuple(siblings.columns):
                raise ValueError("not all the dataframes feature the same column names")
            self._dataframes.append( df )
    # borrowed from `SPTFiles`
    @property
    def alias(self):
        raise AttributeError('alias is write-only')
    @alias.setter
    def alias(self, name):
        if callable(name):
            for f in self._dataframes:
                f.alias = name
        else:
            raise TypeError('global alias is not callable')
    @property
    def aliases(self):
        return [ f.alias for f in self._dataframes ]
    def filter_by_source(self, source_filter, return_index=False):
        yield from SPTFiles.filter_by_source(self, source_filter, return_index=return_index)


__all__ = [ 'SPTData', 'SPTDataItem', 'SPTParameters', 'StandaloneDataItem', 'SPTDataIterator',
        'SPTDataInitializer', 'HasROI', 'HasAnalysisTree', '_SPTDataFrame', 'SPTDataFrame',
        'StandaloneSPTDataFrame', 'SPTDataFrames',
        'SPTFile', 'RawSPTFile', 'SPTFiles', 'RawSPTFiles',
        '_SPTAsciiFile', 'SPTAsciiFile', 'StandaloneSPTAsciiFile', 'SPTAsciiFiles',
        '_RWAFile', 'RWAFile', 'StandaloneRWAFile', 'RWAFiles',
        '_SPTMatFile', 'SPTMatFile', 'StandaloneSPTMatFile', 'SPTMatFiles',
        'RWAnalyses', 'MultipleRWAnalyses', 'RWGenerator', 'LocalizationFile', 'TrackerOutput',
        'compute_dtypes', 'glob' ]

