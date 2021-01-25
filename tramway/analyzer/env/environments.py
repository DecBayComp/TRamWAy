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
from .abc import *
from ..spt_data.abc import SPTData
from ..spt_data import _normalize
from ..roi.abc import ROI
from ..time.abc import Time
import os
import sys
import time
import multiprocessing
import subprocess
import tempfile
import shutil
import glob
import traceback
from tramway.core.hdf5.store import load_rwa, save_rwa
from tramway.core.analyses.base import append_leaf


def join_arguments(args):
    _args = []
    args = iter(args)
    while True:
        try:
            arg = next(args)
        except StopIteration:
            break
        if arg.startswith('--'):
            try:
                key, val = arg[2:].split('=')
            except ValueError:
                return []
            if val[0] == '"':
                cont = []
                while val[-1] != '"':
                    try:
                        val = next(args)
                    except StopIteration:
                        return []
                    cont.append(val)
                if cont:
                    arg = ' '.join([arg]+cont)
            _args.append(arg)
        else:
            return []
    return _args


class Env(AnalyzerNode):
    """
    Implements parts of classes suitable for the :class:`~tramway.analyzer.RWAnalyzer`
    :attr:`~tramway.analyzer.RWAnalyzer.env` main attribute.

    See :class:`LocalHost` or :class:`SlurmOverSSH` for examples of concrete classes.
    """
    __slots__ = ('_interpreter','_script','_working_directory','_worker_count',
            '_pending_jobs','_selectors','_selector_classes','_temporary_files',
            '_collectibles','debug')
    def __init__(self, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._interpreter = 'python3'
        self._script = None
        self._selectors = None
        self._selector_classes = {}
        self._temporary_files = []
        self._working_directory = None
        self.collectibles = None
        self._worker_count = None
        self.pending_jobs = []
        self.debug = False
    @property
    def logger(self):
        """
        Parent analyzer's logger
        """
        return self._parent.logger
    @property
    def working_directory(self):
        """
        *str*: Path of the working directory on the worker side
        """
        return self._working_directory
    @working_directory.setter
    def working_directory(self, wd):
        self._working_directory = wd
    @property
    def wd(self):
        return self.working_directory
    wd.__doc__ = working_directory.__doc__
    @wd.setter
    def wd(self, wd):
        self.working_directory = wd
    @property
    def wd_is_available(self):
        """
        *bool*: :const:`True` if the working directory is ready on the worker side
        """
        return True
    @property
    def interpreter(self):
        """
        *str*: Interpreter command on the worker side
        """
        return self._interpreter
    @interpreter.setter
    def interpreter(self, cmd):
        self._interpreter = cmd
    @property
    def script(self):
        """
        *str*: Path to the local script to be executed; meaningful on the submit side only
        """
        return self._script
    @script.setter
    def script(self, file):
        if file is None:
            self._script = None
        elif file.startswith('/'):
            self._script = file
        else:
            self._script = os.path.abspath(os.path.expanduser(file))
    @property
    def worker_count(self):
        """
        *int*: Desired number of workers
        """
        return self._worker_count
    @worker_count.setter
    def worker_count(self, wc):
        self._worker_count = wc
    @property
    def wc(self):
        return self.worker_count
    wc.__doc__ = worker_count.__doc__
    @wc.setter
    def wc(self, wc):
        self.worker_count = wc
    @property
    def pending_jobs(self):
        """
        *list* of *str*: Specification of the jobs to be submitted
        """
        return self._pending_jobs
    @pending_jobs.setter
    def pending_jobs(self, jobs):
        self._pending_jobs = jobs
    @property
    def selectors(self):
        """
        *dict*: Wrapper classes for main :class`~tramway.analyzer.RWAnalyzer` attributes
        """
        return self._selectors
    @selectors.setter
    def selectors(self, sel):
        self._selectors = sel
    @property
    def temporary_files(self):
        """
        *list* of *str*: Temporary files generated on the local side
        """
        return self._temporary_files
    @property
    def analyzer(self):
        return self._parent
    @property
    def collectibles(self):
        return self._collectibles
    @collectibles.setter
    def collectibles(self, cs):
        if cs:
            self._collectibles = set(cs)
        else:
            self._collectibles = set()
    def pending_collectibles(self):
        """
        *set* of *str*: Names or paths of files to be retrieved from the worker side to the submit side
        """
        return self.collectibles
    @property
    def current_stage(self):
        """
        *int* or *list* of *int*: Index of the current stage; meaningful on the submit side only
        """
        assert self.worker_side
        return self.selectors['stage_index']
    def early_setup(self, *argv):
        """
        Determines which side is running and sets the `submit_side`/`worker_side` attributes.

        Takes command-line arguments (``sys.argv``).
        """
        assert argv
        # join arguments with spaces
        args = join_arguments(argv[1:])
        #
        valid_keys = set(('stage-index', 'source', 'region-index', 'segment-index', 'cell-index',
            'working-directory'))
        valid_arguments = {}
        for arg in args:
            if arg.startswith('--'):
                try:
                    key, val = arg[2:].split('=')
                except ValueError:
                    valid_arguments = None
                else:
                    if key in valid_keys:
                        key = key.replace('-', '_')
                        vals = val.split(',')
                        val, vals = vals[0], vals[1:]
                        if val[0]=='"' and val[-1]=='"':
                            val = val[1:-1]
                            _vals = [val]
                            for v in vals:
                                if v[0]=='"' and v[-1]=='"':
                                    _vals.append(v[1:-1])
                                else:
                                    valid_arguments = None
                                    break
                        else:
                            try:
                                val = int(val)
                            except ValueError:
                                valid_arguments = None
                                break
                            _vals = [val]
                            for v in vals:
                                try:
                                    _vals.append(int(v))
                                except ValueError:
                                    valid_arguments = None
                                    break
                        if vals:
                            val = tuple(_vals)
                        valid_arguments[key] = val
                    else:
                        valid_arguments = None
            else:
                valid_arguments = None
            if valid_arguments is None:
                break
        if valid_arguments:
            # worker side
            _valid_arguments = dict(valid_arguments) # copy
            self.wd = _valid_arguments.pop('working_directory', self.wd)
            self.selectors = _valid_arguments
        return valid_arguments
    def setup(self, *argv):
        """
        Determines which side is running and alters iterators of the main
        :class:`~tramway.analyzer.RWAnalyzer` attributes.

        Takes command-line arguments (``sys.argv``).
        """
        valid_arguments = self.early_setup(*argv)
        if valid_arguments:
            # worker side
            self.wd = valid_arguments.pop('working_directory', self.wd)
            self.selectors = valid_arguments
            #self.logger.debug('the following selectors apply to the current job:\n\t{}'.format(self.selectors))
            #
            if self.script is not None and self.script.endswith('.ipynb'):
                self.script = self.script[:-5]+'py'
            #
            try:
                sources = valid_arguments['source']
            except KeyError:
                pass
            else:
                if isinstance(sources, str):
                    sources = os.path.expanduser(sources)
                else:
                    sources = tuple([
                        os.path.expanduser(source) for source in sources
                        ])
                self.selectors['source'] = sources
                #self.logger.debug('selecting source: '+', '.join((sources,) if isinstance(sources, str) else sources))
            #
            for f in self.analyzer.spt_data: # TODO: check why self.spt_data_selector(..) does not work
                f.analyses.rwa_file = self.make_temporary_file(suffix='.rwa', output=True)
                f.analyses.autosave = True
        elif self.script is None:
            # not tested!
            candidate_files = [ f for f in os.listdir() \
                    if f.endswith('.py') or f.endswith('.ipynb') ]
            if candidate_files and not candidate_files[1:]:
                self.script = candidate_files[0]
                self.logger.info('candidate script: {} (in {})'.format(self.script, os.getcwd()))
            raise ValueError('attribute `script` is not set')
        else:
            self.make_working_directory()
            self.logger.info('working directory: '+self.wd)
    def spt_data_selector(self, spt_data_attr):
        """
        Wraps the :attr:`~tramway.analyzer.RWAnalyzer.spt_data` attribute.
        """
        if isinstance(spt_data_attr, Initializer) or isinstance(spt_data_attr, Proxy):
            return spt_data_attr
        cls = type(spt_data_attr)
        try:
            selector_cls = self._selector_classes[cls]
        except KeyError:
            try:
                sources = self.selectors['source']
            except KeyError:
                return spt_data_attr
            if isinstance(sources, str):
                sources = (sources,)
            #logger = self.logger
            try:
                alias = all([ bool(f.alias) for f in self._eldest_parent.spt_data ])
            except AttributeError:
                alias = False
            if alias:
                aliases = sources # TODO: this can be checked
                class selector_cls(Proxy):
                    __slots__ = ()
                    def __iter__(self):
                        for f in self._proxied:
                            if f.alias in aliases:
                                #logger.debug('source {} selected'.format(f.source))
                                yield f
            else:
                sources = set([ _normalize(_s) for _s in sources ])
                class selector_cls(Proxy):
                    __slots__ = ()
                    def __iter__(self):
                        for f in self._proxied:
                            if _normalize(f.source) in sources:
                                #logger.debug('source {} selected'.format(f.source))
                                yield f
            SPTData.register(selector_cls)
            self._selector_classes[cls] = selector_cls
        return selector_cls(spt_data_attr)
    def roi_selector(self, roi_attr):
        """
        Wraps the :attr:`~tramway.analyzer.RWAnalyzer.roi` attribute and/or the individual
        :attr:`..spt_data.SPTDataItem.roi` attributes.
        """
        if isinstance(roi_attr, Initializer) or isinstance(roi_attr, Proxy):
            return roi_attr
        cls = type(roi_attr)
        try:
            selector_cls = self._selector_classes[cls]
        except KeyError:
            try:
                region = self.selectors['region_index']
            except KeyError:
                return roi_attr
            regions = set([region]) if isinstance(region, int) else set(region)
            def _region(index_arg):
                if index_arg is None:
                    return region
                elif callable(index_arg):
                    return lambda r: r in regions and index_arg(r)
                elif index_arg in regions:
                    return index_arg # or lambda r: True
                else:
                    return lambda r: False
            class selector_cls(Proxy):
                __slots__ = ()
                def as_support_regions(self, index=None, source=None, return_index=False):
                    yield from self._proxied.as_support_regions(_region(index), source, return_index)
                def as_individual_roi(self, *args, **kwargs):
                    raise NotImplementedError
            ROI.register(selector_cls)
            self._selector_classes[cls] = selector_cls
        return selector_cls(roi_attr)
    def time_selector(self, time_attr):
        """
        Wraps the :attr:`~tramway.analyzer.RWAnalyzer.time` attribute.
        """
        if isinstance(time_attr, Initializer) or isinstance(time_attr, Proxy):
            return time_attr
        cls = type(time_attr)
        try:
            segment = self.selectors['segment_index']
        except KeyError:
            return time_attr
        else:
            segments = set([segment]) if isinstance(segment, int) else set(segment)
            def _segment(index_arg):
                if index_arg is None:
                    return segment
                elif callable(index_arg):
                    return lambda t: t in segments and index_arg(t)
                elif index_arg in segments:
                    return index_arg # or lambda t: True
                else:
                    return lambda t: False
            logger = self.logger
            class selector_cls(Proxy):
                __slots__ = ()
                def as_time_segments(self, sampling, maps=None, index=None, return_index=False, return_times=True):
                    yield from self._proxied.as_time_segments(sampling, maps, _segment(index), return_index, return_times)
            Time.register(selector_cls)
            self._selector_classes[cls] = selector_cls
        return selector_cls(time_attr)
    @property
    def submit_side(self):
        """
        *bool*: :const:`True` if currently running on the submit side
        """
        return self.selectors is None
    @property
    def worker_side(self):
        """
        *bool*: :const:`True` if currently running on the worker side
        """
        return self.selectors is not None
    def make_working_directory(self):
        """
        """
        assert self.submit_side
        self.wd = self.make_temporary_file(directory=True)
    def make_temporary_file(self, output=False, directory=False, **kwargs):
        """
        Arguments:

            output (bool): do not delete the file once the task is done.

            directory (bool): make a temporary directory.

        More keyword arguments can be passed to :func:`tempfile.mkstemp`.
        See for example *suffix*.
        """
        if self.wd_is_available:
            parent_dir = self.working_directory
        else:
            parent_dir = None # standard /tmp location
        if directory:
            tmpfile = tempfile.mkdtemp(dir=parent_dir)
        else:
            fd, tmpfile = tempfile.mkstemp(dir=parent_dir, **kwargs)
            os.close(fd)
        if not output:
            self._temporary_files.append(tmpfile)
        return tmpfile
    def dispatch(self, **kwargs):
        """
        Prepares the worker side. To be called from the submit side only.
        """
        if not kwargs:
            self.prepare_script()
            return True
    def __del__(self):
        #self.delete_temporary_data()
        pass
    def delete_temporary_data(self):
        """
        Deletes all the temporary data, on both the submit and worker sides.
        """
        for file in self._temporary_files[::-1]:
            if os.path.isdir(file):
                try:
                    shutil.rmtree(file)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    self.logger.warning('temporary files removal failed with the following error:\n{}'.format(e))
            elif os.path.isfile(file):
                try:
                    os.unlink(file)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    self.logger.debug('temporary file removal failed with the following error:\n{}'.format(e))
        self._temporary_files = []
    def make_job(self, stage_index=None, source=None, region_index=None, segment_index=None):
        """
        Registers a new pending job.

        Arguments:

            stage_index (*int* or *list* of *int*): stage index(ices)

            source (str): SPT datablock identifier (source path or alias)

            region_index (int): index of the support region
                (see also :meth:`~tramway.analyzer.roi.ROI.as_support_regions`)

            segment_index (int): index of the time segment

        """
        assert self.submit_side
        command_options = ['--working-directory="{}"'.format(self.wd)]
        if isinstance(stage_index, list):
            command_options.append(','.join(['--stage-index={}']+['{}']*(len(stage_index)-1)).format(*stage_index))
        elif stage_index is not None:
            command_options.append('--stage-index={:d}'.format(stage_index))
        if source is not None:
            command_options.append('--source="{}"'.format(source))
        if region_index is not None:
            command_options.append('--region-index={:d}'.format(region_index))
        if segment_index is not None:
            command_options.append('--segment-index={:d}'.format(segment_index))
        self.pending_jobs.append(tuple(command_options))
    @classmethod
    def _combine_analyses(cls, wd, data_location, logger, *args, directory_mapping={}):
        """
        Loads the generated rwa files, combines them and returns the list of the combined files.

        To be run on the worker side, where the collectible files are available.
        """
        analyses, original_files = {}, {}
        output_files = glob.glob(os.path.join(wd, '*.rwa'))
        loaded_files = []
        while output_files:
            output_file = output_files.pop()
            if os.stat(output_file).st_size == 0:
                logger.info('skipping empty file '+output_file)
                os.unlink(output_file)
                continue
            logger.info('reading file: {}'.format(output_file))
            try:
                __analyses = load_rwa(output_file,
                        lazy=True, force_load_spt_data=False)
            except:
                logger.critical(traceback.format_exc())
                #raise
                continue
            try:
                source = __analyses.metadata['datafile']
            except KeyError:
                logger.debug(str(__analyses))
                logger.debug('metadata: ',__analyses.metadata)
                logger.critical('key `datafile` not found in the metadata')
                #analyses = {}; break
                continue
            try:
                _analyses = analyses[source]
            except KeyError:
                analyses[source] = __analyses
                original_files[source] = [output_file]
            else:
                try:
                    append_leaf(_analyses, __analyses)
                except ValueError:
                    print(_analyses)
                    print(__analyses)
                    raise
                else:
                    original_files[source].append(output_file)
            loaded_files.append(output_file)
        end_result_files = []
        for source in analyses:
            logger.info('for source file: {}...'.format(source))
            # TODO: if isinstance(spt_data, (StandaloneRWAFile, RWAFiles))
            #       pass the input rwa file paths to _combine_analyses
            rwa_file = os.path.splitext(os.path.normpath(source))[0]+'.rwa'
            #logger.info((rwa_file, os.path.isabs(rwa_file), directory_mapping))
            if os.path.isabs(rwa_file):
                if directory_mapping:
                    for to_be_replaced in directory_mapping:
                        if rwa_file.startswith(to_be_replaced):
                            rwa_file = rwa_file[len(to_be_replaced):]
                            replacement = directory_mapping[to_be_replaced]
                            if replacement:
                                rwa_file = replacement+rwa_file # and NOT os.path.join, since rwa_file may start with '/'
            elif not os.path.isabs(os.path.expanduser(rwa_file)) and data_location:
                rwa_file = os.path.join(data_location, rwa_file)
            logger.info('writing file: {}'.format(rwa_file))
            if original_files[source][1:]:
                try:
                    save_rwa(os.path.expanduser(rwa_file), analyses[source], force=True, compress=False)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:# FileNotFoundError:
                    logger.warning('writing file failed: {}'.format(rwa_file))
                    local_rwa_file = rwa_file
                    fd, remote_rwa_file = tempfile.mkstemp(dir=wd, suffix='.rwa')
                    os.close(fd)
                    logger.info('writing file: {}...'.format(remote_rwa_file))
                    save_rwa(remote_rwa_file, analyses[source], force=True, compress=False)
                    rwa_file = (remote_rwa_file, local_rwa_file)
            else:
                analyses[source]._data.store.close() # close the file descriptor
                original_file = original_files[source][0]
                # copy original_file to rwa_file
                try:
                    with open(rwa_file, 'wb') as o:
                        with open(original_file, 'rb') as i:
                            o.write(i.read())
                except OSError:
                    remote_rwa_file = original_file
                    local_rwa_file = rwa_file
                    rwa_file = (remote_rwa_file, local_rwa_file)
            end_result_files.append(rwa_file)
        #
        to_keep = [ f[0] if isinstance(f,tuple) else f for f in end_result_files ]
        for output_file in loaded_files:
            if output_file not in to_keep:
                try:
                    os.unlink(output_file)
                except PermissionError as e:
                    if os.name == 'nt':
                        logger.debug(str(e))
                    else:
                        raise
        #
        return end_result_files
    def collect_results(self, stage_index=None):
        """
        Calls :meth:`_combine_analyses` for the remote data location and the
        current working directory and stage index,
        and retrieves the combined files from the worker side to the submit side,
        if they are different hosts.

        Returns :const:`True` if files are collected/retrieved.
        """
        return bool(self._combine_analyses(self.wd, None, self.logger, stage_index))
    def prepare_script(self, script=None):
        main_script = script is None
        if main_script:
            script = self.script
        # load
        if script.endswith('.ipynb'):
            content = self.import_ipynb(script)
        else:
            with open(self.script, 'r') as f:
                content = f.readlines()
        # modify
        filtered_content = self.filter_script_content(content)
        if script in self.temporary_files:
            tmpfile = script # reuse file
        else:
            tmpfile = self.make_temporary_file(suffix='.py', text=True)
        # flush
        with open(tmpfile, 'w') as f:
            for line in filtered_content:
                f.write(line)
        if main_script:
            self.script = tmpfile
        else:
            return tmpfile
    def filter_script_content(self, content):
        r"""
        Processes the script content for its dispatch onto the worker side.

        Arguments:

            content (*list* of *str*): lines with the :const:`'\n'` character at the end of each line

        Returns:

            *list* of *str*: modified lines

        """
        filtered_content = []
        for line in content:
            if line.startswith('#'):
                pass
            elif line.startswith('get_ipython('):
                continue
            elif '.run()' in line:
                # last line
                filtered_content.append(line)
                break
            filtered_content.append(line)
        return filtered_content
    def import_ipynb(self, notebook):
        """
        Extracts the code content of a IPython notebook.

        Arguments:

            notebook (str): path of the *.ipynb* notebook file

        Returns:

            *list* of *str*: extracted lines

        """
        cmd = 'jupyter nbconvert --to python "{}" --stdout'.format(notebook)
        self.logger.info('running: '+cmd)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if out:
            if not isinstance(out, str):
                out = out.decode('utf-8')
            content = out.splitlines(True)
        else:
            content = None
        if err:
            if not isinstance(err, str):
                err = err.decode('utf-8')
            if err != '[NbConvertApp] Converting notebook {} to python\n'.format(notebook):
                self.logger.error(err)
        return content
    def interrupt_jobs(self):
        """
        Interrupts the running jobs.
        """
        pass


class LocalHost(Env):
    """
    Runs jobs in local **python** processes.
    """
    __slots__ = ('running_jobs',)
    def __init__(self, **kwargs):
        Env.__init__(self, **kwargs)
        self.interpreter = sys.executable
        self.running_jobs = []
        self.wc = None
    @property
    def worker_count(self):
        return self._worker_count
    @worker_count.setter
    def worker_count(self, wc):
        if wc is None:
            wc = max(1, multiprocessing.cpu_count() - 1)
        elif wc < 0:
            wc = max(1, multiprocessing.cpu_count() + wc)
        self._worker_count = wc
    def submit_jobs(self):
        assert self.submit_side
        self.running_jobs = []
        for j,job in enumerate(self.pending_jobs):
            if len(self.running_jobs) == self.wc:
                self.wait_for_job_completion(1)
            self.logger.debug('submitting: '+( ' '.join(['{}']*(len(job)+2)).format(self.interpreter, self.script, *job) ))
            p = subprocess.Popen([self.interpreter, self.script, *job],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.running_jobs.append((j,p))
        self.pending_jobs = []
    def wait_for_job_completion(self, count=None):
        assert self.submit_side
        n = 0
        for j,p in self.running_jobs:
            out, err = p.communicate()
            out, err = out.rstrip(), err.rstrip()
            if out:
                if not isinstance(out, str):
                    out = out.decode('utf-8')
                self.logger.info(out)
            if err:
                if not isinstance(err, str):
                    err = err.decode('utf-8')
                self.logger.error(err)
            self.logger.debug('job {:d} done\n'.format(j))
            n += 1
            if n==count:
                self.running_jobs = self.running_jobs[n:]
                return
        self.running_jobs = []
    def interrupt_jobs(self):
        for j,p in self.running_jobs:
            p.terminate()
        for j,p in self.running_jobs:
            out, err = p.communicate()
            if err:
                self.logger.error(err)
                return False
            elif out:
                self.logger.info(out)
        self.running_jobs = []
        return True

Environment.register(LocalHost)


class UpdatedDict(dict):
    __slots__ = ('_proper', '_inherited')
    def __init__(self, proper, shared, shared_keys=None):
        self._proper = proper
        if shared_keys:
            self._inherited = { k: shared[k] for k in shared if k in shared_keys }
        else:
            self._inherited = dict(shared)
        self._inherited.update(self._proper)
    def __repr__(self):
        return repr(self._inherited)
    def __str__(self):
        return str(self._inherited)
    def __contains__(self, key):
        return key in self._inherited
    def __iter__(self):
        return iter(self._inherited)
    def __len__(self):
        return len(self._inherited)
    def __getitem__(self, key):
        return self._inherited[key]
    def __setitem__(self, key, value):
        self._proper[key] = value
        self._inherited[key] = value
    def __delitem__(self, key):
        del self._inherited[key]
        try:
            del self._proper[key]
        except KeyError:
            pass
    def clear(self):
        self._proper.clear()
        self._inherited = {}
    def copy(self):
        return self._inherited.copy()
    def get(self, *args):
        return self._inherited.get(*args)
    def items(self):
        return self._inherited.items()
    def keys(self):
        return self._inherited.keys()
    def pop(self, key, *args):
        try:
            value = self._inherited.pop(key)
        except KeyError:
            if args:
                value = args[0]
            else:
                raise
        else:
            try:
                del self._proper[key]
            except KeyError:
                pass
        return value
    def popitem(self):
        key, value = self._inherited.popitem()
        try:
            del self._proper[key]
        except KeyError:
            pass
        return key, value
    def setdefault(self, key, default=None):
        try:
            value = self._inherited[key]
        except KeyError:
            value = self._proper[key] = self._inherited[key] = default
        return value
    def update(self, other):
        self._proper.update(other)
        self._inherited.update(other)
    def values(self):
        return self._inherited.values()


class Slurm(Env):
    """
    Not supposed to properly run, as TRamWAy is expected to be called
    inside a container;
    see :class:`SlurmOverSSH` instead.
    """
    __slots__ = ('_sbatch_options','_job_id','refresh_interval','_srun_options')
    def __init__(self, **kwargs):
        Env.__init__(self, **kwargs)
        self._sbatch_options = dict(
                output='%J.out',
                error='%J.err',
                )
        self._srun_options = {}
        self._job_id = None
        self.refresh_interval = 20
    @property
    def sbatch_options(self):
        return self._sbatch_options
    @property
    def srun_options(self):
        return UpdatedDict(self._srun_options, self.sbatch_options, self.slurm_common_options)
    @srun_options.setter
    def srun_options(self, options):
        if options is None:
            self._srun_options = {}
        elif isinstance(options, SlurmOptions):
            self._srun_options = options._proper
        elif isinstance(options, dict):
            self._srun_options = options
        else:
            raise TypeError("wrong type for srun_options: '{}'".format(type(options).__name__))
    @property
    def slurm_common_options(self):
        """
        tuple:
            set of options that are common to sbatch and srun;
            if a "common" option found in :attr:`sbatch_options`
            and not in :attr:`srun_options`, :attr:`srun_options`
            is updated with this option
        """
        return ('p', 'partition', 'q', 'qos')
    @property
    def job_name(self):
        try:
            name = self.sbatch_options['job-name']
        except KeyError:
            try:
                name = self.sbatch_options['J']
            except KeyError:
                name = self.sbatch_options['job-name'] = os.path.splitext(os.path.basename(self.script))[0]
            else:
                self.sbatch_options['job-name'] = name
        return name
    @job_name.setter
    def job_name(self, name):
        self.sbatch_options['job-name'] = name
    @property
    def job_id(self):
        return self._job_id
    @job_id.setter
    def job_id(self, i):
        self._job_id = i
    @property
    def task_count(self):
        return self.worker_count
    @task_count.setter
    def task_count(self, count):
        self.worker_count = count
    def setup(self, *argv):
        Env.setup(self, *argv)
        if self.submit_side:
            self.job_name # sets job-name sbatch option
            output_log = self.sbatch_options['output']
            if not os.path.isabs(os.path.expanduser(output_log)):
                output_log = '/'.join((self.wd, output_log)) # NOT os.path.join
                self.sbatch_options['output'] = output_log
            error_log = self.sbatch_options['error']
            if not os.path.isabs(os.path.expanduser(error_log)):
                error_log = '/'.join((self.wd, error_log))
                self.sbatch_options['error'] = error_log
    def make_sbatch_script(self, stage=None, path=None):
        assert self.submit_side
        if path is None:
            sbatch_script = self.make_temporary_file(suffix='.sh' if stage is None else '-stage{:d}.sh'.format(stage), text=True)
        else:
            sbatch_script = path
        self.job_name # set default job name if not defined yet
        with open(sbatch_script, 'w', encoding='utf-8', newline='\n') as f:
            f.write('#!/bin/bash\n')
            for option, value in self.sbatch_options.items():
                if option[1:]:
                    line = '#SBATCH --{}={}\n'
                else:
                    line = '#SBATCH -{} {}\n'
                if isinstance(value, str) and ' ' in value:
                    value = '"{}"'.format(value)
                f.write(line.format(option, value))
            f.write('#SBATCH --array=0-{:d}{}\n'.format(
                len(self.pending_jobs)-1,
                '' if self.wc is None else '%{:d}'.format(self.wc)))
            f.write('\ndeclare -a tasks\n')
            wd, si = None, None
            for j, job in enumerate(self.pending_jobs):
                assert job[0].startswith('--working-directory=')
                if wd is None:
                    assert j == 0
                    wd = job[0]
                else:
                    assert wd == job[0]
                if job[1].startswith('--stage-index='):
                    if si is None:
                        assert j == 0
                        si = job[1]
                    else:
                        assert si == job[1]
                    job = job[2:]
                else:
                    assert si is None
                    job = job[1:]
                f.write('tasks[{:d}]="{}"\n'.format(j,
                    ' '.join(['{}']*len(job)).format(*job).replace('"',r'\"')))
            if si is None:
                common_task_args = wd
            else:
                common_task_args = '{} {}'.format(wd.replace('"',r'\"'), si)
            #f.write('\npushd "{}" > /dev/null\n'.format(self.wd))
            f.write('\n{} {} {} ${{tasks[SLURM_ARRAY_TASK_ID]}}\n'.format(self.interpreter, self.script,
                common_task_args))
            #f.write('\npopd > /dev/null\n')
            f.write('\nunset tasks\n')
        if True:
            with open(sbatch_script, 'r') as f:
                self.logger.debug(f.read())
        return sbatch_script
    def submit_jobs(self):
        # this code below does not actually run as the :meth:`submit_jobs` method is overwritten
        # in :class:`SlurmOverSSH`
        sbatch = 'sbatch'
        sbatch_script = self.make_sbatch_script()
        self.logger.info('running: {} {}'.format(sbatch, sbatch_script))
        p = subprocess.Popen([sbatch, sbatch_script],
                stderr=subprocess.STDOUT, encoding='utf-8')#, stdout=subprocess.PIPE)
        out, err = p.communicate()
        if out:
            self.logger.info(out)
            if out.startswith('Submitted batch job '):
                self.job_id = out.strip().split(' ')[-1]
        if err:
            self.logger.error(err)
        self.pending_jobs = []
    def wait_for_job_completion(self):
        prefix = '{}_['.format(self.job_id)
        try:
            while True:
                time.sleep(self.refresh_interval)
                p = subprocess.Popen(('squeue', '-j '+self.job_id, '-h', '-o "%.18i %.2t %.10M %R"'),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                out, err = p.communicate()
                if err:
                    self.logger.error(err)
                elif out:
                    for row in out.splitlines():
                        if row.startswith(prefix):
                            break
                    out = row.split(' ')
                    status, time_used, reason = out[1], out[2], ' '.join(out[3:])
                    self.logger.info('status: {}   time used: {}   reason: {}'.format(status, time_used, reason))
                else:
                    break
        except:
            self.logger.info('killing jobs with: scancel '+self.job_id)
            subprocess.Popen(('scancel', self.job_id)).communicate()
            raise


__remote_host_attrs__ = ('_ssh','_local_data_location','_remote_data_location','_directory_mapping','_remote_dependencies')

class RemoteHost(object):
    """
    children classes should define attributes listed in `__remote_host_attrs__`.

    minimum base implementation:

    .. code-block:: python

        assert issubclass(cls, Env)

        class Cls(cls, RemoteHost):
            __slots__ = __remote_host_attrs__
            def __init__(self, **kwargs):
                cls.__init__(self, **kwargs)
                RemoteHost.__init__(self)
            wd_is_available = RemoteHost.wd_is_available
            make_working_directory = RemoteHost.make_working_directory
            def setup(self, *argv):
                cls.setup(self, *argv)
                RemoteHost.setup(self, *argv)
            def filter_script_content(self, content):
                return RemoteHost.filter_script_content(self,
                        cls.filter_script_content(self, content))
            def make_job(self, stage_index=None, source=None, region_index=None,
                        segment_index=None):
                cls.make_job(self, stage_index, self.format_source(source),
                        region_index, segment_index)
            def collect_results(self, stage_index=None):
                log_pattern = '*.out' # to be adapted
                RemoteHost.collect_results(self, log_pattern, stage_index):
            def delete_temporary_data(self):
                RemoteHost.delete_temporary_data(self)
                cls.delete_temporary_data(self)

    """
    __slots__ = ()
    def __init__(self):
        self._ssh = None
        self._local_data_location = None
        self._remote_data_location = None
        self._directory_mapping = {}
        self._remote_dependencies = None
    @property
    def ssh(self):
        """
        SSH communication interface.

        See also :class:`~tramway.analyzer.env.ssh.Client`.
        """
        if self._ssh is None:
            from tramway.analyzer.env import ssh
            self._ssh = ssh.Client()
        return self._ssh
    @property
    def remote_data_location(self):
        """
        Data location on the remote host (worker side).

        If defined, the current directory will be changed to this location
        on the remote host.
        """
        return self._remote_data_location
    @remote_data_location.setter
    def remote_data_location(self, path):
        former_location = self.remote_data_location
        if path != former_location:
            if self.local_data_location is not None:
                if path is None: # former_location is not None
                    # therefore key `local_data_location` can be found in `directory_mapping`
                    del self.directory_mapping[self.local_data_location]
                else:
                    self.directory_mapping[self.local_data_location] = path
            self._remote_data_location = path
    @property
    def local_data_location(self):
        """
        Data location on the local host (submit side).
        """
        return self._local_data_location
    @local_data_location.setter
    def local_data_location(self, path):
        former_location = self.local_data_location
        if path != former_location:
            if self.remote_data_location is not None:
                if path is None: # former_location is not None
                    del self.directory_mapping[former_location]
                else:
                    self.directory_mapping[path] = self.remote_data_location
            self._local_data_location = path
    @property
    def directory_mapping(self):
        """
        Directory mapping with local paths as keys
        and the corresponding remote paths as values.

        Paths should not have slashes at the end.
        """
        return self._directory_mapping
    @property
    def remote_dependencies(self):
        """
        Command to be run before batch submission.
        """
        return self._remote_dependencies
    @remote_dependencies.setter
    def remote_dependencies(self, deps):
        self._remote_dependencies = deps
    @property
    def collection_interpreter(self):
        return self.interpreter
    @property
    def wd_is_available(self):
        return self.worker_side
    def make_working_directory(self):
        if self.submit_side:
            cmd = 'mkdir -p "{}"; mktemp -d -p "{}"'.format(self.working_directory, self.working_directory)
            out, err = self.ssh.exec(cmd)
            if err:
                # attribute `logger` must be defined by the concrete parent class
                self.logger.error(err)
            elif out:
                self.working_directory = out.rstrip()
        else:
            assert os.path.isdir(self.wd)
    def setup(self, *argv):
        if self.worker_side:
            self.script = '/'.join((self.wd, os.path.basename(self.script)))
    @classmethod
    def _format_collectibles(cls, paths):
        home = os.path.expanduser('~')
        collectibles = []
        for path in paths:
            mapping = isinstance(path, tuple)
            if mapping:
                src_path, path = path
            path = os.path.normpath(path)
            if os.path.isabs(path) and path.startswith(home):
                path = '~'+path[len(home):]
            if mapping:
                path = '{{{0}=>{1}}}'.format(src_path, path)
            collectibles.append(path)
        return collectibles
    def format_collectibles(self, paths=None):
        """
        Calls `_format_collectibles` with the list of registered collectibles.
        """
        if paths is None:
            paths = self.collectibles
        return self._format_collectibles(paths)
    def format_source(self, source):
        if source is not None:
            source = os.path.normpath(source)
            home = os.path.expanduser('~')
            if os.path.isabs(source) and source.startswith(home):
                source = '~'+source[len(home):]
            for _dir in self.directory_mapping:
                if source.startswith(_dir):
                    source = self.directory_mapping[_dir]+source[len(_dir):]
                    break
            source = source.replace(os.sep, '/')
        return source
    def dispatch(self, **kwargs):
        if not kwargs:
            self.prepare_script()
            src = self.script
            dest = os.path.basename(self.script)
            #if dest.endswith('.ipynb'):
            #    dest = dest[:-5]+'py'
            dest = '/'.join((self.wd, dest))
            attrs = self.ssh.put(src, dest)
            self.logger.debug(attrs)
            self.script = dest
            self.logger.info('Python script location: '+dest)
            return True
    @classmethod
    def _collectible_prefix(cls, stage_index=None):
        """
        Computes the starting substring of the log message that reports
        the list of collectibles.
        """
        if stage_index is None:
            prefix = 'OUTPUT_FILES='
        elif isinstance(stage_index, (tuple, list)):
            prefix = 'OUTPUT_FILES[{}]='.format('-'.join([ str(i) for i in stage_index ]))
        else:
            prefix = 'OUTPUT_FILES[{}]='.format(stage_index)
        return prefix
    def collectible_prefix(self, stage_index=None):
        """
        Calls :meth:`_collectible_prefix` for the current stage(s).
        """
        if stage_index is None:
            # attribute `current_stage` is expected to defined in the concrete parent class
            stage_index = self.current_stage
        return self._collectible_prefix(stage_index)
    def collect_results(self, _log_pattern, stage_index=None, _parent_cls='Env'):
        """
        Downloads the reported collectibles from the remote host (worker side)
        to the local host (submit side).
        """
        _prefix = self.collectible_prefix(stage_index)
        data_loc = self.remote_data_location if self.remote_data_location else ''
        home = os.path.expanduser('~')
        if home not in self.directory_mapping:
            self.directory_mapping[home] = '~'
        code = """\
from tramway.analyzer import environments, BasicLogger

wd = '{}'
data_location = '{}'
logger = BasicLogger()
stage = {!r}
log_pattern = '{}'
directory_mapping = {}

files  = environments.{}._combine_analyses(wd, data_location, logger, stage,
            directory_mapping=directory_mapping)
files += environments.RemoteHost._collectibles_from_log_files(wd, log_pattern, stage)

files  = environments.RemoteHost._format_collectibles(files)

print('{}'+';'.join(files))\
""".format(self.wd, data_loc, stage_index, _log_pattern, self.directory_mapping, _parent_cls, _prefix)
        local_script = self.make_temporary_file(suffix='.py', text=True)
        with open(local_script, 'w') as f:
            f.write(code)
        remote_script = '/'.join((self.wd, os.path.basename(local_script)))
        attrs = self.ssh.put(local_script, remote_script, confirm=True)
        self.logger.debug(attrs)
        cmd = '{}{} {}; rm {}'.format(
                '' if self.remote_dependencies is None else self.remote_dependencies+'; ',
                self.collection_interpreter, remote_script, remote_script)
        out, err = self.ssh.exec(cmd, shell=True, logger=self.logger)
        out, err = out.rstrip(), err.rstrip()
        if err:
            self.logger.error(err)
        if not out:
            return False
        self.logger.debug(out)
        out = out.splitlines()
        while True:
            try:
                line = out.pop() # starting from last line
            except IndexError: # empty list
                raise RuntimeError('missing output: {}...'.format(_prefix)) from None
            if line.startswith(_prefix):
                end_result_files = line[len(_prefix):].split(';')
                break
        any_transfer = False
        for end_result_file in end_result_files:
            if not end_result_file:
                continue
            if end_result_file[0]=='{' and '=>' in end_result_file and end_result_file[-1]=='}':
                src, dest = end_result_file[1:-1].split('=>',1)
            else:
                src = dest = end_result_file
            dest = os.path.normpath(dest)
            self.logger.info('retrieving file: '+dest)
            for local, remote in self.directory_mapping.items():
                if dest.startswith(remote):
                    dest = local+dest[len(remote):]
                    break
            try:
                self.ssh.get(src, dest)
            except FileNotFoundError: # the target file might be empty
                self.logger.warning('failed')
            else:
                any_transfer = True
        return any_transfer
    @classmethod
    def _collectibles_from_log_files(cls, wd, log_pattern, stage_index=None):
        """
        Reads the collectible names reported in the log files.
        """
        collectibles = []
        _prefix = cls._collectible_prefix(stage_index)
        home = os.path.expanduser('~')
        log_files = glob.glob(os.path.join(wd,log_pattern))
        for log_file in log_files:
            with open(log_file, 'r') as f:
                try:
                    last_line = f.readlines()[-1]
                except IndexError:
                    continue
            if last_line.startswith(_prefix):
                last_line = last_line.rstrip()
                for collectible in last_line[len(_prefix):].split(';'):
                    if collectible:
                        collectible = os.path.normpath(collectible)
                        #if os.path.isabs(collectible):
                        #    if collectible.startswith(home):
                        #        collectible = '~'+collectible[len(home):]
                        collectibles.append(collectible)
        return collectibles
    def filter_script_content(self, content):
        filtered_content = []
        if self.remote_data_location:
            filtered_content.append('import os\n')
            filtered_content.append("os.chdir(os.path.expanduser('{}'))\n".format(self.remote_data_location))
            filtered_content.append('\n')
        for line in content:
            pattern = '.spt_data.from_'
            if self.remote_data_location and pattern in line:
                analyzer, suffix = line.split(pattern)
                parts = suffix.split('(')[::-1]
                initializer = parts.pop()
                arg = parts.pop()
                quote = arg[0]
                if quote in "'\"":
                    arg, tail = arg[1:].split(quote, 1)
                    if tail and tail[0] == ')':
                        self.local_data_location, arg = os.path.split(arg)
                        line = '{}{}{}({}{}/{}{})\n'.format(analyzer, pattern, initializer,
                                quote,
                                self.remote_data_location, arg,
                                quote)
            filtered_content.append(line)
        # last line is "[analyzer].run()\n"
        call = line.find('.run()')
        if 0<call:
            analyzer_var = line[:call]
            # insert mapping for home directories
            filtered_content = filtered_content[:-1]
            filtered_content.append("""
if {1}.env.directory_mapping is None:
    {1}.env.directory_mapping = {{}}
if '{0}' not in {1}.env.directory_mapping:
    {1}.env.directory_mapping['{0}'] = '~'
""".format(os.path.expanduser('~').replace('\\',r'\\'), analyzer_var))
            filtered_content.append(line)
            # append output listing
            filtered_content.append(\
                    "\nprint({0}.env.collectible_prefix()+';'.join({0}.env.format_collectibles()))\n".format(
                        analyzer_var))
        # 
        return filtered_content
    def delete_temporary_data(self):
        """
        Deletes the worker-side working directory.
        """
        out, err = self.ssh.exec('rm -rf '+self.wd)
        if err:
            self.logger.error(err)
        if out:
            self.logger.info(out)


class SlurmOverSSH(Slurm, RemoteHost):
    """
    Calls *sbatch* through an SSH connection to a Slurm server.

    .. note::

        All filepaths should be absolute or relative to the $HOME directory.
        This applies especially to all file-based initializers.

    """
    __slots__ = __remote_host_attrs__
    def __init__(self, **kwargs):
        Slurm.__init__(self, **kwargs)
        RemoteHost.__init__(self)
    wd_is_available = RemoteHost.wd_is_available
    make_working_directory = RemoteHost.make_working_directory
    def setup(self, *argv):
        Slurm.setup(self, *argv)
        RemoteHost.setup(self, *argv)
    def dispatch(self, **kwargs):
        if 'stage_options' in kwargs:
            stage_options = kwargs['stage_options']
            for option in stage_options:
                if option == 'sbatch_options':
                    self.sbatch_options.update(stage_options[option])
                else:
                    self.logger.debug('ignoring option: '+option)
        return RemoteHost.dispatch(self, **kwargs)
    def filter_script_content(self, content):
        return RemoteHost.filter_script_content(self,
                Slurm.filter_script_content(self, content))
    def make_job(self, stage_index=None, source=None, region_index=None, segment_index=None):
        Slurm.make_job(self, stage_index, self.format_source(source),
                region_index, segment_index)
    def submit_jobs(self):
        sbatch_script = self.make_sbatch_script()
        dest = '/'.join((self.wd, os.path.basename(sbatch_script)))
        attrs = self.ssh.put(sbatch_script, dest)
        self.logger.debug(attrs)
        self.logger.info('sbatch script transferred to: '+dest)
        #self.logger.info('running: module load singularity')
        #out, err = self.ssh.exec('module load singularity')
        if self.remote_dependencies:
            cmd = self.remote_dependencies+'; sbatch '+dest
        else:
            cmd = 'sbatch '+dest
        self.logger.info('running: '+cmd)
        out, err = self.ssh.exec(cmd, shell=True)
        if out:
            out = out.rstrip()
            self.logger.debug(out)
            if out.startswith('Submitted batch job '):
                self.job_id = out.split(' ')[-1]
        if err:
            self.logger.error(err.rstrip())
        self.pending_jobs = []
    def wait_for_job_completion(self):
        self.logger.info('''\
notice: job failures are not reported before the stage is complete;
        check the .err log files generated in the remote working directory
        and manually interrupt the pipeline if all the jobs seem to fail\
''')
        try:
            cmd = 'squeue -j {} -h -o "%K %t %M %R"'.format(self.job_id)
            while True:
                time.sleep(self.refresh_interval)
                out, err = self.ssh.exec(cmd, shell=True)
                if err:
                    err = err.rstrip()
                    if err == 'slurm_load_jobs error: Invalid job id specified':
                        # complete
                        break
                    self.logger.error(err.rstrip())
                elif out:
                    # parse and print progress info
                    start, _out = None, []
                    _continue = False
                    for line in out.splitlines():
                        parts = line.split()
                        if '-' in parts[0]:
                            if start is None:
                                try:
                                    start, stop = parts[0].split('-')
                                except ValueError:
                                    self.logger.debug('squeue output parsing failed: \n'+out)
                                    _continue = True; break
                                else:
                                    stop = stop.split('%')[0]
                                    start, stop = int(start), int(stop)
                            else:
                                self.logger.debug('squeue output parsing failed: \n'+out)
                                _continue = True; break
                        elif '%' in parts[0]:
                            start = stop = int(parts[0].split('%')[0])
                        else:
                           _out.append(parts)
                    if _continue:
                        continue
                    if start is None:
                        total, pending = None, 0
                    else:
                        total = stop
                        pending = stop - start
                    running = 0
                    other = 0
                    for out in _out:
                        try:
                            array_ix, status, time_used = int(out[0]), out[1], out[2]
                        except ValueError:
                            self.logger.debug('squeue output parsing failed on line: '+' '.join(out))
                            continue
                        reason = ' '.join(out[3:])
                        if status == 'R':
                            running += 1
                        else:
                            other += 1
                        #self.logger.debug(task: {:d}   status: {}   time used: {}   reason: {}'.format(array_ix, status, time_used, reason))
                    self.logger.info('tasks:\t{} done,\t{} running,\t{} pending{}'.format(
                        '(unknown)' if total is None else total-pending-running-other,
                        '(unknown)' if running is None else running,
                        '(unknown)' if pending is None else pending,
                        ',\t{} in abnormal state'.format(other) if other else ''))
                else:
                    # complete
                    break
        except:
            self.logger.info('killing jobs with: scancel '+self.job_id)
            self.ssh.exec('scancel '+self.job_id, shell=True)
            raise
    @property
    def collection_interpreter(self):
        if self._srun_options:
            cmd = ['srun']
            for option, value in self.srun_options.items():
                if option[1:]:
                    fmt = '--{}={}'
                else:
                    fmt = '-{} {}'
                if isinstance(value, str) and ' ' in value:
                    value = '"{}"'.format(value)
                cmd.append(fmt.format(option, value))
            cmd.append(self.interpreter)
            return ' '.join(cmd)
        else:
            return self.interpreter
    def collect_results(self, stage_index=None):
        RemoteHost.collect_results(self, '*.out', stage_index)
    collect_results.__doc__ = RemoteHost.collect_results.__doc__
    def delete_temporary_data(self):
        RemoteHost.delete_temporary_data(self)
        Slurm.delete_temporary_data(self)
    def resume(self, log=None, wd=None, stage_index=None, job_id=None):
        """
        Parses log output of the disconnected instance, looks for the current stage index,
        and tries to collect the resulting files.

        This completes the current stage only. Further stages are not run.
        """
        if wd is None or stage_index is None or job_id is None:
            if log is None:
                log = input('please copy-paste below the log output of the disconnected instance\n(job progress information can be omitted):\n')
            log = log.splitlines()
            #job_id = wd = stage_index = None
            for line in log[::-1]:
                if wd:
                    try:
                        opt = line.index(' --stage-index=')
                    except ValueError:
                        pass
                    else:
                        stage_index = line[opt+15:].split()[0]
                        stage_index = [ int(s) for s in stage_index.split(',') ]
                        break
                elif job_id:
                    assert line.startswith('running: ') and 'sbatch ' in line
                    script = line.split('sbatch ')[-1].rstrip()
                    wd = '/'.join(script.split('/')[:-1])
                elif line.startswith('Submitted batch job '):
                    job_id = line[20:].rstrip()
        if stage_index:
            self.setup(sys.executable)
            assert self.submit_side
            self.delete_temporary_data() # undo wd creation during setup
            self.working_directory = wd
            self.job_id = job_id
            self.logger.info('trying to complete stage(s): '+', '.join([str(i) for i in stage_index]))
            self.wait_for_job_completion()
            self.collect_results(stage_index=stage_index)
        else:
            self.logger.info('cannot identify an execution point where to resume from')


Environment.register(SlurmOverSSH)


class SingularitySlurm(SlurmOverSSH):
    """
    Runs TRamWAy jobs as Slurm jobs in a Singularity container.

    The current default Singularity container is *tramway-hpc-210114.sif*.
    See also `available_images.rst <https://github.com/DecBayComp/TRamWAy/blob/master/containers/available_images.rst>`_.

    Children classes should define the :meth:`hostname` and :meth:`scratch` methods.
    They can be defined as standard methods or class methods.
    """
    @classmethod
    def hostname(cls):
        raise NotImplementedError
    @classmethod
    def scratch(cls, username):
        raise NotImplementedError
    def __init__(self, **kwargs):
        SlurmOverSSH.__init__(self, **kwargs)
        if os.path.isdir('/pasteur'):
            self.interpreter = 'singularity exec -H $HOME -B /pasteur tramway-hpc-210114.sif python3.6 -s'
        else:
            self.interpreter = 'singularity exec -H $HOME tramway-hpc-210114.sif python3.6 -s'
        self.ssh.host = self.hostname()
    @property
    def username(self):
        try:
            username, _ = self.ssh.host.split('@')
        except (AttributeError, ValueError):
            import getpass
            username = getpass.getuser()
        return username
    @username.setter
    def username(self, name):
        if name is None:
            self.ssh.host = self.hostname()
            self.wd = None
        else:
            self.ssh.host = '@'.join((name, self.hostname()))
            if self.wd is None:
                self.wd = self.scratch(name)
    @property
    def container(self):
        parts = self.interpreter.split()
        return parts[parts.index('python3.6')-1]
    @container.setter
    def container(self, path):
        parts = self.interpreter.split()
        p = parts.index('python3.6')
        self.interpreter = ' '.join(parts[:p-1]+[path]+parts[p:])
    def get_container_url(self, container=None):
        if container is None:
            container = self.container
        return {
                'tramway-hpc-200928.sif':   'http://dl.pasteur.fr/fop/VsJygkxP/tramway-hpc-200928.sif',
                'tramway-hpc-210112.sif':   'http://dl.pasteur.fr/fop/tVZe8prV/tramway-hpc-210112.sif',
                'tramway-hpc-210114.sif':   'http://dl.pasteur.fr/fop/cZWZqsDW/tramway-hpc-210114.sif',
                }.get(container, None)
    def setup(self, *argv):
        SlurmOverSSH.setup(self, *argv)
        if self.submit_side:
            self.ssh.download_if_missing(self.container, self.get_container_url(), self.logger)
    @property
    def working_directory(self):
        if self._working_directory is None:
            self._working_directory = self.scratch(self.username)
        return self._working_directory
    working_directory.__doc__ = SlurmOverSSH.working_directory.__doc__
    @working_directory.setter
    def working_directory(self, wd):
        self._working_directory = wd


class Tars(SingularitySlurm):
    """
    Designed for server *tars.pasteur.fr*.

    The server is closed.
    """
    @classmethod
    def hostname(cls):
        return 'tars.pasteur.fr'
    @classmethod
    def scratch(cls, username):
        return os.path.join('/pasteur/scratch/users', username)
    def __init__(self, **kwargs):
        SingularitySlurm.__init__(self, **kwargs)
        self.remote_dependencies = 'module load singularity'


class GPULab(SingularitySlurm):
    """
    Designed for server *adm.inception.hubbioit.pasteur.fr*.
    """
    @classmethod
    def hostname(cls):
        return 'adm.inception.hubbioit.pasteur.fr'
    @classmethod
    def scratch(cls, username):
        return os.path.join('/master/home', username, 'scratch')


class Maestro(SingularitySlurm):
    """
    Designed for server *maestro.pasteur.fr*.
    """
    @classmethod
    def hostname(cls):
        return 'maestro.pasteur.fr'
    @classmethod
    def scratch(cls, username):
        return os.path.join('/pasteur/sonic/scratch/users', username)
    def __init__(self, **kwargs):
        SingularitySlurm.__init__(self, **kwargs)
        self.remote_dependencies = 'module load singularity'


__all__ = ['Environment', 'LocalHost', 'SlurmOverSSH', 'Tars', 'GPULab', 'Maestro']

