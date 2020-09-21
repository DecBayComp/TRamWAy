
from ..attribute import *
from .abc import *
from ..spt_data.abc import SPTData
from ..roi.abc import ROI
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


class Proxy(object):
    __slots__ = ('_proxied',)
    def __init__(self, proxied):
        self._proxied = proxied
    def __len__(self):
        return self._proxied.__len__()
    def __iter__(self):
        return self._proxied.__iter__()
    @property
    def _parent(self):
        return self._proxied._parent
    @_parent.setter
    def _parent(self, par):
        self._proxied._parent = par
    def __getattr__(self, attrname):
        return getattr(self._proxied, attrname)
    def __setattr__(self, attrname, val):
        if attrname == '_proxied':
            object.__setattr__(self, '_proxied', val)
        else:
            setattr(self._proxied, attrname, val)


class Env(AnalyzerNode):
    __slots__ = ('_interpreter','_script','_working_directory','_worker_count',
            '_pending_jobs','_selectors', '_selector_classes', '_temporary_files',
            'debug')
    def __init__(self, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._interpreter = 'python3'
        self._script = None
        self._selectors = None
        self._selector_classes = {}
        self._temporary_files = []
        self._working_directory = None
        self._worker_count = None
        self.pending_jobs = []
        self.debug = False
    @property
    def logger(self):
        return self._parent.logger
    @property
    def working_directory(self):
        return self._working_directory
    @working_directory.setter
    def working_directory(self, wd):
        self._working_directory = wd
    @property
    def wd(self):
        return self.working_directory
    @wd.setter
    def wd(self, wd):
        self.working_directory = wd
    @property
    def wd_is_available(self):
        return True
    @property
    def interpreter(self):
        return self._interpreter
    @interpreter.setter
    def interpreter(self, cmd):
        self._interpreter = cmd
    @property
    def script(self):
        return self._script
    @script.setter
    def script(self, file):
        if file is None:
            self._script = None
        else:
            self._script = os.path.abspath(os.path.expanduser(file))
    @property
    def worker_count(self):
        return self._worker_count
    @worker_count.setter
    def worker_count(self, wc):
        self._worker_count = wc
    @property
    def wc(self):
        return self.worker_count
    @wc.setter
    def wc(self, wc):
        self.worker_count = wc
    @property
    def pending_jobs(self):
        return self._pending_jobs
    @pending_jobs.setter
    def pending_jobs(self, jobs):
        self._pending_jobs = jobs
    @property
    def selectors(self):
        return self._selectors
    @selectors.setter
    def selectors(self, sel):
        self._selectors = sel
    @property
    def temporary_files(self):
        return self._temporary_files
    @property
    def analyzer(self):
        return self._parent
    def setup(self, *argv):
        assert argv
        #if not argv:
        #    return
        valid_keys = set(('stage-index', 'source', 'region-index', 'segment-index', 'cell-index',
            'working-directory'))
        valid_arguments = {}
        for arg in argv[1:]:
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
            self.wd = valid_arguments.pop('working_directory', self.wd)
            self.selectors = valid_arguments
            #self.logger.debug(self.selectors)
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
                self.logger.debug('selecting source: '+', '.join((sources,) if isinstance(sources, str) else sources))
            #
            for f in self.spt_data_selector(self.analyzer.spt_data):
                f.analyses.rwa_file = self.make_temporary_file(suffix='.rwa', output=True)
                f.analyses.autosave = True
        elif self.script is None:
            candidate_files = [ f for f in os.listdir() \
                    if f.endswith('.py') or f.endswith('.ipynb') ]
            if candidate_files and not candidate_files[1:]:
                self.script = candidate_files[0]
                os.logger.info('candidate script: {} (in {})'.format(self.script, os.getcwd()))
            raise ValueError('attribute `script` is not set')
        else:
            self.make_working_directory()
            self.logger.info('working directory: '+self.wd)
    def spt_data_selector(self, spt_data_attr):
        if isinstance(spt_data_attr, Initializer) or len(spt_data_attr)==1:
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
            class selector_cls(Proxy):
                __slots__ = ()
                def __iter__(self):
                    for f in cls.__iter__(self):
                        if f.source in sources:
                            #logger.debug('source {} selected'.format(f.source))
                            yield f
            SPTData.register(selector_cls)
            self._selector_classes[cls] = selector_cls
        return selector_cls(spt_data_attr)
    def roi_selector(self, roi_attr):
        if isinstance(roi_attr, Initializer):
            return roi_attr
        cls = type(roi_attr)
        try:
            selector_cls = self._selector_classes[cls]
        except KeyError:
            try:
                source = self.selectors['source']
            except KeyError:
                source = None
                def _source(source_arg):
                    return source_arg
            else:
                sources = set([source]) if isinstance(source, str) else set(source)
                def _source(source_arg):
                    if source_arg is None:
                        return source
                    if callable(source_arg):
                        return lambda src: src in sources and source_arg(src)
                    elif source_arg in sources:
                        return source_arg # or lambda src: True
                    else:
                        return lambda src: False
            try:
                region = self.selectors['region_index']
            except KeyError:
                if source is None:
                    return roi_attr
                def _region(index_arg):
                    return index_arg
            else:
                regions = set([region]) if isinstance(region, int) else set(regions)
                def _region(index_arg):
                    if index_arg is None:
                        return region
                    elif callable(index_arg):
                        return lambda r: r in regions and index_arg(r)
                    elif index_arg in regions:
                        return index_arg # or lambda r: True
                    else:
                        return lambda r: False
            logger = self.logger
            class selector_cls(Proxy):
                __slots__ = ()
                def as_support_regions(self, index=None, source=None, return_index=False):
                    yield from cls.as_support_regions(self, _region(index), _source(source), return_index)
                    #for i,r in cls.as_support_regions(self, _region(index), _source(source), True):
                    #    logger.debug('region {} selected'.format(i if r.label is None else i))
                    #    if return_index:
                    #        yield i,r
                    #    else:
                    #        yield r
                def as_individual_roi(self, *args, **kwargs):
                    raise NotImplementedError
            ROI.register(selector_cls)
            self._selector_classes[cls] = selector_cls
        return selector_cls(roi_attr)
    @property
    def submit_side(self):
        return self.selectors is None
    @property
    def worker_side(self):
        return self.selectors is not None
    def make_working_directory(self):
        assert self.submit_side
        self.wd = self.make_temporary_file(directory=True)
    def make_temporary_file(self, output=False, directory=False, **kwargs):
        """
        Arguments:

            output (bool): do not delete the file once the task is done.

            directory (bool): make a temporary directory.

        More keyword arguments can be passed to :fun:`tempfile.mkstemp`.
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
        if not kwargs:
            self.prepare_script()
            return True
    def __del__(self):
        #self.delete_temporary_data()
        pass
    def delete_temporary_data(self):
        for file in self._temporary_files[::-1]:
            if os.path.isdir(file):
                try:
                    shutil.rmtree(file)
                except:
                    self.logger.debug('temporary files removal failed with the following error:\n'+traceback.format_exc())
            elif os.path.isfile(file):
                try:
                    os.unlink(file)
                except:
                    self.logger.debug('temporary file removal failed with the following error:\n'+traceback.format_exc())
        self._temporary_files = []
    def make_job(self, stage_index=None, source=None, region_index=None, segment_index=None):
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
    def _collect_results(cls, wd, logger):
        analyses = {}
        output_files = glob.glob(os.path.join(wd, '*.rwa'))
        while output_files:
            output_file = output_files.pop()
            if os.stat(output_file).st_size == 0:
                logger.info('skipping empty file '+output_file)
                continue
            logger.info('reading file: {}...'.format(output_file))
            try:
                __analyses = load_rwa(output_file, lazy=True)
            except:
                traceback.print_exc()
                raise
            try:
                source = __analyses.metadata['datafile']
            except KeyError:
                logger.debug(str(__analyses))
                logger.debug('metadata: ',__analyses.metadata)
                logger.critical('key `datafile` not found in the metadata')
                return
            try:
                _analyses = analyses[source]
            except KeyError:
                analyses[source] = __analyses
            else:
                append_leaf(_analyses, __analyses)
        end_result_files = []
        for source in analyses:
            rwa_file = os.path.splitext(source)[0]+'.rwa'
            logger.info('writing file: {}...'.format(rwa_file))
            save_rwa(os.path.expanduser(rwa_file), analyses[source], force=True)
            end_result_files.append(rwa_file)
        return end_result_files
    def collect_results(self):
        self._collect_results(self.wd, self.logger)
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
        filtered_content = self.cleanup_script_content(content)
        if script in self.temporary_files:
            tmpfile = script # reuse file
        else:
            tmpfile = self.make_temporary_file(suffix='.py', text=True)
        # flush
        with open(tmpfile, 'w') as f:
            for line in filtered_content:
                f.write(line+'\n')
        if main_script:
            self.script = tmpfile
        else:
            return tmpfile
    def cleanup_script_content(self, content):
        filtered_content = []
        for line in content:
            if line.startswith('get_ipython('):
                continue
            elif '.run()' in line:
                # last line
                filtered_content.append(line)
                break
            filtered_content.append(line)
        return filtered_content
    def import_ipynb(self, notebook):
        cmd = 'jupyter nbconvert --to python "{}" --stdout'.format(notebook)
        self.logger.info('running: '+cmd)
        p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = p.communicate()
        if out:
            if not isinstance(out, str):
                out = out.decode('utf-8')
            content = out.splitlines()
        else:
            content = None
        if err:
            if not isinstance(err, str):
                err = err.decode('utf-8')
            if err != '[NbConvertApp] Converting notebook {} to python\n'.format(notebook):
                self.logger.error(err)
        return content
    def interrupt_jobs(self):
        pass


class LocalHost(Env):
    """
    """
    __slots__ = ('running_jobs',)
    def __init__(self, **kwargs):
        Env.__init__(self, **kwargs)
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
                    stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            self.running_jobs.append((j,p))
        self.pending_jobs = []
    def wait_for_job_completion(self, count=None):
        assert self.submit_side
        n = 0
        for j,p in self.running_jobs:
            out, err = p.communicate()
            if out:
                if not isinstance(out, str):
                    out = out.decode('utf-8')
                self.logger.info(out)
            if err:
                if not isinstance(err, str):
                    err = err.decode('utf-8')
                self.logger.error(err)
            self.logger.debug('job {:d} done'.format(j))
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


class Slurm(Env):
    """
    Not supposed to properly run, as TRamWAy is expected to be called
    inside a container;
    see :class:`SlurmOverSSH` instead.
    """
    __slots__ = ('_sbatch_options','_job_id','refresh_interval')
    def __init__(self, **kwargs):
        Env.__init__(self, **kwargs)
        self._sbatch_options = dict(
                output='%J.out',
                error='%J.err',
                )
        self._job_id = None
        self.refresh_interval = 10
    @property
    def sbatch_options(self):
        return self._sbatch_options
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
                output_log = os.path.join(self.wd, output_log)
                self.sbatch_options['output'] = output_log
            error_log = self.sbatch_options['error']
            if not os.path.isabs(os.path.expanduser(error_log)):
                error_log = os.path.join(self.wd, error_log)
                self.sbatch_options['error'] = error_log
    def make_sbatch_script(self, stage=None, path=None):
        assert self.submit_side
        if path is None:
            sbatch_script = self.make_temporary_file(suffix='.sh' if stage is None else '-stage{:d}.sh'.format(stage), text=True)
        else:
            sbatch_script = path
        self.job_name # set default job name if not defined yet
        with open(sbatch_script, 'w') as f:
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
        sbatch = 'sbatch'
        sbatch_script = self.make_sbatch_script()
        self.logger.info('running: {} {}'.format(sbatch, sbatch_script))
        p = subprocess.Popen([sbatch, sbatch_script],
                stderr=subprocess.STDOUT)#, stdout=subprocess.PIPE)
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
                        stderr=subprocess.PIPE, stdout=subprocess.PIPE)
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


class SlurmOverSSH(Slurm):
    """
    Calls *sbatch* through an SSH connection to a Slurm server.
    """
    __slots__ = ('_ssh','remote_dependencies','local_data_location','remote_data_location')
    def __init__(self, **kwargs):
        Slurm.__init__(self, **kwargs)
        from tramway.analyzer.env import ssh
        self._ssh = ssh.Client()
        self.remote_dependencies = None
        self.local_data_location = None
        self.remote_data_location = None
    @property
    def ssh(self):
        return self._ssh
    @property
    def wd_is_available(self):
        return self.worker_side
    def make_working_directory(self):
        if self.submit_side:
            cmd = 'mkdir -p "{}"; mktemp -d -p "{}"'.format(self.working_directory, self.working_directory)
            out, err = self.ssh.exec(cmd)
            if err:
                self.logger.error(err)
            elif out:
                self.working_directory = out.rstrip()
        else:
            assert os.path.isdir(self.wd)
    def setup(self, *argv):
        Slurm.setup(self, *argv)
        if self.worker_side:
            self.script = '/'.join((self.wd, os.path.basename(self.script)))
    def dispatch(self, **kwargs):
        if not kwargs:
            self.prepare_script()
            src = self.script
            dest = os.path.basename(self.script)
            #if dest.endswith('.ipynb'):
            #    dest = dest[:-5]+'py'
            dest = '/'.join((self.wd, dest))
            attrs = self.ssh.put(src, dest)
            self.logger.info(attrs)
            self.script = dest
            self.logger.info('Python script location: '+dest)
            return True
        elif 'stage_options' in kwargs:
            stage_options = kwargs['stage_options']
            for option in stage_options:
                if option == 'sbatch_options':
                    self.sbatch_options.update(stage_options[option])
                else:
                    self.logger.debug('ignoring option: '+option)
    def cleanup_script_content(self, content):
        content = Slurm.cleanup_script_content(self, content)
        filtered_content = []
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
        return filtered_content
    def make_job(self, stage_index=None, source=None, region_index=None, segment_index=None):
        if source is not None:
            if self.remote_data_location:
                source = '/'.join((self.remote_data_location, os.path.basename(source)))
            else:
                home = os.path.expanduser('~')
                if os.path.isabs(source) and os.path.normpath(source).startswith(home):
                    source = '~'+source[len(home):]
        Slurm.make_job(self, stage_index, source, region_index, segment_index)
    def submit_jobs(self):
        sbatch_script = self.make_sbatch_script()
        dest = '/'.join((self.wd, os.path.basename(sbatch_script)))
        attrs = self.ssh.put(sbatch_script, dest)
        self.logger.info(attrs)
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
        try:
            cmd = 'squeue -j {} -h -o "%.2t %.10M %R"'.format(self.job_id)
            while True:
                time.sleep(self.refresh_interval)
                out, err = self.ssh.exec(cmd, shell=True)
                if err:
                    self.logger.error(err.rstrip())
                elif out:
                    out = out.splitlines()[0]
                    out = out.split()
                    status, time_used, reason = out[0], out[1], ' '.join(out[2:])
                    self.logger.info('status: {}   time used: {}   reason: {}'.format(status, time_used, reason))
                else:
                    break
        except:
            self.logger.info('killing jobs with: scancel '+self.job_id)
            self.ssh.exec('scancel '+self.job_id, shell=True)
            raise
    def collect_results(self):
        _prefix = 'OUTPUT_FILES='
        code = """
from tramway.analyzer import environments, BasicLogger

wd = '{}'
files = environments.LocalHost._collect_results(wd, BasicLogger())

print('{}'+';'.join(files))
""".format(self.wd, _prefix)
        local_script = self.make_temporary_file(suffix='.sh', text=True)
        with open(local_script, 'w') as f:
            f.write(code)
        remote_script = '/'.join((self.wd, os.path.basename(local_script)))
        attrs = self.ssh.put(local_script, remote_script, confirm=True)
        self.logger.info(attrs)
        cmd = '{}{} {}; rm {}'.format(
                '' if self.remote_dependencies is None else self.remote_dependencies+'; ',
                self.interpreter, remote_script, remote_script)
        out, err = self.ssh.exec(cmd, shell=True, logger=self.logger)
        if err:
            self.logger.error(err.rstrip())
        if out:
            self.logger.debug(out)
            out = out.splitlines()
            while True:
                try:
                    line = out.pop() # starting from last line
                except IndexError: # empty list
                    raise RuntimeError('missing output: {}...'.format(_prefix))
                if line.startswith(_prefix):
                    end_result_files = line[len(_prefix):].split(';')
                    break
            for end_result_file in end_result_files:
                if not end_result_file:
                    continue
                self.logger.info('retrieving file: '+end_result_file)
                if self.local_data_location:
                    dest = os.path.join(
                            os.path.expanduser(self.local_data_location),
                            os.path.basename(end_result_file),
                        )
                else:
                    dest = end_result_file
                self.ssh.get(end_result_file, dest)
    def delete_temporary_data(self):
        Slurm.delete_temporary_data(self)
        # delete worker-side working directory
        out, err = self.ssh.exec('rm -rf '+self.wd)
        if err:
            self.logger.error(err)
        if out:
            self.logger.info(out)


Environment.register(SlurmOverSSH)


class Tars(SlurmOverSSH):
    """
    Designed for server *tars.pasteur.fr*.
    """
    def __init__(self, **kwargs):
        SlurmOverSSH.__init__(self, **kwargs)
        self.interpreter = 'singularity exec -H $HOME -B /pasteur tramway2-200910.sif python3.6 -s'
        self.remote_dependencies = 'module load singularity'
    @property
    def username(self):
        return None if self.ssh.host is None else self.ssh.host.split('@')[0]
    @username.setter
    def username(self, name):
        self.ssh.host = None if name is None else name+'@tars.pasteur.fr'
        if self.wd is None:
            self.wd = '/pasteur/scratch/users/'+name
    @property
    def container(self):
        parts = self.interpreter.split()
        return parts[parts.index('python3.6')-1]
    @container.setter
    def container(self, path):
        parts = self.interpreter.split()
        p = parts.index('python3.6')
        self.interpreter = ' '.join(parts[:p-1]+[path]+parts[p:])


class GPULab(SlurmOverSSH):
    """
    Designed for server *adm.inception.hubbioit.pasteur.fr*.
    """
    def __init__(self, **kwargs):
        SlurmOverSSH.__init__(self, **kwargs)
        self.interpreter = 'singularity exec -H $HOME tramway2-200910.sif python3.6 -s'
    @property
    def username(self):
        return None if self.ssh.host is None else self.ssh.host.split('@')[0]
    @username.setter
    def username(self, name):
        self.ssh.host = None if name is None else name+'@adm.inception.hubbioit.pasteur.fr'
        if self.wd is None:
            self.wd = '/master/home/{}/scratch'.format(name)
    @property
    def container(self):
        parts = self.interpreter.split()
        return parts[parts.index('python3.6')-1]
    @container.setter
    def container(self, path):
        parts = self.interpreter.split()
        p = parts.index('python3.6')
        self.interpreter = ' '.join(parts[:p-1]+[path]+parts[p:])


__all__ = ['Environment', 'LocalHost', 'SlurmOverSSH', 'Tars', 'GPULab']

