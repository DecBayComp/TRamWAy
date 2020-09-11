
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
import paramiko
import getpass


class Proxy(object):
    __slots__ = ('_proxied',)
    def __init__(self, proxied):
        self._proxied = proxied
    def __getattr__(self, attrname):
        if attrname == '_proxied':
            raise TypeError('class {} does not declare attribute `_proxied`'.format(type(self)))
        return getattr(self._proxied, attrname)
    def __setattr__(self, attrname, val):
        if attrname == '_proxied':
            object.__setattr__(self, '_proxied', val)
        else:
            setattr(self._proxied, attrname, val)


class Env(AnalyzerNode):
    __slots__ = ('_interpreter','_script','_working_directory','_worker_count',
            '_pending_jobs','_selectors', '_selector_classes', '_temporary_files')
    def __init__(self, wd=None, wc=None, **kwargs):
        AnalyzerNode.__init__(self, **kwargs)
        self._interpreter = 'python3'
        self._script = None
        self._selectors = None
        self._selector_classes = {}
        self._temporary_files = []
        self.working_directory = wd
        self.worker_count = wc
        self.pending_jobs = []
    @property
    def logger(self):
        return self._parent.logger
    @property
    def working_directory(self):
        return self._working_directory
    @working_directory.setter
    def working_directory(self, wd):
        if wd is None:
            wd = os.getcwd()
        else:
            wd = os.path.abspath(os.path.expanduser(wd))
            if not os.path.isdir(wd):
                os.makedirs(wd)
        self._working_directory = wd
    @property
    def wd(self):
        return self.working_directory
    @wd.setter
    def wd(self, wd):
        self.working_directory = wd
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
        if not argv:
            return
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
            if self.script.endswith('.ipynb'):
                self.script = self.script[:-5]+'py'
            for f in self.spt_data_selector(self.analyzer.spt_data):
                f.analyses.rwa_file = self.make_temporary_file(output=True)
                f.analyses.autosave = True
        elif self.script is None:
            raise ValueError('attribute `script` is not set')
        else:
            self.wd = self.make_temporary_file(directory=True)
            self.logger.info('working directory: '+self.wd)
            self.temporary_files.append(self.wd)
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
                sources = self.selectors['source']
            except KeyError:
                sources = None
                def _source(source_arg):
                    return source_arg
            else:
                if isinstance(sources, str):
                    sources = (sources,)
                def _source(source_arg):
                    if source_arg is None:
                        return lambda src: src in sources
                    if callable(source_arg):
                        return lambda src: src in sources and source_arg(src)
                    elif source_arg in sources:
                        return source_arg
                    else:
                        return lambda src: False
            try:
                regions = self.selectors['region_index']
            except KeyError:
                if sources is None:
                    return roi_attr
                def _region(index_arg):
                    return index_arg
            else:
                if isinstance(regions, int):
                    regions = (regions,)
                def _region(index_arg):
                    if index_arg is None:
                        return lambda r: r in regions
                    elif callable(index_arg):
                        return lambda r: r in regions and index_arg(r)
                    elif index_arg in regions:
                        return index_arg
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
                    print('not implemented')
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
    def make_temporary_file(self, output=False, directory=False, standard_location=False, **kwargs):
        _dir = None if standard_location else self.working_directory
        if directory:
            tmpfile = tempfile.mkdtemp(dir=_dir)
        else:
            suffix = kwargs.pop('suffix', '.rwa' if output else None)
            fd, tmpfile = tempfile.mkstemp(dir=_dir, suffix=suffix, **kwargs)
            os.close(fd)
        if not output:
            self._temporary_files.append(tmpfile)
        return tmpfile
    def dispatch(self, **kwargs):
        if not kwargs:
            self.prepare_script()
            return True
    def save_analyses(self, spt_data):
        tmpfile = self.make_temporary_file(suffix='.rwa')
        spt_data.to_rwa_file(tmpfile, force=True)
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
        if stage_index is not None:
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
                logger.warning('skipping empty file '+output_file)
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
                print(__analyses)
                print('metadata: ',__analyses.metadata)
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
            tmpfile = self.make_temporary_file(suffix='.py', text=True, standard_location=True)
            if not os.path.isfile(tmpfile):
                self.temporary_files.append(tmpfile)
        # flush
        with open(tmpfile, 'w') as f:
            for line in filtered_content:
                f.write(line)
        if main_script:
            self.script = tmpfile
        else:
            return tmpfile
    def cleanup_script_content(self, content):
        filtered_content = []
        for line in content:
            if line.startswith('get_ipython('):
                continue
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


class LocalHost(Env):
    """
    This class is of no practical use, as a local script
    will not operate a proper pipeline, but directly run
    the processing steps instead.

    This class is maintained as a prototype for job
    management over multiple hosts.
    """
    __slots__ = ('running_jobs',)
    def __init__(self, wd=None, wc=None, **kwargs):
        Env.__init__(self, wd, wc, **kwargs)
        self.running_jobs = []
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
        for job in self.pending_jobs:
            self.logger.debug('submitting: '+( ' '.join(['{}']*(len(job)+2)).format(self.interpreter, self.script, *job) ))
            p = subprocess.Popen([self.interpreter, self.script, *job],
                    stderr=subprocess.STDOUT)#, stdout=subprocess.PIPE)
            self.running_jobs.append(p)
        self.pending_jobs = []
    def wait_for_job_completion(self):
        assert self.submit_side
        for p in self.running_jobs:
            out, err = p.communicate()
            if out:
                print(out)
            if err:
                print(err)
        self.running_jobs = []

Environment.register(LocalHost)


class Slurm(Env):
    __slots__ = ('_sbatch_options','_job_id','_sbatch_script','refresh_interval')
    def __init__(self, wd=None, wc=None, **kwargs):
        Env.__init__(self, wd, wc, **kwargs)
        self._sbatch_options = dict(
                output='%J.out',
                error='%J.err',
                )
        self._job_id = None
        self._sbatch_script = None
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
    def setup(self, *argv):
        Env.setup(self, *argv)
        if self.submit_side:
            output_log = self.sbatch_options['output']
            if not os.path.isabs(os.path.expanduser(output_log)):
                output_log = os.path.join(self.wd, output_log)
                self.sbatch_options['output'] = output_log
            error_log = self.sbatch_options['error']
            if not os.path.isabs(os.path.expanduser(error_log)):
                error_log = os.path.join(self.wd, error_log)
                self.sbatch_options['error'] = error_log
    def make_sbatch_script(self, path=None):
        if path is None:
            sbatch_script = self.make_temporary_file(standard_location=True)
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
            f.write('#SBATCH --array=0-{:d}\n'.format(len(self.pending_jobs)-1))
            f.write('\ndeclare -a tasks\n')
            for j, job in enumerate(self.pending_jobs):
                f.write('tasks[{:d}]="{}"\n'.format(j,
                    ' '.join(['{}']*len(job)).format(*job).replace('"',r'\"')))
            #f.write('\npushd "{}" > /dev/null\n'.format(self.wd))
            f.write('\n{} {} ${{tasks[SLURM_ARRAY_TASK_ID]}}\n'.format(self.interpreter, self.script))
            #f.write('\npopd > /dev/null\n')
            f.write('\nunset tasks\n')
        self._sbatch_script = sbatch_script
    @property
    def sbatch_script(self):
        if self._sbatch_script is None:
            self.make_sbatch_script()
        return self._sbatch_script
    def submit_jobs(self):
        sbatch = 'sbatch'
        self.logger.info('running: {} {}'.format(sbatch, self.sbatch_script))
        p = subprocess.Popen([sbatch, self.sbatch_script],
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
        while True:
            time.sleep(self.refresh_interval)
            p = subprocess.Popen(('squeue', '-j '+self.job_id, '-h', '-o "%.2t %.10M %R"'),
                    stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = p.communicate()
            if err:
                self.logger.error(err)
            elif out:
                out = out.splitlines()[0]
                out = out.split(' ')
                status, time_used, reason = out[0], out[1], ' '.join(out[2:])
                self.logger.info('status: {}   time used: {}   reason: {}'.format(status, time_used, reason))
            else:
                break


class SlurmOverSSH(Slurm):
    __slots__ = ('ssh_host','_ssh_conn','_sftp_conn','_password',
            'remote_dependencies','local_data_location','remote_data_location')
    def __init__(self, wd=None, wc=None, **kwargs):
        Slurm.__init__(self, wd, wc, **kwargs)
        self.ssh_host = None
        self._ssh_conn = None
        self._sftp_conn = None
        self._password = None
        self.remote_dependencies = None
        self.local_data_location = None
        self.remote_data_location = None
    @property
    def ssh_conn(self):
        if self._ssh_conn is None:
            user, host = self.ssh_host.split('@')
            self._ssh_conn = paramiko.SSHClient()
            self._ssh_conn.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self._ssh_conn.load_system_host_keys()
            self._ssh_conn.connect(host, 22, user, self.password)
        return self._ssh_conn
    @property
    def sftp_conn(self):
        if self._sftp_conn is None:
            self._sftp_conn = self.ssh_conn.open_sftp()
        return self._sftp_conn
    @property
    def password(self):
        if self._password is None:
            self._password = getpass.getpass(self.ssh_host+"'s password: ")
        return self._password
    @property
    def working_directory(self):
        return self._working_directory
    @working_directory.setter
    def working_directory(self, wd):
        if self.worker_side:
            if wd is None:
                wd = os.getcwd()
            else:
                wd = os.path.abspath(os.path.expanduser(wd))
                if not os.path.isdir(wd):
                    os.makedirs(wd)
        self._working_directory = wd
    def make_working_directory(self):
        if self.submit_side:
            cmd = ' '.join(['mkdir','-p',self.working_directory])
            _in, out, err = self.ssh_conn.exec_command(cmd)
            out = out.read() # should be silent
            if out:
                self.logger.info(out)
            err = err.read()
            if err:
                self.logger.error(err)
        else:
            wd = os.path.abspath(os.path.expanduser(wd))
            if not os.path.isdir(wd):
                os.makedirs(wd)
    def make_temporary_file(self, output=False, directory=False, standard_location=False, local=False, **kwargs):
        local = local or standard_location
        if self.worker_side or local:
            return Slurm.make_temporary_file(self, output, directory, local, **kwargs)
        else:
            cmd = ['mktemp','-p','"{}"'.format(self.wd)]
            if directory:
                cmd.append('-d')
                self.make_working_directory()
            _in, out, err = self.ssh_conn.exec_command(' '.join(cmd))
            err = err.read()
            if err:
                self.logger.error(err)
            out = out.read()
            if out:
                if not isinstance(out, str):
                    out = out.decode('utf-8')
                return out.rstrip()
            else:
                raise RuntimeError('temporary file creation failed')
    def setup(self, *argv):
        Slurm.setup(self, *argv)
        if self.worker_side:
            self.script = '/'.join((self.wd, os.path.basename(self.script)))
    def dispatch(self, **kwargs):
        if not kwargs:
            self.prepare_script()
            dest = os.path.basename(self.script)
            #if dest.endswith('.ipynb'):
            #    dest = dest[:-5]+'py'
            dest = '/'.join((self.wd, dest))
            attrs = self.sftp_conn.put(src, dest, confirm=False)
            self.logger.info(attrs)
            self.script = dest
            self.logger.info('Python script location: '+dest)
            return True
    def cleanup_script_content(self, content):
        content = Slurm.cleanup_script_content(content)
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
        if source is not None and self.remote_data_location:
            source = '/'.join((self.remote_data_location, os.path.basename(source)))
        Slurm.make_job(self, stage_index, source, region_index, segment_index)
    def submit_jobs(self):
        dest = '/'.join((self.wd, os.path.basename(self.sbatch_script)))
        attrs = self.sftp_conn.put(self.sbatch_script, dest, confirm=False)
        self.logger.info(attrs)
        self.logger.info('sbatch script transferred to: '+dest)
        #self.logger.info('running: module load singularity')
        #_in, out, err = self.ssh_conn.exec_command('module load singularity')
        if self.remote_dependencies:
            cmd = self.remote_dependencies+'; sbatch '+dest
        else:
            cmd = 'sbatch '+dest
        cmd = 'bash -l -c "{}"'.format(cmd.replace('"',r'\"'))
        self.logger.info('running: '+cmd)
        _in, out, err = self.ssh_conn.exec_command(cmd)
        out = out.read()
        if out:
            if not isinstance(out, str):
                out = out.decode('utf-8')
            out = out.rstrip()
            self.logger.debug(out)
            if out.startswith('Submitted batch job '):
                self.job_id = out.split(' ')[-1]
        err = err.read()
        if err:
            if not isinstance(err, str):
                err = err.decode('utf-8')
            self.logger.error(err.rstrip())
        self.pending_jobs = []
    def wait_for_job_completion(self):
        cmd = 'squeue -j {} -h -o "%.2t %.10M %R"'.format(self.job_id)
        cmd = 'bash -l -c "{}"'.format(cmd.replace('"',r'\"'))
        while True:
            time.sleep(self.refresh_interval)
            _in, out, err = self.ssh_conn.exec_command(cmd)
            err = err.read()
            if err:
                if not isinstance(err, str):
                    err = err.decode('utf-8')
                self.logger.error(err.rstrip())
            else:
                out = out.read()
                if out:
                    if not isinstance(out, str):
                        out = out.decode('utf-8')
                    out = out.splitlines()[0]
                    out = out.split()
                    status, time_used, reason = out[0], out[1], ' '.join(out[2:])
                    self.logger.info('status: {}   time used: {}   reason: {}'.format(status, time_used, reason))
                else:
                    break
    def collect_results(self):
        code = """
from tramway.analyzer import environments, BasicLogger

wd = '{}'
files = environments.LocalHost._collect_results(wd, BasicLogger())

print(';'.join(files))
""".format(self.wd)
        local_script = self.make_temporary_file(local=True)
        with open(local_script, 'w') as f:
            f.write(code)
        remote_script = '/'.join((self.wd, os.path.basename(local_script)))
        attrs = self.sftp_conn.put(local_script, remote_script)
        self.logger.info(attrs)
        cmd = '{}{} {}; rm {}'.format(
                '' if self.remote_dependencies is None else self.remote_dependencies+'; ',
                self.interpreter, remote_script, remote_script)
        cmd = 'bash -l -c "{}"'.format(cmd.replace('"',r'\"'))
        _in, out, err = self.ssh_conn.exec_command(cmd)
        err = err.read()
        if err:
            if not isinstance(err, str):
                err = err.decode('utf-8')
            self.logger.error(err.rstrip())
        out = out.read()
        if out:
            if not isinstance(out, str):
                out = out.decode('utf-8')
            end_result_files = out.splitlines()[-1].split(';')
            for end_result_file in end_result_files:
                self.logger.info('retrieving file: '+end_result_file)
                self.sftp_conn.get(end_result_file,
                        os.path.join(os.path.expanduser(self.local_data_location),
                            os.path.basename(end_result_file)))


Environment.register(SlurmOverSSH)


class Tars(SlurmOverSSH):
    def __init__(self, wd=None, wc=None, **kwargs):
        SlurmOverSSH.__init__(self, wd, wc, **kwargs)
        self.interpreter = 'singularity exec -H $HOME:/home -B /pasteur tramway2-200907.sif python3.6 -s'
        self.remote_dependencies = 'module load singularity'
    @property
    def username(self):
        return None if self.ssh_host is None else self.ssh_host.split('@')[0]
    @username.setter
    def username(self, name):
        self.ssh_host = None if name is None else name+'@tars.pasteur.fr'
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


__all__ = ['Environment', 'LocalHost', 'SlurmOverSSH', 'Tars']

