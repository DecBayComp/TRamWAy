
import os
import sys
import numpy
import pandas
import pytest
import tempfile
import subprocess
from tramway.analyzer import *
from test_analyzer import reset_random_generator, sptdatafiles, one_sptdatafile, all_sptdatafiles, timefree, staticmesh, dynamicmesh

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def run_script0(tmpdir, dynamicmesh, env='', options=''):
    tmpdir = tmpdir.strpath
    fd, script_name = tempfile.mkstemp(suffix='.py', dir=tmpdir)
    os.close(fd)

    def script(env='', options=''):
        #
        return """\
from tramway.analyzer import *
from tramway.analyzer import BasicLogger
import tempfile
import os
import numpy
import pandas
a=RWAnalyzer()
logger = a.logger
a._logger = BasicLogger() # let subprocess.check_output catch the log output
assert os.path.isfile(os.path.expanduser('{input}'))
a.spt_data.from_ascii_file('{input}')
a.spt_data.localization_precision = 1e-4
roi = [[.2,-.1],[-.3,.3],[0.,.1],[-.2,-0.]]
a.roi.from_squares(numpy.array(roi), .2, group_overlapping_roi=True)
assert len(list(a.roi.as_support_regions()))==2
a.tesseller = tessellers.Hexagons
a.time.from_sliding_window(30)
def infer(cells):
    i, n = zip(*[ (cell.index, len(cell)) for cell in cells.values() ])
    return pandas.DataFrame(dict(n=list(n)), index=list(i))
a.mapper.from_plugin(infer)
{env}
def fresh_start(self):
    for f in self.spt_data:
        try:
            os.unlink(os.path.expanduser(f.source[:-3]+'rwa'))
        except FileNotFoundError:
            pass
def tessellate(self):
    dry_run = True
    for f in self.spt_data:
        assert not os.path.exists(os.path.expanduser(f.source[:-3]+'rwa'))
        with f.autosaving() as tree:
            assert tree.autosave
            self.logger.info('autosaving in {{}}...'.format(tree.rwa_file))
            for r in f.roi.as_support_regions():
                df = r.crop()
                sampling = self.sampler.sample(df)
                tree[r.label] = sampling
                assert tree.modified
                self.logger.info('#segments: {{}}'.format(len(sampling.tessellation.time_lattice)))
                self.logger.info(str(f.analyses))
                dry_run = False
    if dry_run:
        self.logger.info('stage skipped')
        assert False
def reload(self):
    #self.logger.debug(os.path.splitext(self.spt_data.source)[0]+'.rwa')
    self.spt_data.reload_from_rwa_files()
def map(self):
    dry_run = True
    for r in self.roi.as_support_regions():
        with r.autosaving() as tree:
            assert tree.autosave
            self.logger.info('autosaving in {{}}...'.format(tree.rwa_file))
            sampling = r.get_sampling()
            self.logger.info('#segments: {{}}'.format(self.time.n_time_segments(sampling)))
            for ts, s in self.time.as_time_segments(sampling):
                maps = self.mapper.infer(s)
                tree[r.label][self.time.segment_label('n', ts, s)] = maps
                assert tree.modified
                dry_run = False
    if dry_run:
        self.logger.info('stage skipped')
        assert False
a.pipeline.append_stage(fresh_start, run_everywhere=True)
a.pipeline.append_stage(tessellate, granularity='roi')
a.pipeline.append_stage(reload, requires_mutability=True)
a.pipeline.append_stage(map, granularity='time segment'{options})
""".format(input=dynamicmesh.replace('\\','/'), env=env, options=options)
        #

    #
    test = """
a.spt_data.reload_from_rwa_files()
a.logger.info(str(a.spt_data.analyses))
for r in a.roi.as_support_regions():
    sampling = r.get_sampling()
    maps = a.time.combine_segments('n', sampling, True)
a.logger.info(str(a.spt_data.analyses))
"""
    #
    env = env + """
a.env.worker_count = 4
a.env.script = __file__
"""
    #

    with open(script_name, 'w') as f:
        f.write(script(env))
        f.write('a.run()\n')
    out = subprocess.check_output([sys.executable, script_name], encoding='utf8', timeout=360)
    logger.info(out)
    with open(script_name, 'w') as f:
        f.write(script())
        f.write(test)
    out = subprocess.check_output([sys.executable, script_name], encoding='utf8', timeout=180)
    logger.info(out)

    #
    assert out.endswith("""\
<class 'pandas.core.frame.DataFrame'>
	'roi000-002-003' <class 'tramway.tessellation.base.Partition'>
		'n -- t=0.05-30.05s' <class 'tramway.inference.base.Maps'>
		'n -- t=30.05-60.05s' <class 'tramway.inference.base.Maps'>
		'n -- t=60.05-90.05s' <class 'tramway.inference.base.Maps'>
	'roi001' <class 'tramway.tessellation.base.Partition'>
		'n -- t=0.15-30.15s' <class 'tramway.inference.base.Maps'>
		'n -- t=30.15-60.15s' <class 'tramway.inference.base.Maps'>
		'n -- t=60.15-90.15s' <class 'tramway.inference.base.Maps'>
<class 'pandas.core.frame.DataFrame'>
	'roi000-002-003' <class 'tramway.tessellation.base.Partition'>
		'n' <class 'tramway.inference.base.Maps'>
	'roi001' <class 'tramway.tessellation.base.Partition'>
		'n' <class 'tramway.inference.base.Maps'>
""")
    #


def run_script1(tmpdir, dynamicmesh, env='', options=''):
    tmpdir = tmpdir.strpath
    fd, script_name = tempfile.mkstemp(suffix='.py', dir=tmpdir)
    os.close(fd)

    def script(env='', options=''):
        #
        return """\
from tramway.analyzer import *
from tramway.analyzer import BasicLogger
from tramway.core.xyt import *
import os.path
import numpy
import pandas
a=RWAnalyzer()
logger = a.logger
a._logger = BasicLogger() # let subprocess.check_output catch the log output
assert os.path.isfile(os.path.expanduser('{input}'))
a.spt_data.from_ascii_files('{input}') # should work similarly to from_ascii_file
a.spt_data.localization_precision = 1e-4
a.tesseller = tessellers.GWR
a.time.from_sliding_window(20)
a.time.window_shift = 10
translocations = single(a.spt_data).dataframe
trajectories = translocations_to_trajectories(translocations)
def infer(cells):
    ids, tlen = [], []
    for i in cells:
        traj_ids = numpy.unique(cells[i].n)
        traj_ls = []
        for traj_ix in traj_ids:
            traj_len = numpy.sum(trajectories['n'].values==traj_ix)
            traj_ls.append(traj_len)
        if traj_ls:
            ids.append(i)
            tlen.append(numpy.mean(traj_ls))
    return pandas.DataFrame({{'traj. len.': tlen}}, index=ids)
a.mapper.from_plugin(infer)
{env}
def fresh_start(self):
    for f in self.spt_data:
        try:
            os.unlink(os.path.expanduser(f.source[:-3]+'rwa'))
        except FileNotFoundError:
            pass
a.pipeline.append_stage(fresh_start, run_everywhere=True)
a.pipeline.append_stage(stages.tessellate(label='gwr'))
a.pipeline.append_stage(stages.reload())
a.pipeline.append_stage(stages.infer('traj. features', sampling_label='gwr'))
""".format(input=dynamicmesh.replace('\\','/'), env=env, options=options)
        #

    #
    test = """
a.spt_data.reload_from_rwa_files()
a.logger.info(str(single(a.spt_data).analyses))
"""
    #
    env = env + """
a.env.worker_count = 4
a.env.script = __file__
"""
    #

    with open(script_name, 'w') as f:
        f.write(script(env))
        f.write('a.run()\n')
    out = subprocess.check_output([sys.executable, script_name], encoding='utf8', timeout=360)
    logger.info(out)
    with open(script_name, 'w') as f:
        f.write(script())
        f.write(test)
    out = subprocess.check_output([sys.executable, script_name], encoding='utf8', timeout=180)
    logger.info(out)

    #
    assert out.endswith("""\
<class 'pandas.core.frame.DataFrame'>
	'gwr' <class 'tramway.tessellation.base.Partition'>
		'traj. features' <class 'tramway.inference.base.Maps'>
""")
    #


def run_script2(tmpdir, dynamicmesh, env='', options='', overwrite=False, script1_ran_before=True):
    tmpdir = tmpdir.strpath
    fd, script_name = tempfile.mkstemp(suffix='.py', dir=tmpdir)
    os.close(fd)

    def script(env='', options=''):
        #
        return """\
from tramway.analyzer import *
from tramway.analyzer import BasicLogger
import tempfile
import os
import numpy
import pandas
a=RWAnalyzer()
logger = a.logger
a._logger = BasicLogger() # let subprocess.check_output catch the log output
assert os.path.isfile(os.path.expanduser('{input}'))
a.spt_data.from_ascii_file('{input}')
a.spt_data.localization_precision = 1e-4
roi = [[.2,-.1],[-.3,.3],[0.,.1],[-.2,-0.]]
a.roi.from_squares(numpy.array(roi), .2, group_overlapping_roi=True)
assert len(list(a.roi.as_support_regions()))==2
a.tesseller = tessellers.Hexagons
a.time.from_sliding_window(30)
def infer(cells):
    i, n = zip(*[ (cell.index, len(cell)) for cell in cells.values() ])
    return pandas.DataFrame(dict(n=list(n)), index=list(i))
a.mapper.from_plugin(infer)
{env}
def fresh_start(self):
    for f in self.spt_data:
        try:
            os.unlink(os.path.expanduser(f.source[:-3]+'rwa'))
        except FileNotFoundError:
            pass
a.pipeline.append_stage(stages.tessellate_and_infer(map_label='n', overwrite={overwrite}))
""".format(input=dynamicmesh.replace('\\','/'), env=env, options=options,
        overwrite=overwrite)
        #

    #
    test = """
a.spt_data.reload_from_rwa_files()
a.logger.info(str(a.spt_data.analyses))
"""
    #
    env = env + """
a.env.worker_count = 4
a.env.script = __file__
"""
    #

    with open(script_name, 'w') as f:
        f.write(script(env))
        f.write('a.run()\n')
    out = subprocess.check_output([sys.executable, script_name], encoding='utf8', timeout=360)
    logger.info(out)
    with open(script_name, 'w') as f:
        f.write(script())
        f.write(test)
    out = subprocess.check_output([sys.executable, script_name], encoding='utf8', timeout=120)
    logger.info(out)

    #
    test2 = out.endswith("""\
<class 'pandas.core.frame.DataFrame'>
	'gwr' <class 'tramway.tessellation.base.Partition'>
		'traj. features' <class 'tramway.inference.base.Maps'>
	'roi000-002-003' <class 'tramway.tessellation.base.Partition'>
		'n' <class 'tramway.inference.base.Maps'>
	'roi001' <class 'tramway.tessellation.base.Partition'>
		'n' <class 'tramway.inference.base.Maps'>
""")
    test1 = out.endswith("""\
<class 'pandas.core.frame.DataFrame'>
	'roi000-002-003' <class 'tramway.tessellation.base.Partition'>
		'n' <class 'tramway.inference.base.Maps'>
	'roi001' <class 'tramway.tessellation.base.Partition'>
		'n' <class 'tramway.inference.base.Maps'>
""")
    if script1_ran_before:
        if test1:
            logger.warning("test 2 passes assuming test 1 failed")
        else:
            assert test2
    else:
        assert test1


class TestPipeline(object):

    def test_LocalHost0(self, tmpdir, dynamicmesh):
        env = "a.env = environments.LocalHost"
        run_script0(tmpdir, dynamicmesh, env)

    def test_Maestro0(self, tmpdir, dynamicmesh):
        with open(os.path.join(os.path.dirname(__file__), 'maestro.credentials'), 'r') as f:
            username = f.readline().rstrip()
            password = f.readline().rstrip()
        #
        env = """\
a.env = environments.Maestro
a.env.username = '{username}'
a.env.ssh._password = '{password}'
a.env.sbatch_options.update(dict(p='dbc_pmo', qos='dbc'))\
""".format(username=username, password=password)
        #
        input_file = os.path.join('~', os.path.relpath(dynamicmesh, os.path.expanduser('~')))
        input_file = '~/Projects/TRamWAy/tests/test_analyzer_200803/test04_moving_potential_sink.txt'
        run_script0(tmpdir, input_file, env)

    def test_GPULab0(self, tmpdir, dynamicmesh):
        with open(os.path.join(os.path.dirname(__file__), 'gpulab.credentials'), 'r') as f:
            username = f.readline().rstrip()
            password = f.readline().rstrip()
        #
        env = """\
a.env = environments.GPULab
a.env.username = '{username}'
a.env.ssh._password = '{password}'
a.env.sbatch_options.update(dict(p='dbc'))\
""".format(username=username, password=password)
        #
        input_file = os.path.join('~', os.path.relpath(dynamicmesh, os.path.expanduser('~')))
        run_script0(tmpdir, input_file, env)


    def test_LocalHost1(self, tmpdir, dynamicmesh):
        env = "a.env = environments.LocalHost"
        run_script1(tmpdir, dynamicmesh, env)

    def test_Maestro1(self, tmpdir, dynamicmesh):
        with open(os.path.join(os.path.dirname(__file__), 'maestro.credentials'), 'r') as f:
            username = f.readline().rstrip()
            password = f.readline().rstrip()
        #
        env = """\
a.env = environments.Maestro
a.env.username = '{username}'
a.env.ssh._password = '{password}'
#a.env.container = 'tramway-hpc-test.sif'
a.env.sbatch_options.update(dict(p='dbc_pmo', qos='dbc'))\
""".format(username=username, password=password)
        #
        input_file = os.path.join('~', os.path.relpath(dynamicmesh, os.path.expanduser('~')))
        run_script1(tmpdir, input_file, env)

    def test_GPULab1(self, tmpdir, dynamicmesh):
        with open(os.path.join(os.path.dirname(__file__), 'gpulab.credentials'), 'r') as f:
            username = f.readline().rstrip()
            password = f.readline().rstrip()
        #
        env = """\
a.env = environments.GPULab
a.env.username = '{username}'
a.env.ssh._password = '{password}'
a.env.sbatch_options.update(dict(p='dbc'))\
""".format(username=username, password=password)
        #
        input_file = os.path.join('~', os.path.relpath(dynamicmesh, os.path.expanduser('~')))
        run_script1(tmpdir, input_file, env)

    def test_LocalHost2(self, tmpdir, dynamicmesh):
        env = "a.env = environments.LocalHost"
        run_script2(tmpdir, dynamicmesh, env)

    def test_Maestro2(self, tmpdir, dynamicmesh):
        with open(os.path.join(os.path.dirname(__file__), 'maestro.credentials'), 'r') as f:
            username = f.readline().rstrip()
            password = f.readline().rstrip()
        #
        env = """\
a.env = environments.Maestro
a.env.username = '{username}'
a.env.ssh._password = '{password}'
a.env.sbatch_options.update(dict(p='dbc_pmo', qos='dbc'))\
""".format(username=username, password=password)
        #
        input_file = os.path.join('~', os.path.relpath(dynamicmesh, os.path.expanduser('~')))
        input_file = '~/Projects/TRamWAy/tests/test_analyzer_200803/test04_moving_potential_sink.txt'
        run_script2(tmpdir, input_file, env)

    def test_GPULab2(self, tmpdir, dynamicmesh):
        with open(os.path.join(os.path.dirname(__file__), 'gpulab.credentials'), 'r') as f:
            username = f.readline().rstrip()
            password = f.readline().rstrip()
        #
        env = """\
a.env = environments.GPULab
a.env.username = '{username}'
a.env.ssh._password = '{password}'
a.env.sbatch_options.update(dict(p='dbc'))\
""".format(username=username, password=password)
        #
        input_file = os.path.join('~', os.path.relpath(dynamicmesh, os.path.expanduser('~')))
        run_script2(tmpdir, input_file, env)

