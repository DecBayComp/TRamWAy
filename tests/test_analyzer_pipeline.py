
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


def run_script1(tmpdir, dynamicmesh, env='', options=''):
    tmpdir = tmpdir.strpath
    fd, script_name = tempfile.mkstemp(suffix='.py', dir=tmpdir)
    os.close(fd)
    def script(env='', options=''):
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
            os.unlink(f.source[:-3]+'rwa')
        except FileNotFoundError:
            pass
def tessellate(self):
    dry_run = True
    for f in self.spt_data:
        with f.autosaving() as tree:
            assert tree.autosave
            #self.logger.info('autosaving in {{}}...'.format(tree.rwa_file))
            for r in f.roi.as_support_regions():
                df = r.crop()
                sampling = self.sampler.sample(df)
                tree[r.label] = sampling
                assert tree.modified
                dry_run = False
    if dry_run:
        self.logger.info('stage skipped')
def reload(self):
    #self.logger.debug(os.path.splitext(self.spt_data.source)[0]+'.rwa')
    self.spt_data.reload_from_rwa_files()
def map(self):
    dry_run = True
    for r in self.roi.as_support_regions():
        with r.autosaving() as tree:
            assert tree.autosave
            #self.logger.info('autosaving in {{}}...'.format(tree.rwa_file))
            sampling = r.get_sampling()
            for ts, s in self.time.as_time_segments(sampling):
                maps = self.mapper.infer(s)
                tree[r.label][self.time.segment_label('n', ts, s)] = maps
                assert tree.modified
                dry_run = False
    if dry_run:
        self.logger.info('stage skipped')
a.pipeline.append_stage(fresh_start)
a.pipeline.append_stage(tessellate, granularity='roi')
a.pipeline.append_stage(reload, requires_mutability=True)
a.pipeline.append_stage(map, granularity='time segment'{options})
""".format(input=dynamicmesh, env=env, options=options)

    test = """
a.spt_data.reload_from_rwa_files()
a.logger.info(str(a.spt_data.analyses))
for r in a.roi.as_support_regions():
    sampling = r.get_sampling()
    maps = a.time.combine_segments('n', sampling, True)
a.logger.info(str(a.spt_data.analyses))
"""
    env = env + """
a.env.worker_count = 4
a.env.script = __file__
"""
    with open(script_name, 'w') as f:
        f.write(script(env))
        f.write('a.run()\n')
    out = subprocess.check_output([sys.executable, script_name], encoding='utf8')
    logger.info(out)
    with open(script_name, 'w') as f:
        f.write(script())
        f.write(test)
    out = subprocess.check_output([sys.executable, script_name], encoding='utf8')
    logger.info(out)
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

class TestPipeline(object):
    def test_LocalHost(self, tmpdir, dynamicmesh):
        env = "a.env = environments.LocalHost"
        run_script1(tmpdir, dynamicmesh, env)
    def test_GPULab(self, tmpdir, dynamicmesh):
        with open(os.path.join(os.path.dirname(__file__), 'credentials'), 'r') as f:
            username = f.readline().rstrip()
            password = f.readline().rstrip()
        env = """\
a.env = environments.GPULab
a.env.username = '{username}'
a.env.ssh._password = '{password}'
a.env.container = 'tramway-hpc-201230.sif'
#a.env.debug = True
a.env.sbatch_options.update(dict(p='dbc'))\
""".format(username=username, password=password)
#        options = """,
#        sbatch_options=dict(c=a.mapper.worker_count)\
#"""
        input_file = os.path.join('~', os.path.relpath(dynamicmesh, os.path.expanduser('~')))
#        setup = """\
#a.pipeline.early_setup()
#if a.env.submit_side:
#    import logging
#    logfile = os.path.splitext('{input}')[0]+'.log'
#    a.logger.addHandler(logging.FileHandler(os.path.expanduser(logfile)))
#""".format(input=input_file)
        run_script1(tmpdir, input_file, env)

