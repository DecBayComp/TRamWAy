
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


class TestPipeline(object):
    def test_LocalHost(self, tmpdir, dynamicmesh):
        tmpdir = tmpdir.strpath
        fd, script_name = tempfile.mkstemp(suffix='.py', dir=tmpdir)
        os.close(fd)
        script = """\
from tramway.analyzer import *
from tramway.analyzer import BasicLogger
import tempfile
import os
import numpy
import pandas
a=RWAnalyzer()
a._logger = BasicLogger() # let subprocess.check_output catch the log output
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
a.env = environments.LocalHost
a.env.worker_count = 4
a.env.script = __file__
def fresh_start(self):
    for f in self.spt_data:
        try:
            os.unlink(f.source[:-3]+'rwa')
        except FileNotFoundError:
            pass
def tessellate(self):
    for f in self.spt_data:
        with f.autosaving() as tree:
            for r in f.roi.as_support_regions():
                df = r.crop()
                sampling = self.sampler.sample(df)
                tree[r.label] = sampling
def reload(self):
    self.spt_data.reload_from_rwa_files()
def map(self):
    for r in self.roi.as_support_regions():
        with r.autosaving() as tree:
            #tree.rwa_file = self.env.make_temporary_file(suffix='.rwa')
            self.logger.info('autosaving in {{}}...'.format(tree.rwa_file))
            sampling = r.get_sampling()
            for ts, s in self.time.as_time_segments(sampling):
                maps = self.mapper.infer(s)
                tree[r.label][self.time.segment_label('n', ts, s)] = maps
a.pipeline.append_stage(fresh_start)
a.pipeline.append_stage(tessellate, granularity='roi')
a.pipeline.append_stage(reload, requires_mutability=True)
a.pipeline.append_stage(map, granularity='time segment')
a.run()
reload(a)
a.logger.info(str(a.spt_data.analyses))
for r in a.roi.as_support_regions():
    sampling = r.get_sampling()
    maps = a.time.combine_segments('n', sampling, True)
a.logger.info(str(a.spt_data.analyses))
""".format(input=dynamicmesh)
        with open(script_name, 'w') as f:
            f.write(script)
        out = subprocess.check_output([sys.executable, script_name], encoding='utf8')
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
        #logger.info(out)

