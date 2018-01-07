
from __future__ import print_function

import os
import sys
import subprocess
import hashlib
try:
	from urllib.request import urlretrieve
except: # Python2
	from urllib import urlretrieve
import pytest
import numpy
import shutil
import tarfile
import traceback


data_server = 'http://dl.pasteur.fr/fop/WjVX3div/'
data_archive = 'glycine_receptor_171218.tar.bz2'
data_file = 'glycine_receptor.trxyt'
data_dir = 'data'

seed = 4294947105


def _print(tmpdir, *args, **kwargs):
	try:
		logfile = os.path.join(tmpdir.strpath, 'test.log')
	except AttributeError:
		logfile = os.path.join(tmpdir, 'test.log')
	with open(logfile, 'a') as f:
		kwargs['file'] = f
		print(*args, **kwargs)

@pytest.fixture
def datadir(tmpdir, request):
	tests_dir = _dir = os.path.dirname(request.module.__file__)
	if data_dir:
		_dir = os.path.join(_dir, data_dir)
	if not os.path.isdir(_dir):
		_print(tmpdir, 'downloading {}... '.format(data_archive), end=' ')
		dest = os.path.join(tests_dir, data_archive)
		try:
			urlretrieve(os.path.join(data_server, data_archive), dest)
		except:
			_print(tmpdir, '[failed]')
			_dir = None
		else:
			_print(tmpdir, '[done]')
			_print(tmpdir, 'extracting {}... '.format(data_archive), end=' ')
			try:
				with tarfile.open(dest) as archive:
					archive.extractall(tests_dir)
			except:
				_print(tmpdir, '[failed]')
				_print(traceback.format_exc())
				_dir = None
			else:
				_print(tmpdir, '[done]')
				if not os.path.isdir(_dir):
					_dir = None
	if _dir is None:
		raise OSError('test data not found')
	return _dir

def prepare_file(filename, datadir, tmpdir):
	try:
		tmpdir = tmpdir.strpath
	except AttributeError:
		pass
	destfile = os.path.join(tmpdir, filename)
	if not os.path.isfile(destfile):
		srcfile = os.path.join(datadir, filename)
		if os.path.isfile(srcfile):
			shutil.copyfile(srcfile, destfile)
		else:
			raise OSError('file not found: {}'.format(filename))
	return destfile

def execute(cmd, *args):
	if args:
		cmd = cmd.format(*args)
	return subprocess.call(cmd.split())


class TestTesselation(object):

	def print(self, *args, **kwargs):
		_print(self.tmpdir, *args, **kwargs)

	def xytfile(self):
		return prepare_file(data_file, self.datadir, self.tmpdir)

	def rwafile(self, reference):
		basename, _ = os.path.splitext(data_file)
		rwa = 'test_{}_{}.rwa'.format(reference, basename)
		return prepare_file(rwa, self.datadir, self.tmpdir)

	def common(self, tmpdir, datadir, cmd, reference=None):
		self.tmpdir, self.datadir = tmpdir, datadir
		numpy.random.seed(seed)
		input_file = self.xytfile()
		status = execute('{} -m tramway sample -m {} -i {}', sys.executable, cmd, input_file)
		assert status == 0
		output_file = '{}.rwa'.format(os.path.splitext(input_file)[0])
		assert os.path.isfile(output_file)
		if reference:
			reference = self.rwafile(reference)
			p = subprocess.Popen(('h5diff', reference, output_file))
			out, err = p.communicate()
			if out:
				self.print(out)
			if err:
				self.print(err)
			assert not out

	def test_grid(self, tmpdir, datadir):
		self.common(tmpdir, datadir, 'grid -c 50', 'grid0')
	def test_kdtree(self, tmpdir, datadir):
		self.common(tmpdir, datadir, 'kdtree', 'kdtree0')
	def test_kmeans(self, tmpdir, datadir):
		self.common(tmpdir, datadir, 'kmeans -w -c 40', 'kmeans0')
	def test_gwr(self, tmpdir, datadir):
		self.common(tmpdir, datadir, 'gwr', 'gwr0')
	def test_overlapping_knn(self, tmpdir, datadir):
		self.common(tmpdir, datadir, 'grid -c 20 --knn 80', 'knn0')
#	def test_plot_mesh(self, tmpdir, datadir):
#		pass

#class InferenceTest(object):
#	def test_d(self, tmpdir):
#		pass
#	def test_df(self, tmpdir):
#		pass
#	def test_dd(self, tmpdir):
#		pass
#	def test_dv(self, tmpdir):
#		pass
#	def test_plot_map(self, tmpdir):
#		pass

