
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
import random


py2_hash, py3_hash = 'MGtbXz14', 'sYCBf80j'
data_server = 'http://dl.pasteur.fr/fop/{}/'.format(py2_hash if sys.version_info[0] == 2 else py3_hash)
data_update = '200909'
data_file = 'glycine_receptor.trxyt'

data_dir = '{}_py{}_{}'.format('test_commandline', sys.version_info[0], data_update)
data_archive = '{}.tar.bz2'.format(data_dir)

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

def prepare_file(filename, datadir, tmpdir, copy=True):
    try:
        tmpdir = tmpdir.strpath
    except AttributeError:
        pass
    destfile = os.path.join(tmpdir, filename)
    if not os.path.isfile(destfile) and copy:
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


class TestTessellation(object):

    def print(self, *args, **kwargs):
        _print(self.tmpdir, *args, **kwargs)

    def xytfile(self):
        return prepare_file(data_file, self.datadir, self.tmpdir)

    def rwafile(self, reference):
        basename, _ = os.path.splitext(data_file)
        rwa = 'tessellation_output_{}.rwa'.format(reference)
        return prepare_file(rwa, self.datadir, self.tmpdir)

    def common(self, tmpdir, datadir, cmd, reference=None):
        self.tmpdir, self.datadir = tmpdir, datadir
        input_file = self.xytfile()
        random.seed(seed)
        numpy.random.seed(seed)
        status = execute('{} -m tramway tessellate {} -i {} --seed {}', sys.executable, cmd, input_file, seed)
        assert status == 0
        output_file = '{}.rwa'.format(os.path.splitext(input_file)[0])
        assert os.path.isfile(output_file)
        if reference:
            reference = self.rwafile(reference)
            p = subprocess.Popen(('h5diff', reference, output_file),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            if out:
                if not isinstance(out, str):
                    out = out.decode('utf-8')
                self.print(out)
                out = out.splitlines()
                out = '\n'.join([ '\n'.join((line1, line2)) \
                    for (line1, line2) in zip(out[:-1:2], out[1::2])
                    if not (line1.startswith('Failed reading attribute') or \
                        line1.startswith('attribute: <TITLE of ') or \
                        '/_metadata/' in line1) ])
            if err:
                if not isinstance(err, str):
                    err = err.decode('utf-8')
                self.print(err)
            assert not out

    def test_grid(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'grid -s 50', 'grid0')
    def test_kdtree(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'kdtree', 'kdtree0')
    def test_kmeans(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'kmeans -w -s 40', 'kmeans0')
    def test_gwr(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'gwr -w -ss 6', 'gwr0')
    def test_random(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'random --cell-count 20 -ss 20', 'rand0')
    def test_overlapping_knn(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'random --cell-count 250 -ss 10 --knn 200', 'knn0')
    def test_hexagon(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'hexagon -c 100 --tilt .1', 'hex0')
    def test_nesting(self, tmpdir, datadir):
        output_file = os.path.join(tmpdir.strpath, 'nested_tessellations.rwa')
        label = 'nested'
        reference = 'nesting0'
        self.tmpdir, self.datadir = tmpdir, datadir
        input_file = self.xytfile()
        random.seed(seed)
        numpy.random.seed(seed)
        cmd = 'window --duration 50 --shift 50'
        status = execute('{} -m tramway tessellate {} -i {} -o {} -l {} --seed {}', sys.executable, cmd, input_file, output_file, label, seed)
        assert status == 0
        cmd = 'random -c 100'
        status = execute('{} -m tramway tessellate {} -i {} -L {} --inplace --seed {}', sys.executable, cmd, output_file, label, seed)
        assert status == 0
        if reference:
            reference = self.rwafile(reference)
            p = subprocess.Popen(('h5diff', reference, output_file),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            if out:
                if not isinstance(out, str):
                    out = out.decode('utf-8')
                self.print(out)
                out = out.splitlines()
                out = '\n'.join([ '\n'.join((line1, line2)) \
                    for (line1, line2) in zip(out[:-1:2], out[1::2])
                    if not (line1.startswith('Failed reading attribute') or \
                        line1.startswith('attribute: <TITLE of ') or \
                        '/_metadata/' in line1) ])
            if err:
                if not isinstance(err, str):
                    err = err.decode('utf-8')
                self.print(err)
            assert not out

class TestInference(object):

    def print(self, *args, **kwargs):
        _print(self.tmpdir, *args, **kwargs)

    def common(self, tmpdir, datadir, cmd, reference):
        self.tmpdir = tmpdir
        initial_file = prepare_file('inference_input.rwa', datadir, tmpdir)
        input_file = prepare_file('test_inference_{}.rwa'.format(reference),
            datadir, tmpdir, False)
        ref_file = 'inference_output_{}.rwa'.format(reference)
        i, o = open(initial_file, 'rb'), open(input_file, 'wb')
        o.write(i.read())
        o.close(), i.close()
        status = execute('{} -m tramway infer {} -i {} --seed {}', sys.executable, cmd, input_file, seed)
        assert status == 0
        ref_file = prepare_file(ref_file, datadir, tmpdir)
        generated, expected = input_file, ref_file
        p = subprocess.Popen(('h5diff', expected, generated),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if out:
            if not isinstance(out, str): # Py3
                out = out.decode('utf-8')
            self.print(out)
            out = out.splitlines()
            out = '\n'.join([ '\n'.join((line1, line2)) \
                for (line1, line2) in zip(out[:-1:2], out[1::2])
                if not (line1.startswith('Failed reading attribute') or \
                    line1.startswith('attribute: <TITLE of ') or \
                    line1.endswith('/runtime>') or \
                    '/_metadata/' in line1) ])
            self.print(out)
        if err:
            if not isinstance(err, str): # Py3
                err = err.decode('utf-8')
            self.print(err)
        assert not out

    def test_d(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'd -j', 'd0')
    def test_df(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'df -j', 'df0')
    def test_dd(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'dd -j', 'dd0')
    def test_dv0(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'dv -j --max-iter 10', 'dv0')
    def test_dv1(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'dv -d 1 -v 1 --max-iter 10', 'dv1')
    def test_smooth_d(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'd -d 1 -j --max-iter 10', 'd1')
    def test_smooth_df(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'df -d 1 -j --max-iter 10', 'df1')
    def test_smooth_dd(self, tmpdir, datadir):
        self.common(tmpdir, datadir, 'dd -d 1 -j --max-iter 10', 'dd1')

