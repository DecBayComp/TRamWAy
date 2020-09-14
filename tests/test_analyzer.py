
import os
from math import *
import numpy
import random
import subprocess
import pytest
import copy
import warnings
from tramway.core.exceptions import *
from tramway.core.scaler import unitrange
from tramway.helper import *
from tramway.helper.roi import *
from tramway.helper.simulation import *
from tramway.analyzer import *

data_version = '200803'

seed = 4294947105


def reset_random_generator(seed):
    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
def gen_spt_data(generator, dir, filename, seed=None):
    filepath = '/'.join((dir, filename))
    if not os.path.isfile(filepath):
        reset_random_generator(seed)
        traj = generator()
        traj.to_csv(filepath, sep='\t', index=False)

def gen_linear_gradient_D():
    return random_walk_2d(
            n_trajs=400,
            N_mean=5,
            dt=.05,
            D0=.1,
            amplitude_D=.05,
            mode_D='linear',
            sigma_noise=1e-4,
            )
def gen_linear_gradient_V():
    return random_walk_2d(
            n_trajs=400,
            N_mean=5,
            dt=.05,
            D0=.1,
            amplitude_V=2.,
            mode_V='potential_linear',
            sigma_noise=1e-4,
            )
def gen_radial_sink_D():
    return random_walk_2d(
            n_trajs=400,
            N_mean=5,
            dt=.05,
            D0=.1,
            amplitude_D=.05,
            mode_D='gaussian_trap',
            sigma_noise=1e-4,
            )
def gen_radial_sink_V():
    return random_walk_2d(
            n_trajs=400,
            N_mean=5,
            dt=.05,
            D0=.1,
            amplitude_V=2.,
            mode_V='potential_force',
            sigma_noise=1e-4,
            )

def gen_moving_sink_V():
    def force(xy, t):
        sink_r = .17
        t1 = 50.
        t2 = 90.
        if t<t1:
            sink_center = ((t1-t)/t1) * np.r_[-.3,-.1] + (t/t1) * np.r_[.1,.1]
        else:
            sink_center = ((t2-t)/(t2-t1)) * np.r_[.1,.1] + ((t-t1)/(t2-t1)) * np.r_[.25,-.45]
        r = xy - sink_center
        norm = np.sqrt(np.dot(r,r))
        if norm < sink_r:
            scaled_gradV = (r/norm) * (1. + cos(norm/sink_r*pi)) / 2.
            f = 5. * (-scaled_gradV)
        else:
            f = np.zeros_like(r)
        return f
    return random_walk(
            diffusivity=.3,
            force=force,
            trajectory_mean_count=10,
            duration=90.,
            box=(-.5,-.5,1.,1.),
            )


@pytest.fixture
def sptdatafiles(tmpdir):
    testdir = os.path.dirname(__file__)
    testdir = '{}/test_analyzer_{}'.format(testdir, data_version)
    if not os.path.isdir(testdir):
        os.makedirs(testdir)
    sptfile0 = 'test00_linear_diffusivity_gradient.txt'
    gen_spt_data(gen_linear_gradient_D, testdir, sptfile0, seed)
    sptfile1 = 'test01_linear_potential_gradient.txt'
    gen_spt_data(gen_linear_gradient_V, testdir, sptfile1, seed)
    sptfile2 = 'test02_radial_diffusivity_sink.txt'
    gen_spt_data(gen_radial_sink_D, testdir, sptfile2, seed)
    sptfile3 = 'test03_radial_potential_sink.txt'
    gen_spt_data(gen_radial_sink_V, testdir, sptfile3, seed)
    filenames = [sptfile0, sptfile1, sptfile2, sptfile3]
    return testdir, filenames

@pytest.fixture
def one_sptdatafile(sptdatafiles):
    testdir, filenames = sptdatafiles
    return '/'.join((testdir, filenames[0]))
@pytest.fixture
def all_sptdatafiles(sptdatafiles):
    testdir, _ = sptdatafiles
    return testdir+'/*.txt'

def timefree(tree):
    if isinstance(tree.data, Maps):
        try:
            tree.data.runtime = None
        except AttributeError:
            pass
    new_tree = type(tree)(tree.data)
    new_tree.metadata = { k:tree.metadata[k]
            for k in tree.metadata
            if k != 'datetime' }
    for label in tree.labels:
        new_tree.add(timefree(tree[label]), label, tree.comments[label])
    return new_tree


class Common(object):

    def equal(self, tree_a, tree_b, label=None):
        try:
            tree_a = tree_a.statefree()
        except AttributeError:
            pass
        assert type(tree_a) is type(tree_b)
        if label is None:
            label = self.label
        else:
            label = '_'.join((self.label, label))
        file_a = os.path.join(self.tmpdir, label+'_a.rwa')
        file_b = os.path.join(self.tmpdir, label+'_b.rwa')
        save_rwa(file_a, timefree(tree_a), force=True)
        save_rwa(file_b, timefree(tree_b), force=True)
        p = subprocess.Popen(('h5diff', file_a, file_b),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            print(err)
        if out:
            print(out)
        return not (bool(err) or bool(out))

    @property
    def tmpdir(self):
        return self._tmpdir
    @tmpdir.setter
    def tmpdir(self, tmpdir):
        try:
            tmpdir = tmpdir.strpath
        except AttributeError:
            pass
        self._tmpdir = tmpdir


class TestSptDataAccess(Common):

    def test_prms(self, all_sptdatafiles):
        a=RWAnalyzer()
        a.spt_data.from_ascii_files(all_sptdatafiles)
        a.spt_data.localization_precision = .03
        for f in a.spt_data:
            assert f.localization_precision == .03
        for i, f in enumerate(a.spt_data):
            f.localization_precision = float(i) * .01
        assert 1 < i
        try:
            a.spt_data.localization_error
        except AttributeError:
            pass
        else:
            assert False # localization_error is unique and should not
        for f in a.spt_data:
            f.localization_error = .0009
        assert a.spt_data.localization_precision == .03

class TestTesseller(Common):

    def test_full_fov_grid(self, one_sptdatafile, tmpdir):
        self.tmpdir = tmpdir
        self.label = 'full_fov_grid'
        #
        a=RWAnalyzer()
        a.spt_data.from_ascii_file(one_sptdatafile)
        a.spt_data.localization_precision = 1e-4
        a.tesseller.from_callable(tessellers.Squares)
        for i, r in enumerate(a.roi.as_support_regions()):
            assert i == 0
            df = r.discard_static_trajectories(r.crop())
            sampling = a.sampler.sample(df)
            r.add_sampling(sampling, 'full fov grid')
            tree_a = r._spt_data.analyses
        #
        data_b = load_xyt(one_sptdatafile)
        tree_b = Analyses(discard_static_trajectories(data_b, 1e-4**2))
        tessellate(tree_b, 'grid', label='full fov grid')
        tree_b.data = data_b
        #
        assert self.equal(tree_a, tree_b)

    def test_single_roi_kmeans(self, one_sptdatafile, tmpdir):
        self.tmpdir = tmpdir
        self.label = 'single_roi_kmeans'
        #
        a=RWAnalyzer()
        a.spt_data.from_ascii_file(one_sptdatafile)
        a.spt_data.localization_precision = 1e-4
        a.tesseller.from_callable(tessellers.KMeans)
        a.roi.from_squares(numpy.zeros((1,2)), 1.)
        a.sampler.from_voronoi()
        a.sampler.min_location_count = 10
        for i, r in enumerate(a.roi.as_support_regions()):
            assert i == 0
            df = r.discard_static_trajectories(r.crop())
            sampling = a.sampler.sample(df)
            r.add_sampling(sampling, 'single roi kmeans')
            tree_a = r._spt_data.analyses
        #
        data_b = load_xyt(one_sptdatafile)
        tree_b = Analyses(discard_static_trajectories(crop(data_b, [-.5,-.5,.5,.5]), 1e-4**2))
        tessellate(tree_b, 'kmeans', min_n=10, prune=False, label='single roi kmeans')
        tree_b.data = data_b
        #
        assert self.equal(tree_a, tree_b)

    def test_3roi_hexagons(self, one_sptdatafile, tmpdir):
        self.tmpdir = tmpdir
        self.label = '3roi_hexagons'
        #
        roi = [([-.4,-.5],[0.,-.1]),([-.1,0.],[.3,.4]),([.1,-.3],[.5,.1])]
        roi = [ (numpy.array(lb), numpy.array(ub)) for lb, ub in roi ]
        #
        a=RWAnalyzer()
        a.spt_data.from_ascii_file(one_sptdatafile)
        a.tesseller.from_callable(tessellers.Hexagons)
        a.tesseller.ref_distance = .01
        a.tesseller.avg_location_count = 5
        a.roi.from_bounding_boxes(roi, label='3-roi hexagons', group_overlapping_roi=True)
        b = RoiHelper(one_sptdatafile, verbose=False)
        b.set_bounding_boxes(roi, collection_label='3-roi hexagons')
        b.tessellate('hexagon', ref_distance=.01, avg_location_count=5)
        for i, r in enumerate(a.roi.as_support_regions()):
            df = r.crop()
            a.tesseller.reset() # let the calibration parameters vary from a roi to another
            sampling = a.sampler.sample(df)
            r.add_sampling(sampling)
            tree_a = r.analyses
            tree_b = b.analyses[r.label].statefree()
            assert self.equal(tree_a, tree_b)

    def test_dir_gwr(self, all_sptdatafiles, tmpdir):
        self.tmpdir = tmpdir
        self.label = 'dir_gwr'
        #
        a=RWAnalyzer()
        a.spt_data.from_ascii_files(all_sptdatafiles)
        a.tesseller.from_callable(tessellers.GWR)
        for f in a.spt_data:
            print(f.source)
            #
            df = f.dataframe
            a.tesseller.reset()
            reset_random_generator(seed)
            sampling = a.sampler.sample(df)
            f.add_sampling(sampling, 'dir gwr')
            tree_a = f.analyses
            #
            tree_b = Analyses(load_xyt(f.source))
            reset_random_generator(seed)
            tessellate(tree_b, 'gwr', lifetime=.05, complete_delaunay=True,
                    label='dir gwr')
            #
            #print(sampling)
            #print(tree_b['dir gwr'].data)
            assert self.equal(tree_a, tree_b)


@pytest.fixture
def staticmesh(sptdatafiles):
    testdir, filenames = sptdatafiles
    sptdatafile = '/'.join((testdir, filenames[3]))
    a=RWAnalyzer()
    a.spt_data.from_ascii_file(sptdatafile)
    a.tesseller.from_callable(tessellers.Hexagons)
    f = next(iter(a.spt_data))
    mesh = a.tesseller.tessellate(f.dataframe)
    mesh = Analysis(mesh, f)
    sampling = a.sampler.sample(f.dataframe, mesh)
    sampling.commit_as_analysis('hexagons')
    return f.analyses

@pytest.fixture
def dynamicmesh(tmpdir):
    testdir = os.path.dirname(__file__)
    testdir = '{}/test_analyzer_{}'.format(testdir, data_version)
    if not os.path.isdir(testdir):
        os.makedirs(testdir)
    sptfile = 'test04_moving_potential_sink.txt'
    gen_spt_data(gen_moving_sink_V, testdir, sptfile, seed)
    return '/'.join((testdir, sptfile))


class TestMapper(Common):

    def test_stochastic_dv(self, tmpdir, staticmesh):
        self.tmpdir = tmpdir
        self.label = 'sdv'
        #
        a=RWAnalyzer()
        a.spt_data.from_analysis_tree(copy.deepcopy(staticmesh))
        a.spt_data.localization_precision = 1e-4
        a.mapper.from_plugin('stochastic.dv')
        a.mapper.diffusivity_prior = 0
        a.mapper.potential_prior = 1
        a.mapper.ftol = 1e-3
        a.mapper.worker_count = 1
        a.mapper.verbose = False
        #
        f = next(iter(a.spt_data))
        sampling = f.get_sampling('hexagons')
        reset_random_generator(seed)
        mapping = a.mapper.infer(sampling)
        mapping.commit_as_analysis('sdv')
        tree_a = f.analyses
        #
        tree_b = copy.deepcopy(staticmesh)
        reset_random_generator(seed)
        infer(tree_b, 'stochastic.dv',
                sigma = 1e-4,
                diffusivity_prior = 0,
                potential_prior = 1,
                ftol = 1e-3,
                worker_count = 1,
                input_label = 'hexagons',
                output_label = 'sdv',
                )
        #
        print(tree_a)
        print(tree_b)
        print(tree_a['hexagons']['sdv'].data)
        print(tree_b['hexagons']['sdv'].data)
        assert self.equal(tree_a, tree_b.statefree())

    def test_time_regul(self, dynamicmesh, tmpdir):
        self.tmpdir = tmpdir
        self.label = 'time_regul'
        #
        a=RWAnalyzer()
        a.spt_data.from_ascii_file(dynamicmesh)
        a.spt_data.localization_precision = 1e-4
        a.tesseller.from_plugin('hexagon')
        a.time.from_sliding_window(duration=30)
        a.sampler.from_nearest_neighbors(10)
        a.mapper.from_plugin('stochastic.dv')
        a.mapper.diffusivity_prior = 0
        a.mapper.potential_prior = 1
        a.mapper.potential_time_prior = 1
        a.mapper.ftol = 1e-2
        a.mapper.worker_count = 1
        a.mapper.verbose = False
        #
        f = next(iter(a.spt_data))
        sampling = a.sampler.sample(f.dataframe)
        sampling = commit_as_analysis('hexagons', sampling, f)
        reset_random_generator(seed)
        mapping = a.mapper.infer(sampling)
        mapping.commit_as_analysis('sdv')
        tree_a = f.analyses
        #
        tree_b = Analyses(load_xyt(dynamicmesh))
        tessellate(tree_b, 'hexagon', label='hexagons',
                time_window_duration=30,
                knn=10,
                enable_time_regularization=True)
        reset_random_generator(seed)
        infer(tree_b, 'stochastic.dv',
                sigma=1e-4,
                diffusivity_prior=0,
                potential_prior=1,
                potential_time_prior=1,
                ftol=1e-2,
                worker_count=1,
                output_label='sdv',
                )
        #
        assert self.equal(tree_a, tree_b)


class TestIndexer(Common):

    def test_nooverlap_roi(self, dynamicmesh):
        #
        a=RWAnalyzer()
        a.spt_data.from_ascii_file(dynamicmesh)
        a.spt_data.localization_precision = 1e-4
        roi = [[.2,-.1],[-.3,.3],[0.,.1],[-.2,-0.]]
        a.roi.from_squares(np.array(roi), .2, group_overlapping_roi=False)
        #
        for i, r in a.roi.as_individual_roi(index=1, return_index=True):
            assert i == 1
            lb,ub = r.bounding_box
            assert np.all( lb == np.array(roi[1])-.1 )
            assert np.all( ub == np.array(roi[1])+.1 )
        #
        j = 3
        for i, r in a.roi.as_individual_roi(index=[3,2,1,0], return_index=True):
            assert i == j
            j -= 1
            lb,ub = r.bounding_box
            assert np.all( lb == np.array(roi[i])-.1 )
            assert np.all( ub == np.array(roi[i])+.1 )
        #
        j = 0
        for r in a.roi.as_individual_roi(index=set((0,3))):
            lb,ub = r.bounding_box
            assert np.all( lb == np.array(roi[j])-.1 )
            assert np.all( ub == np.array(roi[j])+.1 )
            j += 3
        #
        for r0, r1 in zip(
                a.roi.as_individual_roi(),
                a.roi.as_support_regions(),
                ):
            lb0,ub0 = r0.bounding_box
            lb1,ub1 = r1.bounding_box
            assert np.all( lb0 == lb1 )
            assert np.all( ub0 == ub1 )

    def test_overlapping_roi(self, dynamicmesh):
        #
        a=RWAnalyzer()
        a.spt_data.from_ascii_file(dynamicmesh)
        a.spt_data.localization_precision = 1e-4
        roi = [[.2,-.1],[-.3,.3],[0.,.1],[-.2,-0.]]
        a.roi.from_squares(np.array(roi), .2, group_overlapping_roi=True)
        #
        j = 0
        for i,r in a.roi.as_individual_roi(return_index=True):
            assert i == j
            lb,ub = r.bounding_box
            assert np.all( lb == np.array(roi[j])-.1 )
            assert np.all( ub == np.array(roi[j])+.1 )
            j += 1
        #
        for i,r in a.roi.as_support_regions(return_index=True):
            if i==0:
                lb,ub = r.bounding_box
                assert np.isclose(lb[0], -.3)
                assert np.isclose(lb[1], -.2)
                assert np.isclose(ub[0], .3)
                assert np.isclose(ub[1], .2)
            elif i==1:
                lb,ub = r.bounding_box
                assert lb[0] == roi[1][0]-.1
                assert lb[1] == roi[1][1]-.1
                assert ub[0] == roi[1][0]+.1
                assert ub[1] == roi[1][1]+.1
            else:
                assert i<2 # False

    def test_time_segments(self, dynamicmesh):
        #
        a=RWAnalyzer()
        a.spt_data.from_ascii_file(dynamicmesh)
        #roi = [[.2,-.1],[-.3,.3],[0.,.1],[-.2,-0.]]
        #a.roi.from_squares(np.array(roi), .2)
        #
        a.time.from_sliding_window(duration=10)
        a.time.sync_start_times()
        #
        for r in a.roi.as_support_regions(index=0):
            df = r.crop()
            sampling = a.sampler.sample(df)
            for t, s in a.time.as_time_segments(sampling):
                assert t[0]<=s.points['t'].min()
                assert s.points['t'].max()<=t[1]
        #
        a.mapper.from_plugin('d')
        j = 3
        for d in a.spt_data:
            for r in d.roi.as_support_regions(index=0):
                df = r.crop()
                sampling = a.sampler.sample(df)
                for i,t,s in a.time.as_time_segments(sampling, index=j, return_index=True):
                    assert i == j
                    assert np.all(t == sampling.tessellation.time_lattice[j])
                    assert t[0]<=s.points['t'].min()
                    assert s.points['t'].max()<=t[1]


class TestAssignment(object):
    def test_readonly_properties(self):
        a = RWAnalyzer()
        for prop in ('pipeline',):
            try:
                setattr(a, prop, True)
            except AttributeError:
                pass
            else:
                assert prop is not prop # the mentioned property is writable
    def test_warnings_for_misplaced_attributes(self, one_sptdatafile):
        warnings.simplefilter('error', MisplacedAttributeWarning)
        a = RWAnalyzer()
        a.spt_data.from_ascii_file(one_sptdatafile)
        for attr, val in (
                ('localization_error', 0),
                ('localization_precision', 0),
                ('columns', list('nxyt')),
                ('scaler', unitrange()),
                ('resolution', .1),
                ('dt', .04),
                ('time_step', .05),
                ):
            try:
                setattr(a, attr, val)
            except MisplacedAttributeWarning:
                pass
            else:
                assert attr is not attr # no warning on setting the mentioned attribute
    def test_init_for_misplaced_attributes(self):
        warnings.simplefilter('ignore', MisplacedAttributeWarning)
        a = RWAnalyzer()
        for attr, val, permissive in (
                ('localization_error', 0, False),
                ('localization_precision', 0, False),
                ('columns', list('nxyt'), False),
                ('scaler', unitrange(), False),
                ('resolution', .1, False),
                ('dt', .04, False),
                ('time_step', .05, False),
                ):
            try:
                setattr(a, attr, val)
            except AttributeError:
                if permissive:
                    raise # could not set the attribute before initializing its parent
                else:
                    pass
            else:
                if permissive:
                    pass
                else:
                    assert attr is not attr # could set the attribute before initializing its parent

