
import glob
import os.path
from math import *
import numpy as np
import tempfile
from tramway.helper import *
from tramway.helper.simulation import *

try:
    trajectory_file = glob.glob(os.path.join(tempfile.gettempdir(), '.*_tutorial0\.txt'))[0]
except IndexError:
    _fd, trajectory_file = tempfile.mkstemp('_tutorial0.txt')
    os.close(_fd)
    _fd, rwa_file = tempfile.mkstemp('_tutorial0.rwa')
    os.close(_fd)
    _fd, trajectories_with_time_varying_properties = tempfile.mkstemp('_tutorial1.txt')
    os.close(_fd)
    _fd, analyses_with_time_varying_properties = tempfile.mkstemp('_tutorial1.rwa')
    os.close(_fd)
else:
    rwa_file = glob.glob(os.path.join(tempfile.gettempdir(), '.*_tutorial0\.rwa'))[0]
    trajectories_with_time_varying_properties = glob.glob(os.path.join(tempfile.gettempdir(), '.*_tutorial1\.txt'))[0]
    analyses_with_time_varying_properties = glob.glob(os.path.join(tempfile.gettempdir(), '.*_tutorial1\.rwa'))[0]

def _exists(f):
    return 0 < os.stat(f).st_size


def load_default_trajectories(time_varying_properties=False):
    """
    Load default trajectories.

    Trajectories are generated and stored for further calls in files trajectories.txt and trajectories2.txt.
    """
    if time_varying_properties:
        # transient potential energy sink
        if _exists(trajectories_with_time_varying_properties):
            trajs = load_xyt(trajectories_with_time_varying_properties)
        else:
            delta_V = 3. # maximum force amplitude in kB.T
            sink_space_center = np.array([.5, .5]) # x, y
            sink_time_center = 5. # t
            sink_radius = .2
            sink_half_duration = 4.
            s2r = sink_radius * sink_radius / (-log(.05)) # squared gaussian scale
            s2t = sink_half_duration * sink_half_duration / (-log(.05))
            def F(r, t):
                r, t = r - sink_space_center, t - sink_time_center
                grad_V = 2. * r / s2r * exp(-np.dot(r, r) / s2r)
                V_t = exp(-(t * t) / s2t)
                f = delta_V * (-grad_V) * V_t
                return f
            trajs = random_walk(
                diffusivity=.1, force=F,
                trajectory_mean_count=500, lifetime_tau=.25,
                reflect=True,# minor_step_count=999,
                )
            trajs.to_csv(trajectories_with_time_varying_properties, sep='\t', header=False)
        return trajs

    else:
        # linear potential energy gradient
        if _exists(trajectory_file):
            trajs = load_xyt(trajectory_file)
        else:
            trajs = random_walk_2d(
                n_trajs=400, N_mean=5, dt=.05,
                D0=.1, amplitude_D=0,
                amplitude_V=-2., mode_V='potential_linear',
                )
            trajs.to_csv(trajectory_file, sep='\t', header=False)
        return trajs


def load_default_partition(time_varying_properties=False):
    """
    Load a default partition.

    A segmentation and data partition are generated and stored for further calls in files
    trajectories.rwa and trajectories2.rwa.
    """
    if time_varying_properties:
        # transient sink
        if _exists(analyses_with_time_varying_properties):
            analysis_tree = load_rwa(analyses_with_time_varying_properties, lazy=True)
        else:
            analysis_tree = Analyses(load_default_trajectories(time_varying_properties=True))
        if 'default' in analysis_tree.labels:
            cells = analysis_tree['default'].data
        else:
            cells = tessellate(analysis_tree, 'hexagon',
                time_window_duration=2.,
                time_window_options=dict(time_dimension=True),
                knn=10,
                output_label='default')
            save_rwa(analyses_with_time_varying_properties, analysis_tree, force=True)
        return cells

    else:
        # linear gradient
        if _exists(rwa_file):
            analysis_tree = load_rwa(rwa_file, lazy=True)
        else:
            analysis_tree = Analyses(load_default_trajectories())
        if 'default' in analysis_tree.labels:
            cells = analysis_tree['default'].data
        else:
            cells = tessellate(analysis_tree, 'hexagon',
                lower_bound=np.array((-.6,-.6)), upper_bound=np.array((.6,.6)),
                output_label='default')
            save_rwa(rwa_file, analysis_tree, force=True)
        return cells


def load_default_maps(_and_partition=False):
    """
    Load default maps of inferred parameters, including 'diffusivity', 'force' and 'potential'.
    """
    if _exists(rwa_file):
        analysis_tree = load_rwa(rwa_file, lazy=True)
        if 'default' not in analysis_tree.labels:
            raise NotImplementedError('please run load_default_partition or delete the {} file'.format(rwa_file))
        if _and_partition:
            cells = analysis_tree['default'].data
    else:
        cells = load_default_partition()
        analysis_tree = Analyses(cells.points)
        analysis_tree.add(cells, label='default')
    default_subtree = analysis_tree['default']
    if 'default' not in default_subtree.labels:
        infer(analysis_tree, 'DV', localization_error=.01, input_label='default', output_label='default')
        save_rwa(rwa_file, analysis_tree, force=True)
    maps = default_subtree['default'].data
    if _and_partition:
        return maps, cells
    else:
        return maps


def load_default_tree(time_varying_properties=False):
    """
    Load a default analysis tree.

    Files trajectories.rwa or trajectories2.rwa are generated if necessary and loaded.
    """
    if time_varying_properties:
        _file = analyses_with_time_varying_properties
    else:
        _file = rwa_file
    if not _exists(_file):
        load_default_partition(time_varying_properties=time_varying_properties)
    return load_rwa(rwa_file, lazy=True)


__all__ = ['trajectory_file', 'rwa_file',
        'trajectories_with_time_varying_properties', 'analyses_with_time_varying_properties',
        'load_default_trajectories', 'load_default_partition', 'load_default_maps', 'load_default_tree']

