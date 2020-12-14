# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import *
from .abc import *
from tramway.core import *
import itertools
import numpy as np
import pandas as pd


class BaseTracker(AnalyzerNode):
    __slots__ = ()
    @property
    def spt_data(self):
        if not self._eldest_parent.spt_data.initialized:
            self._eldest_parent.spt_data.from_tracker()
        return self._eldest_parent.spt_data
    @property
    def frame_interval(self):
        return self.spt_data.frame_interval
    @frame_interval.setter
    def frame_interval(self, dt):
        self.spt_data.frame_interval = dt
    @property
    def dt(self):
        return self.spt_data.dt
    @dt.setter
    def dt(self, dt):
        self.spt_data.dt = dt
    @property
    def localization_error(self):
        return self.spt_data.localization_error
    @localization_error.setter
    def localization_error(self, err):
        self.spt_data.localization_error = err
    @property
    def localization_precision(self):
        return self.spt_data.localization_precision
    @localization_precision.setter
    def localization_precision(self, sigma):
        self.spt_data.localization_precision = sigma


class SingleParticleTracker(BaseTracker):
    __slots__ = ()
    def track(self, locations, register=False, source=None):
        if isinstance(locations, str):
            loc_file = locations
            locations = load_xyt(loc_file, columns=list('xyt'))
        else:
            loc_file = None

        trajectory_index = pd.DataFrame(np.full((len(locations),1), 1, dtype=int), columns=['n'])
        trajectories = trajectory_index.join(locations)

        if register:
            self.spt_data.add_tracked_data(trajectories, loc_file if source is None else source)

        return trajectories

Tracker.register(SingleParticleTracker)


def _tolist(a):
    a = a.tolist()
    if not isinstance(a, list):
        a = [ a ]
    return a

class NonTrackingTracker(BaseTracker):
    """ Non-tracking tracker.
    """
    __slots__ = ('_high_diffusivity','_large_length')
    def __init__(self, **kwargs):
        BaseTracker.__init__(self, **kwargs)
        self._high_diffusivity = None
        self._large_length = None
    @property
    def sigma(self):
        return self.localization_precision
    @property
    def estimated_high_diffusivity(self):
        return self._high_diffusivity
    @estimated_high_diffusivity.setter
    def estimated_high_diffusivity(self, D):
        self._high_diffusivity = D
    @property
    def estimated_large_length(self):
        if self._large_length is None and \
                self.estimated_high_diffusivity is not None and \
                self.dt is not None:
            self._large_length = 2.*np.sqrt(2.*self.estimated_high_diffusivity*self.dt)
        return self._large_length
    @estimated_large_length.setter
    def estimated_large_length(self, length):
        self._large_length = length
    ###
    def track(self, locations, register=False, source=None):
        import tramway.tracking.track_non_track.file_processing_loc as nt
        images = self._eldest_parent.images
        if isinstance(locations, str):
            loc_file = locations
            locations = load_xyt(loc_file, columns=list('xyt'))
        else:
            loc_file = None
            locations = locations[list('xyt')]

        movie_per_frame, n_unique = nt.convert_to_list(locations.values)
        dt_theo = self.dt
        t_init = self.dt * 1
        if images.initialized:
            t_end = self.dt * images.n_frames
        else:
            t_end = self.dt * n_unique

        sigma = self.localization_precision
        D_high = self.estimated_high_diffusivity
        length_high = self.estimated_large_length

        if sigma is None:
            raise AttributeError('attribute localization_precision is not set')
        if D_high is None:
            raise AttributeError('attribute estimated_high_diffusivity is not set')

        trajectories = {}

        current_trajectory_index = np.uint32(1)
        currently_assigned = set()

        for frame_index in range(n_unique-1):

            C = nt.get_cost_function(frame_index, movie_per_frame)

            try:
                C_eff,_,_,_,row_eff,col_eff,M,N,n_row_eff,n_col_eff,anomaly = \
                    nt.correct_cost_function(C, length_high)
            except IndexError:
                #print(C, length_high)
                currently_assigned = set()
                continue

            _, row, col = \
                    nt.get_assigment_matrix_from_reduced_cost(C_eff, row_eff, col_eff,
                        M, N, n_col_eff, n_row_eff, anomaly)

            assert isinstance(row, np.ndarray)
            if row.size == 0:
                currently_assigned = set()
                continue

            if currently_assigned:
                source_ = movie_per_frame[frame_index]
                destination = movie_per_frame[frame_index+1]
                row_to_col = np.full(len(source_), -1, dtype=int)
                row_to_col[row] = col
                if row.size == 1:
                    row, col = _tolist(row), _tolist(col)
                newly_assigned = set(row)
                new_assignment = np.zeros(len(destination), dtype=np.uint32)
                growing_trajectory = currently_assigned & newly_assigned
                for i in growing_trajectory:
                    traj_index = current_assignment[i]
                    assert 0<traj_index
                    j = row_to_col[i]
                    trajectories[traj_index].append(destination[[j]])
                    new_assignment[j] = traj_index
                for i in newly_assigned - growing_trajectory:
                    traj_index = current_trajectory_index
                    current_trajectory_index += 1
                    j = row_to_col[i]
                    trajectories[traj_index] = [ source_[[i]], destination[[j]] ]
                    new_assignment[j] = traj_index
                current_assignment = new_assignment

            else:
                if row.size == 1:
                    row, col = _tolist(row), _tolist(col)
                source_ = movie_per_frame[frame_index][row]
                destination = movie_per_frame[frame_index+1][col]
                n = current_trajectory_index
                new_trajectory_indices = np.arange(n, n+len(col), dtype=np.uint32)
                current_trajectory_index += len(col)
                current_assignment = np.zeros(len(movie_per_frame[frame_index+1]), dtype=np.uint32)
                current_assignment[col] = new_trajectory_indices
                for k in np.arange(len(col), dtype=np.uint32):
                    trajectories[new_trajectory_indices[k]] = [ source_[[k]], destination[[k]] ]

            try:
                currently_assigned = set(col)
            except TypeError:
                print(col)

        trajectory_indices, trajectory_coordinates = [], []
        for n in range(1, current_trajectory_index):
            traj = trajectories[np.uint32(n)]
            trajectory_indices.append(np.full((sum([len(t) for t in traj]),1), n))
            trajectory_coordinates.append(traj)
        trajectory_indices = np.vstack(trajectory_indices)
        trajectory_coordinates = np.vstack(list(itertools.chain(*trajectory_coordinates)))
        trajectories = pd.DataFrame(trajectory_indices, columns=['n']).join(
                pd.DataFrame(trajectory_coordinates, columns=list('xyt')))

        if register:
            self.spt_data.add_tracked_data(trajectories, filepath=loc_file if source is None else source)

        return trajectories

Tracker.register(NonTrackingTracker)


class TrackerInitializer(Initializer):
    """ Initializer class for the :class:`~tramway.analyzer.RWAnalyzer`
    :attr:`~tramway.analyzer.RWAnalyzer.tracker` main attribute.

    The :attr:`~tramway.analyzer.RWAnalyzer.tracker` attribute self-modifies
    on calling any of the *from_...* methods.
    """
    __slots__ = ()
    def from_single_particle(self):
        """ Considers every single molecule localization datablocks
        as single trajectories. 

        See also :class:`SingleParticleTracker`."""
        self.specialize( SingleParticleTracker )
    def from_non_tracking(self):
        """ *Non-tracking* tracker.
        
        See also :class:`NonTrackingTracker`."""
        self.specialize( NonTrackingTracker )


__all__ = [ 'Tracker', 'TrackerInitializer', 'SingleParticleTracker', 'NonTrackingTracker' ]

