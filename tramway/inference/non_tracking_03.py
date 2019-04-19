from .base import *
import numpy as np
import pandas as pd
from time import time
from numpy import array, reshape, sum, zeros, ones, arange, dot, max, argmax, log, exp, sqrt, pi
from scipy import optimize as optim
from scipy.special import iv
from scipy.special import gamma as g
from scipy.special import gammaln as lng
from scipy.misc import logsumexp
from scipy.optimize import linear_sum_assignment as kuhn_munkres
from scipy.stats import skellam
from scipy.optimize import minimize
import inspect
import multiprocessing as mp
import sys
from matplotlib import pylab as plt

setup = {
    'infer': 'non_tracking_03',
    'sampling': 'group',
    'arguments': {'cell_index': {}, 'dt': {}, 'p_off': {}, 's2': {}, 'D_bounds': {}, 'D0': {}, 'method': {}, \
                  'tol': {}, 'times': {}}
}
''' The argument passing is very clumsy and needs to be better integrated with existing code.
'''


def non_tracking_03(cells, dt=0.04, p_off=0., mu_on=0., s2=0.0025, D0=0.2, method='None',
                    tol=1e-3, times=[0.]):
    if method == 'None':
        raise ValueError("method must be given")
    inferrer = NonTrackingInferrer(cells=cells,
                                   dt=dt,
                                   gamma=method['gamma'],
                                   smoothing_factor=method['smoothing_factor'],
                                   optimizer=method['optimizer'],
                                   tol=method['tol'],
                                   epsilon=method['epsilon'],
                                   maxiter=method['maxiter'],
                                   phantom_particles=method['phantom_particles'],
                                   messages_type=method['messages_type'],
                                   chemPot=method['chemPot'],
                                   chemPot_gamma=method['chemPot_gamma'],
                                   chemPot_mu=method['chemPot_mu'],
                                   scheme=method['scheme'],
                                   method=method['method'],
                                   distribution=method['distribution'],
                                   temperature=method['temperature'],
                                   parallel=method['parallel'],
                                   hij_init_takeLast=method['hij_init_takeLast'],
                                   p_off=p_off, mu_on=mu_on,
                                   starting_diffusivities=method['starting_diffusivities'],
                                   starting_drifts=method['starting_drifts'],
                                   inference_mode=method['inference_mode'],
                                   cells_to_infer=method['cells_to_infer'],
                                   minlnL=method['minlnL'],
                                   verbose=method['verbose'])
    inferrer.infer()
    return inferrer._final_diffusivities


''' Rough debug is done. Minor bugs may remain
    TODO: add drift [transform parameters array into pandas dataframe]. Store with cartesian coordinates or as direction and intensity ?
    v0.3 should solve Rac1 bug (empty cells)
    v0.3 should support massive parallelization at cluster level (separate SLURM jobs for each cell) using argument `cells_to_infer`
    v0.3 is syntactically closer to an eventual c++ version of the code
    The handling of particle count and drs distance matrices is not very pleasing yet
    IMPLEMENTING DRIFT
'''


# Forebear


class NonTrackingInferrer:

    def __init__(self, cells, dt, gamma=0.8, smoothing_factor=0, optimizer='NM', tol=1e-3, epsilon=1e-8, maxiter=10000,
                 phantom_particles=1, messages_type='CLV', chemPot='None', chemPot_gamma=1, chemPot_mu=1, scheme='1D',
                 method='BP', distribution='gaussian', temperature=1, parallel=1, hij_init_takeLast=False, p_off=0,
                 mu_on=0, starting_diffusivities=[1],
                 starting_drifts=[0, 0], inference_mode='D', cells_to_infer='all', minlnL=-100, verbose=1):
        """
        :param cells: An object of type 'Distributed' containing the data tessellated into cells
        :param gamma: The damping factor of the BP. h = gamma * h_old + (1-gamma) * h_new
        :param smoothing_factor: The coefficient used in the smoothing prior
        :param optimizer: The optimizer to use. Only Nelder-Mead 'NM' is supported
        :param tol: The tolerance for the optimizer. How precise an optimum do we require ?
        :param epsilon: The convergence tolerance for the BP. When two successive energies differ less than epsilon, we say that BP has converged
        :param maxiter: The maximal number of iterations to be done in the BP. If it is attained, BP stops and returns the last energy
        :param phantom_particles: (1) Phantom particles are introduced when necessary, or (0) No phantom particles are introduced
        :param messages_type: 'CLV': for classical messages
                              'JB' : for alternative update formulation
        :param chemPot: 'None': Don't use chemPot
                        'Chertkov': Use Chertkov chemPot
                        'Mezard': Use Mezard chemPot
        :param chemPot_gamma: The gamma parameter in the Mezard formulation of the Chemical Potential. Default is 1
        :param chemPot_mu: The mu parameter in the Chertkov formulation of the Chemical Potential. Default is 1
        :param scheme: The dimension-reducing optimization scheme. Can be '1D' or '2D'
        :param method: The marginalization method to use. Can be 'MPA' or 'BP'
        :param distribution: The likelihood-distribution to use. Can be 'gaussian' or 'rayleigh'
        :param temperature: The temperature of the BP (beta in the notes)
        :param parallel: Boolean. If `True`, we use the parallel implementation. If False, we use the sequential implementation
        :param hij_init_takeLast: NOT YET IMPLEMENTED Boolean. If True, then we initialize h_ij to its last value at the previous BP
        :param dt: The time step between frames
        :param p_off: The probability of a particle disappearing
        :param mu_on: The particle appearance intensity
        :param starting_diffusivities: The starting diffusivities serve only as an input variable. final_diffusivities is immediately initialized to starting_diffusivities
        :param starting_drifts: NOT YET IMPLEMENTED. The starting drifts serve only as an input variable. final_drifts is immediately initialized to starting_drifts.
        :param inference_mode: 'D' : Infer only diffusivity
                               'DD': Infer diffusivity and drift. NOT SUPPORTED YET
        :param working_diffusivities: The working diffusivities contain the values used within the optimization. In particular it is used to store D_out in the 2D optimization scheme
        :param final_diffusivities: NOT A PARAMETER The final diffusivities are the most up-to-date. Only to be changed at the end of an estimation
        :param final_drifts: NOT YET IMPLEMENTED. The final drifts are the most up-to-date. Only to be changed at the end of an estimation
        :param cells_to_infer: An array of indices on which we want to infer
        :param minlnL: The cut-off threshold for small probabilities
        :param verbose: Level of verbosity.
                0 : mute (not advised)
                1 : introverted (advised)
                2 : extroverted
                3 : Don't stop talking !!
        """
        # Idea to reduce the number of parameters: Put diffusivity and drift together in a pandas dataframe
        # Problem data
        self._cells = cells
        self._dt = dt
        self._p_off = p_off
        self._mu_on = mu_on
        if cells_to_infer == 'all':
            self._cells_to_infer = list(cells.keys())
        else:
            self._cells_to_infer = cells_to_infer
        # self._particle_count = self.particle_count()

        # Algorithm tuning parameters (numerical values)
        self._gamma = gamma
        self._smoothing_factor = smoothing_factor
        self._tol = tol
        self._maxiter = maxiter
        self._temperature = temperature
        self._chemPot_gamma = chemPot_gamma
        self._epsilon = epsilon
        self._chemPot_mu = chemPot_mu
        self._minlnL = minlnL
        if len(starting_diffusivities) == 1:
            self._starting_diffusivities = starting_diffusivities[0] * np.ones(len(cells.keys()))
        else:
            self._starting_diffusivities = starting_diffusivities

        if len(starting_drifts) == 2:
            self._starting_drifts = np.tile(starting_drifts, (len(cells.keys()), 1))
        else:
            self._starting_drifts = starting_drifts

        # Methods to use (discrete values)
        self._optimizer = optimizer
        self._scheme = scheme
        self._method = method
        self._parallel = parallel
        self._hij_init_takeLast = hij_init_takeLast
        self._phantom_particles = phantom_particles
        self._chemPot = chemPot
        self._messages_type = messages_type
        self._inference_mode = inference_mode
        self._distribution = distribution

        # Others
        self._final_diffusivities = self._starting_diffusivities
        self._final_drifts = self._starting_drifts  # NOT YET IMPLEMENTED.
        self._verbose = verbose

        self.check_parameters()

    def check_parameters(self):
        if self._method == 'MPA':
            assert (self._phantom_particles is True)
            assert (self._chemPot == 'None')
        if self._chemPot == 'None':
            assert (self._phantom_particles is True)
        if self._distribution != 'gaussian' and self._distribution != 'rayleigh':
            raise ValueError("distribution not supported")

    def confirm_parameters(self):
        """
            prints a summary of the parameters of the inference. This can be useful when we store computed values on files. The log file then contains the information about the algorithm parameters.
        """
        self.vprint(1,
                    f"Starting inference with methods: \n\tmethod={self._method} \n\tscheme={self._scheme} \n\toptimizer={self._optimizer} \n\tdistribution={self._distribution} \n\tparallel={self._parallel} \n\thij_init_takeLast={self._hij_init_takeLast} \n\tchemPot={self._chemPot} \n\tmessages_type={self._messages_type} \n\tinference_mode={self._inference_mode} \n\tphantom_particles={self._phantom_particles}")
        self.vprint(1,
                    f"The tuning is: \n\tgamma={self._gamma} \n\tsmoothing_factor={self._smoothing_factor} \n\ttol={self._tol} \n\tmaxiter={self._maxiter} \n\ttemperature={self._temperature} \n\tchemPot_gamma={self._chemPot_gamma} \n\tchemPot_mu={self._chemPot_mu} \n\tepsilon={self._epsilon} \n\tminlnL={self._minlnL}")
        self.vprint(1, f"Inference will be done on cells {self._cells_to_infer}")
        self.vprint(1, f"p_off is {self._p_off},\tmu_on={self._mu_on}")

    '''
    # TODO : check if this function is really necessary at this level. It gets overridden by the child
    def particle_count(self):
        """
            Used for __init__
        :return: The matrix of particle numbers in the cells first wrt cells, the wrt times
        """

        # Step 1: Compute the times
        r_total = []
        t_total = []
        S_total = 0
        index = list(self._cells.keys())
        for j in index:
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            S = cell.volume
            r_total.extend(r)
            t_total.extend(t)
            S_total += S
        r_total = np.array(r_total)
        t_total = np.array(t_total)
        # List of times corresponding to frames:
        times = np.arange(min(t_total), max(t_total + self._dt / 100.), self._dt)

        # Step 2: Compute the particle number
        particle_number = np.zeros( (len(index),len(times)) )
        print(f"index={index}")
        for j in index:
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            frames_j = self.rt_to_frames(r, t, times)
            N_j = [len(frame) for frame in frames_j]
            particle_number[j,:] = N_j

        return particle_number
    '''

    def infer(self):
        """
            Does the inference in the self._cells_to_infer and stores the result in self._final_diffusivities
        """
        self.confirm_parameters()
        # PARALLEL VERSION with mp #
        if self._parallel:
            n_cores = mp.cpu_count()
            pool = mp.Pool(n_cores)
            self.vprint(1, f"Using {n_cores} cpus")
            result_objects = [pool.apply_async(self.estimate, args=(i,)) for i in
                              self._cells_to_infer]  # How to handle this with drift ?
            pool.close()
            pool.join()
            results = [r.get() for r in result_objects]
            self._final_diffusivities = pd.DataFrame(results, index=list(self._cells_to_infer.keys()), columns=['D'])
            # Check : Is index correct ?

        # SEQUENTIAL VERSION #
        elif not self._parallel:
            """ In the sequential method for the 1D scheme, the inferred value in cell i is assumed as the diffusivity of cell i 
                when inferring cells j>i. In the parallel method, these optimizations are independent.
            """
            for i in self._cells_to_infer:
                self._final_diffusivities[i] = self.estimate(i)
                self.vprint(2, f"Current parameters with tol = {self._tol}")
                self.vprint(2, f"parameters={self._final_diffusivities}")
            self._final_diffusivities = pd.DataFrame(self._final_diffusivities, index=list(self._cells.keys()),
                                                     columns=['D'])
            # Check : Is index correct ?
        else:
            raise ValueError("Invalid argument: parallel")
        self.vprint(1, f"Final diffusivities are:\n {self._final_diffusivities}")

    def estimate(self, i):
        """
            Does a local inference of properties in cell of index i
        :param i: The index of the cell to estimate
        :return: The estimated diffusivity
        """
        self.vprint(2, f"\nCell no.: {i}")

        region_indices = self._cells.neighbours(
            i)  # We take only first order neighbourhood to make a region. Can be generalized here !
        region_indices = np.insert(region_indices, 0, i)  # insert i in the 0th position

        # --- Fit: ---
        fittime = time()
        if self._scheme == '1D':
            D = self._starting_diffusivities[i]
        elif self._scheme == '2D':
            D = array([self._starting_diffusivities[i], self._starting_diffusivities[i]])
        else:
            raise ValueError("This scheme is not supported. Choose '1D' or '2D'.")

        # Select the BP version
        """ On this point we create a specialized copy of self.
            Its attributes have the same values as self.
            There surely is a simpler way of passing the parent attributes to the child, but I did not find it. Sorry :/
        """
        parent_attributes = (
            self._cells, self._dt, self._gamma, self._smoothing_factor, self._optimizer, self._tol, self._epsilon,
            self._maxiter, self._phantom_particles, self._messages_type, self._chemPot, self._chemPot_gamma,
            self._chemPot_mu, self._scheme, self._method, self._distribution, self._temperature, self._parallel,
            self._hij_init_takeLast, self._p_off, self._mu_on, self._starting_diffusivities, self._starting_drifts,
            self._inference_mode, self._cells_to_infer, self._minlnL, self._verbose)
        if self._chemPot == 'Chertkov':
            local_inferrer = NonTrackingInferrerRegionBPchemPotChertkov(parent_attributes, i, region_indices)
        elif self._chemPot == 'Mezard':
            self.vprint(2,"NonTrackingInferrerRegionBPchemPotMezard created")
            local_inferrer = NonTrackingInferrerRegionBPchemPotMezard(parent_attributes, i, region_indices)
        elif self._chemPot == 'None' and self._messages_type == 'CLV':
            local_inferrer = NonTrackingInferrerRegion(parent_attributes, i, region_indices)
        elif self._chemPot == 'None' and self._messages_type == 'JB':
            local_inferrer = NonTrackingInferrerRegionBPalternativeUpdateJB(parent_attributes, i, region_indices)
        else:
            raise ValueError(
                "chemPot or messages_type not valid. Suggestion: Select chemPot='None' and messages_type='CLV'.")

        if self._optimizer == 'NM':
            fit = minimize(local_inferrer.smoothed_posterior, x0=D, method='Nelder-Mead', tol=self._tol,
                           options={'disp': True})
        else:
            raise ValueError("This optimizer is not supported. Suggestion: Choose 'NM'. ")
        D_in = abs((fit.x)[0])
        ''' Remark: Nelder-Mead does not have positivity constraints for D_in, so we might find a negative value.
            'smoothedPosterior' is implemented to be symmetric, so we should still find the optimum.
        '''

        self.vprint(1, f"fit = {fit}")
        # D_i_ub=max([1.5*(D_i-s2/dt),0.])
        # print("D corrected for motion blur and localization error:", D_i_ub)
        self.vprint(1, f"\nCell no.: {i}")
        self.vprint(1, f"{time() - fittime}s")  # make this an attribute to save it in a file later ?
        self.vprint(1, f"Estimation: {D_in}")

        return D_in

    def get_neighbours(self, cell_index, order=1):
        """ TO BE TESTED
            Computes the indices of the neighbours of cell cell_index. Neighbours of order 1 are just neighbours. Neighbours of order 2 also include neighbours of neighbours. Neighbours of order 3 also include neighbours of neighbours of neighbours, and so on...
        :param cell_index: The index of the central cell
        :param order: The order of the neighbourhood
        :return: An array or list (not sure) of the indices of the neighbour cells
        """
        if order == 1:
            return self._cells.neighbours(cell_index)
        else:
            neighbours = []
            for n in self._cells.neighbours(cell_index):
                neighbours.append(self.get_neighbours(n, order=order - 1))
            neighbours = list(dict.fromkeys(neighbours))  # remove duplicates
            neighbours.remove(cell_index)
            return neighbours

    # -----------------------------------------------------------------------------
    # Generate stack of frames
    # -----------------------------------------------------------------------------
    def rt_to_frames(self, r, t, times):
        """
        :param r: The list of positions
        :param t: The list of times
        :param times: The times array, from minimal to maximal time with adequate step
        :return: The list of frames corresponding to times
        """
        frames = []
        for t_ in times:
            frames.append(np.array(r[(t < t_ + self._dt / 100.) & (t > t_ - self._dt / 100.)]))
        return frames

    def local_distance_matrices(self, region_indices):
        """
        :param region_indices: The indices of the cells in the region to consider
        :return: Should return a list. Element number i contains 2 matrices: The first is the position difference between particles in frame i+1 and i in the x-axis. The second matrix is the same, but on the y-axis.
        """
        r_total = []
        t_total = []
        S_total = 0
        for j in region_indices:
            cell = self._cells[j]
            r = cell.r  # the recorded positions in cell
            t = cell.t  # the recorded times in cell
            S = cell.volume
            r_total.extend(r)
            t_total.extend(t)
            S_total += S
        r_total = np.array(r_total)  # r_total contains all the position in the region
        t_total = np.array(t_total)  # t_total contains the corresponding times in the same order
        # List of times corresponding to frames:
        times = np.arange(min(t_total), max(t_total + self._dt / 100.), self._dt)
        # Build lists of frames corresponding to times. A frame is an array of positions:
        frames = self.rt_to_frames(r_total, t_total, times)
        drs = [self.positions_to_distance_matrix(frames[n], frames[n + 1]) for n in range(len(frames) - 1)]
        return drs

    # -----------------------------------------------------------------------------
    # Helper functions for tracking/non-tracking
    # -----------------------------------------------------------------------------
    def positions_to_distance_matrix(self, frame1, frame2):
        '''Matrix of position differences between all measured points in two succesive frames.'''
        N_ = len(frame1)
        M_ = len(frame2)
        dr_ij = np.zeros([2, N_, M_])
        for i in range(N_):
            dr_ij[0, i] = frame2[:, 0] - frame1[i, 0]
            dr_ij[1, i] = frame2[:, 1] - frame1[i, 1]
        return dr_ij

    def p_m(self, x, Delta, N):
        '''Poissonian probability for x particles to appear between frames given Delta=M-N
        and blinking parameters mu_on and mu_off. Consult the article for reference'''
        mu_off = N * self._p_off
        return (self._mu_on * mu_off) ** (x - Delta / 2.) / (
                g(x + 1.) * g(x + 1. - Delta) * iv(Delta, 2 * sqrt(self._mu_on * mu_off)))

    def vprint(self, level, string, end_='\n'):
        if self._verbose == 0:
            pass
        elif self._verbose >= level:
            print(string, end=end_)
        else:
            pass


""" If we are to store region-dependent variables as attributes, for the sake of easy access, we have to do so in a separate class, because parallelism would otherwise cause interference between regions.
Hence the working parameters need to be in the child class
Within here, we cannot change the attributes of the calling motherclass
Within here, we consider a fixed diffusivity
"""


class NonTrackingInferrerRegion(NonTrackingInferrer):

    def __init__(self, parentAttributes, index_to_infer, region_indices):
        super(NonTrackingInferrerRegion, self).__init__(
            *parentAttributes)  # need to get all attribute values from parent class
        self._index_to_infer = index_to_infer
        self._region_indices = region_indices
        assert (self._index_to_infer == self._region_indices[0])
        # self._diffusivityMatrix = self.buildDiffusivityMatrix() # maybe not needed yet
        # print(f"self._gamma : {self._gamma}")
        self._drs = self.local_distance_matrices(region_indices)

        area = 0
        for i in self._region_indices:
            cell = self._cells[i]
            area += cell.volume
        self._region_area = area
        self._particle_count = self.particle_count()
        self._stored_hij = [None] * self._particle_count.shape[1]  # one element for each frame

    def particle_count(self):
        """
            Used for __init__ of nonTrackingInferrerRegion
        :return: The matrix of particle numbers in the cells. First w.r.t cells, then w.r.t times
        """

        # Step 1: Compute the times
        r_total = []
        t_total = []
        S_total = 0
        index = list(self._cells.keys())
        for j in self._region_indices:
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            S = cell.volume
            r_total.extend(r)
            t_total.extend(t)
            S_total += S
        r_total = np.array(r_total)
        t_total = np.array(t_total)
        # List of times corresponding to frames:
        times = np.arange(min(t_total), max(t_total + self._dt / 100.), self._dt)

        # Step 2: Compute the particle number
        particle_number = np.zeros((len(index), len(times)))
        print(f"region_indices={self._region_indices}")
        for j in self._region_indices:
            cell = self._cells[j]
            r = cell.r
            t = cell.t
            frames_j = self.rt_to_frames(r, t, times)
            N_j = [len(frame) for frame in frames_j]
            particle_number[j, :] = N_j

        return particle_number

    def lnp_ij(self, dr, frame_index):
        """
            Computes the log-probability matrix of displacements
        :param dr: the displacement vectors
        :param frame_index: The index of the current frame
        :return: Matrix of log-probabilities for each distance.
        """
        D = self.build_diffusivity_matrix(frame_index)
        try:
            if self._distribution=='gaussian':
                lnp = -log(4 * pi * D * self._dt) - (dr[0] ** 2 + dr[1] ** 2) / (4. * D * self._dt)
            elif self._distribution=='rayleigh':
                r2 = dr[0] ** 2 + dr[1] ** 2
                lnp = log(sqrt(r2)) - log(2. * D * self._dt) - r2 / (4. * D * self._dt)
            else:
                raise ValueError("distribution not supported. Suggestion: Use 'gaussian'. ")
        except ZeroDivisionError:
            raise ValueError("The optimizer somehow tested 0 diffusivity --> ZeroDivisionError")
        lnp[lnp < self._minlnL] = self._minlnL  # avoid numerical underflow
        return lnp

    def Q_ij(self, n_off, n_on, dr, frame_index):
        """
        Matrix of log-probabilities for assignments with appearances and disappearances.
        Structure of Q_ij:
         _                        _
        | Q_11  Q_12 .. Q_1M | 1/S |
        | Q_21  Q_22 .. Q_2M | 1/S |
        |  :     :       :   |  :  |
        | Q_N1  Q_N2 .. Q_NM | 1/S |
        |--------------------------|
        | 1/S   1/S  .. 1/S  |  0  |
        |_1/S   1/S  .. 1/S  |  0 _|
        :param n_off: The number of disappearing particles
        :param n_on:  The number of appearing particles
        :param dr:  The displacement vectors
        :param frame_index: The index of the current frame
        :return: The square matrix Q_ij of size N + n_on - n_off
        """
        n_on = int(n_on)
        n_off = int(n_off)
        N = np.sum(self._particle_count[self._region_indices, frame_index])
        N = int(N)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1])
        M = int(M)
        if self._phantom_particles:
            assert (M == N - n_off + n_on)  # just a consistency check
        out = zeros([n_on + N, n_off + M])
        LM = ones([n_on + N, n_off + M]) * self._region_area
        lnp = self.lnp_ij(dr, frame_index)
        out[:N, :M] = lnp - log(LM[:N, :M])
        out[:N, M:] = -log(LM[:N, M:])
        out[N:, :M] = -log(LM[N:, :M])
        out[N:, M:] = 2. * self._minlnL  # Note: Why not 0 ??
        return out

    def build_diffusivity_matrix(self, frame_index):
        """
            For internal use only. Needs to be processed into a likelihood matrix and completed with phantom particles
        :param frame_index: the index of the first frame
        :return: An N x M matrix containing in each line i the diffusivities of particle i.
        """
        M = int(np.sum(self._particle_count[self._region_indices, frame_index + 1]))
        N = int(np.sum(self._particle_count[self._region_indices, frame_index]))

        Ds = np.zeros([N, M])
        N_ix = np.cumsum(self._particle_count[self._region_indices, frame_index]).astype(int)
        N_ix = np.insert(N_ix, 0, 0)  # insert a 0 at the start for the technical purpose of the for loop
        for j in range(0, len(self._region_indices)):
            Ds[N_ix[j]:N_ix[j + 1]] = ones(Ds[N_ix[j]:N_ix[j + 1]].shape) * self._working_diffusivities[
                self._region_indices[j]]

        return Ds

    # -----------------------------------------------------------------------------
    # smoothing functions
    # -----------------------------------------------------------------------------
    def smoothing_prior(self):
        """
            Computes the smoothing prior over all cells to infer
            Note: This is done only for the diffusivity, not for drift.
            Drift is not regularized.
        :return: the log of the diffusion smoothing prior
        """
        # print("call: smoothing prior")
        penalization = 0
        index = self._cells_to_infer
        for i in index:
            gradD = self._cells.grad(i, self._working_diffusivities)
            penalization += self._cells.grad_sum(i, gradD * gradD)
        return self._smoothing_factor * penalization

    def smoothing_prior_heuristic(self):
        # TODO
        raise ("Not supported yet")

    def smoothed_posterior(self, D):
        D = abs(D)
        self.vprint(3, D, end_='')
        ''' Nelder-Mead does not have positivity constraints for D_in, so we might find a negative value.
            'smoothedPosterior' is implemented to be symmetric, so that we should still find the optimum.'''
        self._working_diffusivities = self._final_diffusivities  # the most up-to-date parameters
        # Note: Perhaps we should check for a value 0 of the diffusivity. It could cause problems later
        # scheme selector
        if self._scheme == '1D':
            self._working_diffusivities[self._index_to_infer] = D
        elif self._scheme == '2D':
            self._working_diffusivities[self._region_indices] = D[1]
            self._working_diffusivities[self._index_to_infer] = D[
                0]  # this overwrites the wrongly assigned value in previous line
        else:
            raise ValueError(f"scheme {self._scheme} not supported")

        # method selector
        if self._method == 'MPA':
            mlnL = self.MPA_minusLogLikelihood_multiFrame()
        elif self._method == 'BP':
            mlnL = self.marginal_minusLogLikelihood_multiFrame()
        else:
            raise ValueError(f"Invalid method {self._method}. Choose either 'BP' or 'MPA'. ")

        # smoothing selector
        if self._smoothing_factor == 0:
            posterior = mlnL
        else:
            posterior = mlnL + self.smoothing_prior()
        self.vprint(3, f"{posterior}")
        return posterior

    # -----------------------------------------------------------------------------
    # Sum-product BP
    # -----------------------------------------------------------------------------
    def initialize_messages(self, N, M):
        """
            Initializes the BP messages to zeros
        :param N: particles number first frame
        :param M: particles number second frame
        :return: Two matrices, of the same size N x M containing only zeros
        """
        hij = zeros([N, M])
        hji = zeros([N, M])
        return hij, hji

    def boolean_matrix(self, N):
        """
            Boolean matrix for parallel implementation of BP.
        :param N: Size of the matrix
        :return: A square matrix of size N x N with 0 in the diagonal and 1 everywhere else
        """
        bool_array = ones([N, N])
        for i in range(N):
            bool_array[i, i] = 0.
        return bool_array

    def bethe_free_energy(self, hij, hji, Q):
        """
            Bethe free energy given BP messages hij and hji and log-probabilities Q.
            This is the "standard" form of the energy
        :param hij: Left side messages
        :param hji: Right side messages
        :param Q: matrix of log-probabilities
        :return: Bethe free energy approximation
        """
        # Naive version
        '''
        naive_energy = + sum(log(1. + exp(Q + hij + hji))) \
                       - sum(log(sum(exp(Q + hji), axis=1))) \
                       - sum(log(sum(exp(Q + hij), axis=0)))
        '''

        # logsumexp version
        logsumexp_energy = + sum(np.logaddexp(0,Q + self._temperature*hij + self._temperature*hji)) \
                           - sum(logsumexp(Q + self._temperature*hji, axis=1)) \
                           - sum(logsumexp(Q + self._temperature*hij, axis=0))

        return logsumexp_energy/self._temperature

    def sum_product_update_rule(self, Q, hij_old, hji_old):
        """
            The "standard" sum-product update rule
        :param Q: matrix of log-probabilities
        :param hij_old: Old left messages
        :param hji_old: Old right messages
        :return: The new messages (undamped)
        """
        hij_new = -log(dot(exp(Q + self._temperature*hji_old), self.boolean_matrix(Q.shape[1])))/self._temperature
        hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature*hij_old)))/self._temperature
        return hij_new, hji_new

    def sum_product_BP(self, Q, hij, hji):
        """
            Sum-product (BP) free energy (minus-log-likelihood) and BP messages.
        :param Q: The log-probability matrix
        :param hij: The left side messages
        :param hji: The right side messages
        :return: F_BP : The final Bethe free energy
                 hij  : The final left side message
                 hji  : The final right side message
                 n    : The number of iterations at convergence
        """
        hij_old = hij
        hji_old = hji
        F_BP_old = self.bethe_free_energy(hij_old, hji_old, Q)
        # Bethe = [F_BP_old]
        for n in range(self._maxiter):
            hij_new, hji_new = self.sum_product_update_rule(Q, hij, hji)
            hij = (1. - self._gamma) * hij_new + self._gamma * hij_old
            hji = (1. - self._gamma) * hji_new + self._gamma * hji_old
            # Stopping condition:
            F_BP = self.bethe_free_energy(hij, hji, Q)
            if abs(F_BP - F_BP_old) < self._epsilon:
                break
            # Update old values of energy and messages:
            F_BP_old = F_BP
            hij_old = hij
            hji_old = hji
            # Bethe.append(F_BP)
        # fig = plt.figure()
        # plt.plot(Bethe)
        # plt.show()
        return F_BP, hij, hji, n  # shall we store n somewhere and send it back to the user in some sort of diagnosis object ?

    def sum_product_energy(self, dr, frame_index, n_on, n_off, N, M):
        """
            Bethe free energy of given graph. The BP algorithm is run. F_BP is returned after convergence
        :param dr:
        :param frame_index:
        :param n_on: The number of appearing particles
        :param n_off: The number of disappearing particles
        :param N: The number of particles in the first frame (not including phantom particles)
        :param M: The number of particles in the second frame (not including phantom particles)
        :return: F_BP the Bethe free energy approximation
        """
        self.vprint(3, f"call: sum_product_energy")
        # If a single particle has been recorded in each frame and tracers are permanent, the link is known:
        if (N == 1) & (n_off == 0) & (n_on == 0):
            F_BP = -self.lnp_ij(dr, frame_index)
        # Else, perform BP to obtain the Bethe free energy:
        else:
            # bool_array = self.boolean_matrix(n_off + M)
            hij, hji = self.initialize_messages(n_on + N, n_off + M)
            assert (hij.shape == hji.shape)
            Q = self.Q_ij(n_off, n_on, dr, frame_index)
            F_BP, hij, hji, n = self.sum_product_BP(Q, hij, hji)
            # if self._hij_init_takeLast:
            #     TODO store hij
            #     self._stored_hij[frame_index,n_on] = hij
            #     raise ValueError("hij_init_takeLast not supported yet")
            if n == self._maxiter - 1:
                self.vprint(1, f"Warning: BP attained maxiter before converging.")
        # print(f"n={n}") # the number of iterations
        return F_BP

    def marginal_minusLogLikelihood(self, dr, frame_index):
        """
            Minus-log-likelihood marginalized over possible graphs and assignments using BP.
            in is the inside cell, out is the outside cells (array)
        :param dr: the displacement vectors
        :param frame_index: the index of the current frame
        :return: the minus log-likelihood between frame frame_index and frame frame_index+1
        """
        self.vprint(3, f"call: marginal_minusLogLikelihood, frame_index={frame_index}")
        self.vprint(3, ".", end_='')
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1].astype(int)
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index].astype(int)
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1]).astype(int)
        N = np.sum(self._particle_count[self._region_indices, frame_index]).astype(int)
        delta = M - N
        if (M_index_to_infer == 0) or (N_index_to_infer == 0):
            mlnL = 0.  # if the cell to infer is empty in one frame, we gain 0 information
        elif ((self._p_off == 0.) & (self._mu_on == 0.) & (delta == 0)) or self._phantom_particles is False:
            # Q = self.Q_ij(0, 0, dr, frame_index)
            mlnL = self.sum_product_energy(dr, frame_index, 0, 0, N, M)
        else:
            # self._stored_hij[frame_index] = [None] * (M - max([0, int(Delta)]))
            mlnL = -logsumexp([log(self.p_m(n_on, delta, N)) + lng(M + 1 - n_on) - lng(M + 1) - lng(N + 1) \
                               - self.sum_product_energy(dr, frame_index, n_on, n_on - delta, N, M) \
                               for n_on in range(max([0, int(delta)]), int(M))])  # Why use lng ?
        return mlnL

    def marginal_minusLogLikelihood_multiFrame(self):
        '''Minus-log-likelihood marginalized over possible graphs and assignments using BP.'''
        self.vprint(3, f"call: marginal_minusLogLikelihood_multiFrame")
        self.vprint(2, "*", end_='')
        mlnL = 0
        for frame_index, dr in enumerate(self._drs):
            # print(f"frame_index={frame_index} \t dr={dr}")
            mlnL += self.marginal_minusLogLikelihood(dr, frame_index)
        return mlnL

    # -----------------------------------------------------------------------------
    # Kuhn-Munkres MPA | ok
    # -----------------------------------------------------------------------------
    def MPA_energy(self, Q, mpa_row, mpa_col):
        return -sum(Q[mpa_row, mpa_col])

    def MPAscore(self, dr, frame_index):
        """
        :param dr: The distance matrix for frame_index
        :param frame_index: The current frame index
        :return: the MPAscore for frame_index
        """
        M_index_to_infer = self._particle_count[self._index_to_infer, frame_index + 1]
        N_index_to_infer = self._particle_count[self._index_to_infer, frame_index]
        M = np.sum(self._particle_count[self._region_indices, frame_index + 1])
        N = np.sum(self._particle_count[self._region_indices, frame_index])
        Delta = M - N
        if (self._p_off == 0.) & (self._mu_on == 0.) & (Delta == 0):
            Q = self.Q_ij(0, 0, dr, frame_index)
            mpa_row, mpa_col = kuhn_munkres(-Q)
            max_score = -self.MPA_energy(Q, mpa_row, mpa_col)
        elif (M_index_to_infer == 0) or (N_index_to_infer == 0):
            max_score = 0.  # if the cell to infer is empty in one frame, we gain 0 information
        else:
            scores_ = []
            for n_on in arange(max([0, Delta]), M):
                Q = self.Q_ij(n_on - Delta, n_on, dr, frame_index)
                mpa_row, mpa_col = kuhn_munkres(-Q)
                logp_m = log(self.p_m(n_on, Delta, N))
                E_MPA = self.MPA_energy(Q, mpa_row, mpa_col)
                score = logp_m - E_MPA + lng(M + 1 - n_on) - lng(M + 1) - lng(N + 1)  # Note: why are there lng terms ?
                scores_.append(score)
            max_score = max(scores_)
        return -max_score

    def MPA_minusLogLikelihood_multiFrame(self):
        """
            Computes the minus log-likelihood for all the frames
        :return: a float, the minus log-likelihood
        """
        mlnL = 0
        for frame_index, dr in enumerate(self._drs):
            mlnL += self.MPAscore(dr, frame_index)

        return mlnL


# -----------------------------------------------------------------------------
# Child classes for different methods ? MPA, BP, BPchemPotChertkov, BPchemPotMezard, BP_JB, maybe others
# -----------------------------------------------------------------------------


class NonTrackingInferrerRegionBPchemPotMezard(NonTrackingInferrerRegion):
    '''
    def __init__(self, parentAttributes, index_to_infer, region_indices):
        super(nonTrackingInferrerRegionBPchemPotMezard, self).__init__(parentAttributes, index_to_infer, region_indices)
    '''

    def sum_product_update_rule(self, Q, hij_old, hji_old):
        hij_new = -log(dot(exp(Q + self._temperature*hji_old), self.boolean_matrix(Q.shape[1])) \
                       + exp(-self._chemPot_gamma))/self._temperature
        hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature*hij_old)) \
                       + exp(-self._chemPot_gamma))/self._temperature
        return hij_new, hji_new

    def bethe_free_energy(self, hij, hji, Q):
        """
            Computes the Bethe free energy for the Mezard formulation
        :param hij: left-side messages
        :param hji: right-side messages
        :param Q: log-likelihood matrix
        :return: The Bethe free energy
        """
        #print("call bethe_free_energy, Mezard")
        #import pdb; pdb.set_trace()
        # We use logsumexp to avoid numerical underflow or overflow problems. The two versions do return different results and the logsumexp version gives realistic results on instances where the naive versions gives abusive results

        # Naive version
        '''
        term1 = + sum(log(1. + exp(Q + hij + hji)))
        term2 = - sum(log(sum(exp(Q + hji), axis=1) + exp(-self._chemPot_gamma)))
        term3 = - sum(log(sum(exp(Q + hij), axis=0) + exp(-self._chemPot_gamma)))
        term4 = - sum(Q.shape) * self._chemPot_gamma
        naive_energy = term1 + term2 + term3 + term4
        '''

        # logsumexp version, no constant term
        #gamma_matrix_ = ones(Q.shape)*self._chemPot_gamma
        term1_bis = + sum(np.logaddexp(0, Q + self._temperature*hij + self._temperature*hji))
        term2_bis = - sum(logsumexp(np.hstack((Q + self._temperature*hji, -ones((Q.shape[0], 1))*self._chemPot_gamma)), axis=1))
        term3_bis = - sum(logsumexp(np.vstack((Q + self._temperature*hij, -ones((1, Q.shape[1]))*self._chemPot_gamma)), axis=0))
        #term4_bis = - sum(Q.shape) * self._chemPot_gamma
        logsumexp_energy = term1_bis + term2_bis + term3_bis

        return logsumexp_energy/self._temperature


class NonTrackingInferrerRegionBPchemPotChertkov(NonTrackingInferrerRegion):

    def sum_product_update_rule(self, Q, hij_old, hji_old):
        hij_new = -log(dot(exp(Q + self._temperature*hji_old), self.boolean_matrix(Q.shape[1])) \
                       + exp(-self._chemPot_mu))/self._temperature
        hji_new = -log(dot(self.boolean_matrix(Q.shape[0]), exp(Q + self._temperature*hij_old)) \
                       + exp(-self._chemPot_mu))/self._temperature
        return hij_new, hji_new

    def bethe_free_energy(self, hij, hji, Q):
        """
            Computes the Bethe free energy for the Chertkov formulation
        :param hij: left-side messages
        :param hji: right-side messages
        :param Q: log-likelihood matrix
        :return: The Bethe free energy
        """
        # We use logsumexp to avoid numerical underflow or overflow problems. The two versions do return different results and the logsumexp version gives realistic results on instances where the naive versions gives abusive results
        # Naive version
        '''
        term1 = + sum(log(1. + exp(Q + hij + hji)))
        term2 = - sum(log(sum(exp(Q + hji), axis=1) + exp(-self._chemPot_mu)))
        term3 = - sum(log(sum(exp(Q + hij), axis=0) + exp(-self._chemPot_mu)))
        naive_energy = term1 + term2 + term3
        '''

        # logsumexp version, no constant term
        #gamma_matrix_ = ones(Q.shape)*self._chemPot_gamma
        term1_bis = + sum(np.logaddexp(0, Q + self._temperature*hij + self._temperature*hji))
        term2_bis = - sum(logsumexp(np.hstack((Q + self._temperature*hji, -ones((Q.shape[0], 1))*self._chemPot_mu)), axis=1))
        term3_bis = - sum(logsumexp(np.vstack((Q + self._temperature*hij, -ones((1, Q.shape[1]))*self._chemPot_mu)), axis=0))
        logsumexp_energy = term1_bis + term2_bis + term3_bis

        return logsumexp_energy/self._temperature


class NonTrackingInferrerRegionBPalternativeUpdateJB(NonTrackingInferrerRegion):

    def initialize_messages(self, N, M):
        """
            Initializes the BP messages to ones
        :param N: particles number first frame
        :param M: particles number second frame
        :return: Two matrices, of the same size N x M containing only ones
        """
        hij = ones([N, M])
        hji = ones([N, M])
        return hij, hji

    def sum_product_update_rule(self, Q, h_ij, alpha_i, alpha_j):
        """
            Perform one sum-product update in the space of hij-messages.
        :param Q: The matrix of log-probabilities log(pij)
        :param h_ij:
        :param alpha_i:
        :param alpha_j:
        :return:
        """
        Xi = np.square(dot(h_ij, np.ones(h_ij.shape) / 2) + dot(np.ones(h_ij.shape) / 2, h_ij) - h_ij)
        hij_new = exp(Q) * dot(exp(alpha_i), exp(alpha_j)) / (exp(Q) * dot(exp(alpha_i), exp(alpha_j)) + Xi)
        Ai_new = (1 - np.sum(hij_new ** 2, axis=1)) / dot(exp(Q), exp(alpha_j))
        Aj_new = (1 - np.sum(hij_new ** 2, axis=0)) / dot(exp(alpha_i), exp(Q))
        return hij_new, log(Ai_new), log(Aj_new)

    def free_energy(self, hij, Q, alpha_i, alpha_j):
        """
            Computes the free energy with Lagrangian relaxation of the normalization condition
        :param hij: the message matrix (equal to the belief that s_ij = 1)
        :param Q: The matrix of log-probabilities log(pij)
        :param alpha_i: Lagrangian multiplier for normalized columns
        :param alpha_j: Lagrangian multiplier for normalized rows
        :return:
        """
        return + sum(hij * (log(hij) - Q) - (1 - hij) * log(1 - hij)) \
               - sum(alpha_i * (sum(hij, axis=1) - 1)) \
               - sum(alpha_j * (sum(hij, axis=0) - 1))

    def sinkhorn_knopp_iteration(self, hij):
        """
            Performs one iteration of the Sinkhorn-Knopp algorithm: Sequentially normalizing rows and columns.
            The hope is that hij becomes 'more bistochastic' after this iteration
        :param hij: The message matrix
        :return: The transformed messages. Same shape as hij.
        """

        # Normalize the columns
        norm = np.sum(hij, axis=1)
        hij = hij / np.tile(norm, (hij.shape[1], 1)).T

        # Normalize the rows
        norm = np.sum(hij, axis=0)
        hij = hij / np.tile(norm, (hij.shape[0], 1))

        return hij

    def sum_product_BP(self, Q, hij, hji):
        """
            Sum-product (BP) free energy (minus-log-likelihood) and BP messages.
        :param Q: The matrix of log-probabilities log(pij)
        :param hij: The left side messages
        :param hji: The right side messages
        :return: F_BP : The final free energy
                 hij  : The final left side message
                 hji  : The final right side message (only kept for compatibility with mother code)
                 n    : The number of iterations at convergence
        """
        hij_old = hij
        alpha_i = np.ones(hij.shape[0])
        alpha_j = np.ones(hij.shape[1])
        F_BP_old = self.free_energy(hij_old, Q, alpha_i, alpha_j)
        for n in range(self._maxiter):
            hij_new, alpha_i, alpha_j = self.sum_product_update_rule(Q, hij, alpha_i, alpha_j)
            hij = (1. - self._gamma) * hij_new + self._gamma * hij_old
            hij = self.sinkhorn_knopp_iteration(hij)
            # Stopping condition:
            F_BP = self.free_energy(hij, Q, alpha_i, alpha_j)
            if abs(F_BP - F_BP_old) < self._epsilon:
                break
            # Update old values of energy and messages:
            F_BP_old = F_BP
            hij_old = hij
        return F_BP, hij, hji, n
