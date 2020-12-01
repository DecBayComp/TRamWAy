# -*- coding: utf-8 -*-

# Copyright © 2017-2019, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
from tramway.core.xyt import crop
import warnings


def random_walk(diffusivity=None, force=None, viscosity=None, drift=None,
        trajectory_mean_count=100, trajectory_count_sd=0, turnover=None,
        initial_trajectory_count=None, new_trajectory_count=None,
        lifetime_tau=None, lifetime=None, single=False,
        box=(0., 0., 1., 1.), duration=10., time_step=.05, minor_step_count=99,
        reflect=False, full=False, count_outside_trajectories=None):
    r"""
    Generate random walks.

    Consider also the alternative generator
    :func:`~tramway.helper.simulation.categoricaltrap.random_walk_2d`.

    The `trajectory_mean_count` and `trajectory_count_sd` arguments are not compatible with
    the `initial_trajectory_count` and `new_trajectory_count` arguments.
    Use either pair of arguments.

    Arguments:

        diffusivity (callable or float): if callable, takes a coordinate vector
            (:class:`~numpy.ndarray`) and time (float),
            and returns the local diffusivity (float) in :math:`\mu m^2.s^{-1}`

        force (callable): takes a coordinate vector (:class:`~numpy.ndarray`) and time (float)
            and returns the local force vector (:class:`~numpy.ndarray`)

        viscosity (callable or float): if callable, takes a coordinate vector
            (:class:`~numpy.ndarray`) and time (float),
            and returns the local viscosity (float) that divides the local force;
            the default viscosity ensures equilibrium conditions

        drift (callable): takes a coordinate vector (:class:`~numpy.ndarray`) and time (float)
            if `force` is not defined,
            or the same two arguments plus the diffusivity (float) and force
            (:class:`~numpy.ndarray`) otherwise, and returns the local drift;
            this lets `viscosity` unevaluated

        trajectory_mean_count (int or float): average number of active trajectories at any time

        trajectory_count_sd (int or float): standard deviation of the number of active
            trajectories

        turnover (float): fraction of trajectories that will end at each time step;
            **deprecated**

        initial_trajectory_count (int): initial number of active trajectories

        new_trajectory_count (int or callable): number of newly generated trajectories;
            if `new_trajectory_count` is ``callable``, then it takes the time of a step as input
            and returns the count for this step as an ``int``

        lifetime_tau (float): trajectory lifetime constant in seconds

        lifetime (callable or float): trajectory lifetime in seconds;
            if `lifetime` is a ``float``, then it acts like `lifetime_tau`;
            if `lifetime` is ``callable``, then it takes as input the initial time of
            the trajectory

        single (bool): allow single-point trajectories

        box (array-like): origin and size of the space bounding box

        duration (float): duration of the simulation in seconds

        time_step (float): duration between two consecutive observations

        minor_step_count (int): number of intermediate unobserved steps

        reflect (bool): reflect the minor steps that would otherwise leave the bounding box

        full (bool): include locations that are outside the bounding box and times that are
            posterior to `duration`

        count_outside_trajectories (bool): include trajectories that have left the bounding box
            in determining the number of trajectories at each observation step;
            **deprecated**

    Returns:

        pandas.DataFrame: simulated trajectories with 'n' the trajectory index,
            't' the time and with other columns for location coordinates

    """
    if turnover is not None:
        warnings.warn('`turnover` is deprecated', DeprecationWarning)
    if count_outside_trajectories is False:
        warnings.warn('`count_outside_trajectories` is deprecated', DeprecationWarning)
    _box = np.asarray(box)
    dim = int(_box.size / 2)
    support_lower_bound = _box[:dim]
    support_size = _box[dim:]
    support_upper_bound = support_lower_bound + support_size
    # default diffusivity and drift maps
    #def null_scalar_map(xy, t):
    #    return 0.
    #def null_vector_map(xy, t):
    #    return np.zeros((dim,))
    if callable(diffusivity):
        pass
    elif np.isscalar(diffusivity) and 0 < diffusivity:
        _D = diffusivity
        diffusivity = lambda x, t: _D
    else:
        raise ValueError('`diffusivity` must be callable or a positive float')
    if force and not callable(force):
        raise TypeError('`force` must be callable')
    if drift:
        _drift = drift
        if force:
            drift = lambda x, t, D: _drift(x, t, D, force(x, t))
        else:
            drift = lambda x, t, D: _drift(x, t)
    elif force:
        if viscosity is None:
            drift = lambda x, t, D: D * force(x, t)
        elif callable(viscosity):
            drift = lambda x, t, D: force(x, t) / viscosity(x, t)
        elif np.isscalar(viscosity) and 0 < viscosity:
            drift = lambda x, t, D: force(x, t) / viscosity
        else:
            raise ValueError('`viscosity` must be callable or a positive float')
    else:
        drift = lambda x, t, D: 0
    #
    N = int(round(float(duration) / time_step)) # number of observed steps
    min_step_count = 1 if single else 2 # minimum number of observed steps per trajectory
    if callable(lifetime):
        lifetime_tau = None
    elif lifetime:
        lifetime_tau, lifetime = lifetime, None
    if lifetime_tau:
        trajectory_count_sd = None
    elif trajectory_count_sd:
        lifetime_tau = float(trajectory_count_sd) * time_step
    else:
        lifetime_tau = 4. * time_step
    # check
    if lifetime:
        assert callable(lifetime)
    else:
        assert bool(lifetime_tau)
    # number of trajectories at each time step
    K = None
    if new_trajectory_count is None:
        if trajectory_count_sd:
            K = np.rint(np.random.randn(N) * trajectory_count_sd + trajectory_mean_count).astype(int)
        else:
            K = np.full(N, trajectory_mean_count, dtype=int)
    elif initial_trajectory_count is None:
        warnings.warn('in combination with `new_trajectory_count`, use `initial_trajectory_count` instead of `trajectory_mean_count`')
        initial_trajectory_count = trajectory_mean_count
    k = np.zeros(N, dtype=int)
    # starting time and duration of the trajectories
    time_support = []
    t = 0.
    for i in range(N):
        t += time_step
        if K is None:
            if i == 0:
                k_new = initial_trajectory_count
            elif callable(new_trajectory_count):
                k_new = new_trajectory_count(t)
            else:
                k_new = new_trajectory_count
        else:
            k_new = K[i] - k[i]
        if k_new <= 0:
            if k_new != 0:
                print('{:d} excess trajectories at step {:d}'.format(-k_new, i))
            continue
        if lifetime:
            _lifetime = np.array([ lifetime(t) for j in range(k_new) ])
        else:
            _lifetime = -np.log(1 - np.random.rand(k_new)) * lifetime_tau
        _lifetime = np.rint(_lifetime / time_step).astype(int) + 1
        _lifetime = _lifetime[min_step_count <= _lifetime]
        time_support.append((t, _lifetime))
        if _lifetime.size == 0:
            continue
        _lifetimes, _count = np.unique(_lifetime, return_counts=True)
        _lifetime = np.zeros(_lifetimes.max(), dtype=_count.dtype)
        _lifetime[_lifetimes-1] = _count
        if K is not None:
            _count = np.flipud(np.cumsum(np.flipud(_lifetime)))
            k[i:min(i+_count.size,k.size)] += _count[:min(k.size-i,_count.size)]
    #
    actual_time_step = time_step / float(minor_step_count + 1)
    total_location_count = sum([ np.sum(_lifetime) for _, _lifetime in time_support ])
    # generate the trajectories
    N = np.empty((total_location_count, ), dtype=int)       # trajectory index
    X = np.empty((total_location_count, dim), dtype=_box.dtype) # spatial coordinates
    T = np.empty((total_location_count, ), dtype=float)     # time
    i, n = 0, 0
    for t0, lifetimes in time_support:
        k = lifetimes.size # number of new trajectories at time t
        X0 = np.random.rand(k, dim) * support_size + support_lower_bound # initial coordinates
        for x0, _lifetime in zip(X0, lifetimes): # for each new trajectory
            n += 1
            N[i] = n
            X[i] = x = x0
            T[i] = t = t0
            i += 1
            # from here the main code (``not reflect``) is duplicated to reduce the number of if-tests
            if reflect:
                # duplicated code
                for j in range(1,_lifetime):
                    for _ in range(minor_step_count+1):
                        D = diffusivity(x, t)
                        A = drift(x, t, D)
                        dx = actual_time_step * A + \
                            np.sqrt(actual_time_step * 2. * D) * np.random.randn(dim)
                        x = x + dx
                        # additional code
                        above = support_upper_bound < x
                        x[above] = 2*support_upper_bound[above] - x[above]
                        below = x < support_lower_bound
                        if np.any(below & above):
                            raise NotImplementedError('the jump is so large that its reflection also exceeds the opposite bound')
                        x[below] = 2*support_lower_bound[below] - x[below]
                        #
                        t = t + actual_time_step
                    t = t0 + j * time_step # moderate numerical precision errors
                    N[i] = n
                    X[i] = x
                    T[i] = t
                    i += 1
            else:
                # reference code
                for j in range(1,_lifetime):
                    for _ in range(minor_step_count+1):
                        D = diffusivity(x, t)
                        A = drift(x, t, D)
                        dx = actual_time_step * A + \
                            np.sqrt(actual_time_step * 2. * D) * np.random.randn(dim)
                        x = x + dx
                        t = t + actual_time_step
                    t = t0 + j * time_step # moderate numerical precision errors
                    N[i] = n
                    X[i] = x
                    T[i] = t
                    i += 1
    # format the data as a dataframe
    columns = 'xyz'
    if dim <= 3:
        xcols = [ d for d in columns[:dim] ]
    else:
        xcols = [ 'x'+str(i) for i in range(dim) ]
    points = pd.DataFrame(N, columns=['n']).join(
        pd.DataFrame(X, columns=xcols)).join(
        pd.DataFrame(T, columns=['t']))
    # post-process
    if not full and not reflect:
        points = crop(points, _box)
        points = points.loc[points['t'] <= duration+time_step*.1]
        if not single:
            n, count = np.unique(points['n'].values, return_counts=True)
            n = n[count == 1]
            if n.size:
                points = points.loc[~points['n'].isin(n)]
        points.index = np.arange(points.shape[0])
    return points


def add_noise(points, sigma, copy=False):
    columns = [ c for c in points.columns if c not in ('n', 't') ]
    dim = len(columns)
    npoints = points.shape[0]
    if copy:
        points = points.copy()
    points[columns] += sigma * np.random.randn(npoints, dim)
    return points


def truth(cells, t=None, diffusivity=None, force=None, potential=None, **kwargs):
    """
    Generate maps for the true diffusivity/force distribution.

    If `cells` is a :class:`~tramway.tessellation.base.Partition` object,
    :func:`~tramway.inference.base.distributed` is called on `cells` and with the trailing
    keyword arguments to build a :class:`~tramway.inference.base.FiniteElements` object.

    Arguments:

        cells (Partition or FiniteElements): distributed data ready for inference

        t (float): time as understood by `diffusivity` and `force`

        diffusivity (callable): admits the coordinates of a single location (array-like)
            and time (float, if `t` is defined) and returns the local diffusivity (float)

        force (callable): admits the coordinates of a single location (array-like)
            and time (float, if `t` is defined) and returns the local force (array-like)

        potential (callable): admits the coordinates of a single location (array-like)
            and time (float, if `t` is defined) and returns the local potential energy (float)

    Returns:

        pandas.DataFrame: diffusivity/force maps
    """
    try:
        cells.cells
    except AttributeError:
        import tramway.inference as ti
        cells = ti.distributed(cells, **kwargs)
    I, DVF = [], []
    for i in cells.cells:
        cell = cells.cells[i]
        if not I:
            dim = cell.center.size
        I.append(i)
        if diffusivity is None:
            D = []
        elif t is None:
            D = [diffusivity(cell.center)]
        else:
            D = [diffusivity(cell.center, t)]
        if force is None:
            F = []
        elif t is None:
            F = force(cell.center)
        else:
            F = force(cell.center, t)
        if potential is None:
            V = []
        elif t is None:
            V = [potential(cell.center)]
        else:
            V = [potential(cell.center, t)]
        DVF.append(np.concatenate((D, V, F)))
    DVF = np.vstack(DVF)
    if diffusivity is None:
        columns = []
    else:
        columns = [ 'diffusivity' ]
    if potential is not None:
        columns.append('potential')
    if force is not None:
        columns += [ 'force x' + str(col+1) for col in range(dim) ]
    return pd.DataFrame(index=I, data=DVF, columns = columns)


