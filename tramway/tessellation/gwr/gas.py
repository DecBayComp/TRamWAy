# -*- coding: utf-8 -*-

# Copyright © 2017-2018, Institut Pasteur
#   Contributor: François Laurent
# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import *
import numpy as np
import numpy.linalg as la
from .graph import *
#from .graph.array import ArrayGraph
import time
from scipy.spatial.distance import cdist
from scipy.special import gamma
from .dichotomy import Dichotomy


class Gas(Graph):
    """Implementation of the *Grow(ing) When Required* clustering algorithm, first inspired from
    [Marsland02]_ and then extensively modified.

    .. [Marsland02] Marsland, S., Shapiro, J. and Nehmzow, U. (2002). A self-organising network that grows when required. Neural Networks 15, 1041-1058. doi:10.1026/S0893-6080(02)00078-3

    Of note, :class:`Gas` features a :attr:`trust` attribute that enables faster learning when
    inserting new nodes. Instead of inserting a node at ``(w + eta) / 2`` (with ``trust == 0``) like
    in the standard algorithm, you can insert sample point `eta` (with ``trust == 1``) or any point
    in between.
    :class:`Gas` also features a 'collapse' step at the end of each batch. This step consists of
    merging nodes that are closer from each other than threshold :attr:`collapse_below`.
    Last but not least, the insertion threshold and collapse threshold can be functions of the
    current training sample and nearest node.

    Implementation note: :class:`Gas` implements :class:`Graph` as a proxy, instanciating or taking
    another `Graph` object with optional input argument `graph` and delegates all :class:`Graph` 's
    method to that object attribute.
    """
    __slots__ = ['graph', 'insertion_threshold', 'trust', 'learning_rate', \
        'habituation_threshold', 'habituation_initial', 'habituation_alpha', \
        'habituation_tau', 'edge_lifetime', 'batch_size', 'collapse_below', 'knn', \
        'topology']

    def connect(self, n1, n2, **kwargs):
        self.graph.connect(n1, n2, **kwargs)
    def disconnect(self, n1, n2, edge=None):
        self.graph.disconnect(n1, n2, edge)
    def get_node_attr(self, n, attr):
        return self.graph.get_node_attr(n, attr)
    def set_node_attr(self, n, **kwargs):
        self.graph.set_node_attr(n, **kwargs)
    def get_edge_attr(self, e, attr):
        return self.graph.get_edge_attr(e, attr)
    def set_edge_attr(self, e, **kwargs):
        self.graph.set_edge_attr(e, **kwargs)
    @property
    def size(self):
        return self.graph.size
    def iter_nodes(self):
        return self.graph.iter_nodes()
    def iter_neighbors(self, n):
        return self.graph.iter_neighbors(n)
    def iter_edges(self):
        return self.graph.iter_edges()
    def iter_edges_from(self, n):
        return self.graph.iter_edges_from(n)
    def has_node(self, n):
        return self.graph.has_node(n)
    def add_node(self, **kwargs):
        return self.graph.add_node(**kwargs)
    def del_node(self, n):
        self.graph.del_node(n)
    def stands_alone(self, n):
        return self.graph.stands_alone(n)
    def find_edge(self, n1, n2):
        return self.graph.find_edge(n1, n2)
    def are_connected(self, n1, n2):
        return self.graph.are_connected(n1, n2)
    def export(self, **kwargs):
        return self.graph.export(**kwargs)
    def square_distance(self, attr, eta, **kwargs):
        return self.graph.square_distance(attr, eta, **kwargs)

    def __init__(self, sample, graph=None):
        if 1 < sample.shape[0]:
            w1 = sample[0]
            w2 = sample[-1]
        else:
            raise ValueError
        if graph:
            if isinstance(graph, type):
                self.graph = graph({'weight': None, 'habituation_counter': 0, \
                        'radius': 0.0}, \
                    {'age': 0})
            else:
                self.graph = graph
        else:
            from .graph.array import ArrayGraph # default implementation
            self.graph = ArrayGraph({'weight': None, 'habituation_counter': 0, \
                    'radius': 0.0}, \
                {'age': 0})
        n1 = self.add_node(weight=w1)
        n2 = self.add_node(weight=w2)
        e  = self.connect(n1, n2)
        self.insertion_threshold = .95
        self.trust = 0 # in [0, 1]
        self.learning_rate = (.2, .006) # default values should be ok
        self.habituation_threshold = float('+inf') # useless, basically..
        self.habituation_initial = 1 # useless, basically..
        self.habituation_alpha = (1.05, 1.05) # should be greater than habituation_initial
        self.habituation_tau = (3.33, 14.33) # supposed to be time, but actually is to be
        # thought as a number of iterations; needs to be appropriately set
        self.edge_lifetime = 100 # may depend on the number of neighbors per node and
        # therefore on the dimensionality of the data;
        # also depends on the number of nodes, as lifetime reset may follow a Poisson distribution;
        # therefore, a floating point value below 1 will be interpreted as a fraction of the
        # number of nodes
        self.batch_size = 1000 # should be an order of magnitude or two below the total
        # sample size
        self.collapse_below = None
        self.knn = None
        self.topology = 'approximate density'

    def local_insertion_threshold(self, eta, node, *vargs):
        """
        """
        ##TODO: handle time more explicitly
        if self.knn:
            return max(self.insertion_threshold[0], \
                min(vargs[0], \
                    self.insertion_threshold[1]))
            # insertion_threshold are already radii (diameters / 2)
        else:
            return self.insertion_threshold

    def collapse_threshold(self, node1, node2):
        """
        """
        if self.knn:
            radius1 = self.get_node_attr(node1, 'radius')
            radius2 = self.get_node_attr(node2, 'radius')
            return self.collapse_below / self.insertion_threshold[0] * \
                max(radius1, radius2)
        else:
            return self.collapse_below

    def habituation_function(self, t, i=0):
        """
        Arguments:
            t (int or float or array): counter or time.
            i (int): proximity level (0=nearest, 1=second nearest).

        Returns:
            float or array: habituation.
        """
        return self.habituation_initial - \
            (1 - exp(-self.habituation_alpha[i] * t / self.habituation_tau[i])) / \
            (self.habituation_alpha[i])

    def habituation(self, node, i=0):
        """Returns the habituation of a node. Set ``i=0`` for the nearest node,
        ``i=1`` for a neighbor of the nearest node."""
        return self.habituation_function(float(self.get_node_attr(node, 'habituation_counter')), i)

    def plot_habituation(self):
        ##TODO: find a more general implementation for reading the habituations, and underlay
        # a histogram of the habituation counter in the graph
        import matplotlib.pyplot as plt
        import matplotlib.colors as clr
        import matplotlib.patches as patches
        if isinstance(self.graph, DictGraph):
            tmax = max(self.graph.nodes['habituation_counter'].values())
            tmax = round(float(tmax) / 100) * 100
        else:
            tmax = self.batch_size / 10
        t = np.arange(0, tmax)
        plt.plot(t, self.habituation_function(t, 1), 'c-')
        plt.plot(t, self.habituation_function(t, 0), 'b-')
        plt.xlim(0, tmax)
        plt.ylim(0, self.habituation_initial)
        plt.title('habituation function: alpha={}, tau={}'.format(self.habituation_alpha[0], \
                self.habituation_tau[0]))
        plt.ylabel('habituation')
        plt.xlabel('iteration number')

    def get_weight(self, node):
        return self.get_node_attr(node, 'weight')

    def set_weight(self, node, weight):
        self.set_node_attr(node, weight=weight)

    def increment_habituation(self, node):
        count = self.get_node_attr(node, 'habituation_counter') + 1
        self.set_node_attr(node, habituation_counter=count)
        return count

    def increment_age(self, edge):
        age = self.get_edge_attr(edge, 'age') + 1
        self.set_edge_attr(edge, age=age)
        return age

    def habituate(self, node):
        if self.edge_lifetime < 1:
            max_age = max(20, self.edge_lifetime * float(self.size))
        else:
            max_age = self.edge_lifetime
        self.increment_habituation(node)
        for edge, neighbor in list(self.iter_edges_from(node)):
            age = self.increment_age(edge)
            self.increment_habituation(neighbor) # increment before checking for age of the
            # corresponding edge, because `neighbor` may no exist afterwards; otherwise, to
            # increment after, it is necessary to check for existence of the node; should be
            # faster this way, due to low deletion rate
            if max_age < age:
                self.disconnect(node, neighbor, edge)
                if self.stands_alone(neighbor):
                    self.del_node(neighbor)
        if self.stands_alone(node):
            self.del_node(node)


    def batch_train(self, sample, eta_square=None, radius=None, grab=None, max_frames=None, **grab_kwargs):
        """This method grows the gas for a batch of data and implements the core GWR algorithm.
        :meth:`train` should be called instead."""
        if eta_square is None:
            eta_square = np.sum(sample * sample, axis=1)
        if radius is None:
            r = []
        errors = []
        for k in np.arange(0, sample.shape[0]):
            eta = sample[k]
            if radius is not None:
                r = [radius[k]]
            # find nearest and second nearest nodes
            dist2, index_to_node = self.square_distance('weight', eta, eta2=eta_square[k])
            i = np.argsort(dist2)
            dist2_min = dist2[i[0]]
            try:
                dist_min = sqrt(dist2_min)
            except ValueError:
                precision = int(np.dtype(dist2_min.dtype).str[-1])
                if (precision==8 and -1e-10 < dist2_min) or (precision==4 and -1e-3 < dist_min):
                    dist_min = 0
                    import warnings
                    warnings.warn('Rounding error: negative distance', RuntimeWarning)
                else:
                    print('square distance=', dist2_min, '   num. type=', dist2_min.dtype)
                    raise ValueError('Negative distance') from None
            nearest, second_nearest = index_to_node(i[:2])
            errors.append(dist_min)
            # test activity and habituation against thresholds
            activity = dist_min
            habituation = self.habituation(nearest)
            w = self.get_weight(nearest)
            activity_threshold = self.local_insertion_threshold(eta, w, *r)
            if isinstance(activity_threshold, tuple):
                assert not self.knn
                raise ValueError('insertion_threshold is defined as (lower, upper) bounds whereas knn is {}'.format(self.knn))
            if activity_threshold < activity and \
                habituation < self.habituation_threshold:
                # insert a new node and connect it with the two nearest nodes
                self.disconnect(nearest, second_nearest)
                l = .5 + self.trust * .5 # mixing coefficient in the range [.5, 1]
                new_node = self.add_node(weight=(1.0 - l) * w + l * eta)
                if self.knn:
                    self.set_node_attr(new_node, radius=activity_threshold)
                self.connect(new_node, nearest)
                self.connect(new_node, second_nearest)
            else:
                # move the nearest node and its neighbors towards the sample point
                self.connect(nearest, second_nearest)
                self.set_weight(nearest, w + self.learning_rate[0] * habituation * (eta - w))
                for i in self.iter_neighbors(nearest):
                    w = self.get_weight(i)
                    self.set_weight(i, w + self.learning_rate[1] * \
                        self.habituation(i, 1) * (eta - w))
            # update habituation counters
            self.habituate(nearest) # also habituates neighbors
            #
            if grab is not None:
                if max_frames is None or k < max_frames:
                    self.grab_frame(grab, **grab_kwargs)
        return errors

    def grab_frame(self, grab, axes=None, color='r', **kwargs):
        assert axes is not None
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        # draw the graph
        lines = []
        for n1 in self.iter_nodes():
            w1 = self.get_weight(n1)
            for n2 in self.iter_neighbors(n1):
                w2 = self.get_weight(n2)
                lines.append(axes.plot([w1[0], w2[0]], [w1[1], w2[1]], color+'-', linewidth=1))
        #
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        axes.set_axis_off()
        # grab
        grab.grab_frame()
        # clear the graph
        while lines:
            lines.pop().pop()

    def grab_batch_init(self, grab, sample, i, batch, axes=None, max_frames=None, **kwargs):
        assert axes is not None
        n = sample.shape[0]
        _batch = np.zeros(n, dtype=bool)
        _batch[batch] = True
        axes.clear()
        axes.plot(sample[~_batch,0], sample[~_batch,1], '.', color='lightgray', markersize=1)
        axes.plot(sample[_batch,0], sample[_batch,1], 'k.', markersize=1)
        kwargs = dict(axes=axes)
        if max_frames is not None:
            kwargs['max_frames'] = max_frames - (i-1)*self.batch_size
        self.grab_frame(grab, **kwargs)
        return kwargs

    def grab_completion(self, grab, sample, axes=None, **kwargs):
        assert axes is not None
        axes.clear()
        axes.plot(sample[:,0], sample[:,1], '.', color='lightgray', markersize=1)
        self.grab_frame(grab, axes, 'k')

    def train(self, sample, pass_count=None, residual_max=None, error_count_tol=1e-6, \
        min_growth=None, collapse_tol=None, stopping_criterion=2, verbose=False, \
        plot=False, grab=None, **kwargs):
        """
        Grow the gas.

        :meth:`train` splits the sample into batches, successively calls :meth:`batch_train`
        on these batches of data, collapses the gas if necessary and stops if stopping criteria
        are met.

        The input arguments that define the stopping criteria are:

        Arguments:

            pass_count (pair of floats or None):
                (min, max) numbers of passes of the sample the algorithm should/can run.
                Any bound can be set to ``None``.
            residual_max (float):
                a residual is calculated for each sample point, before fitting the gas
                towards it, and `residual_max` is a threshold above which a residual is
                regarded as an 'error'. This parameter works in combination with
                `error_count_tol`.
            error_count_tol (float):
                maximum proportion of 'errors' (number of errors in a batch over the
                size of the batch). If this maximum is exceeded, then the :meth:`train`
                samples another batch and keeps on training the gas. Otherwise, the next
                criteria apply.
            min_growth (float):
                minimum relative increase in the number of nodes from an iteration to
                the next.
            collapse_tol (float):
                maximum allowed ratio of the number of collapsed nodes over the total
                number of nodes.
            stopping_criterion (int, deprecated):

                1. performs a linear regression for the pre-fitting residual across time
                   and 'tests' whether the trend is not negative.
                2. stops if the average residual for the current batch is greater than
                   that of the previous batch.


        """
        ## TODO: clarify the code
        if isinstance(sample, tuple):
            locations, displacements = sample
            sample = locations
        elif self.knn and self.topology == 'displacement length':
            raise ValueError('no displacement information found')
        n = sample.shape[0]
        l = 0 # node count
        residuals = []
        if stopping_criterion == 1:
            import statsmodels.api as sm
        # if stopping_criterion is 2
        residual_mean = float('+inf')
        #
        if pass_count:
            p = .95
            #k1 = log(1 - p) / log(1 - 1 / float(n)) # sample size for each point to have probability p of being chosen once
            k1 = n
        if residual_max:
            if error_count_tol < 1:
                error_count_tol = ceil(error_count_tol * float(n))
        # local radius
        if self.knn:
            if not isinstance(self.insertion_threshold, tuple):
                self.insertion_threshold = (self.insertion_threshold, \
                    self.insertion_threshold * 4)
            eta_square = np.sum(sample * sample, axis=1)
            if self.topology == 'approximate density':
                radius = self.boxed_radius(sample, self.knn, self.insertion_threshold[0], \
                        self.insertion_threshold[1], verbose, plot)
            elif self.topology == 'displacement length':
                radius = self.boxed_average(sample, displacements, self.knn, self.insertion_threshold[0], \
                        self.insertion_threshold[1], verbose, plot)
            else:
                raise ValueError("topology=='{}' not supported".format(self.topology))
        # grab
        batch_kwargs = {}
        if grab is not None:
            batch_kwargs['grab'] = grab
            if sample.shape[1] != 2:
                raise ValueError('cannot grab data that are not 2D')
            max_frames = kwargs.get('max_frames', None)
            max_batches = kwargs.get('max_batches', None)
        # loop
        t = []
        i = 0
        do = True
        while do:
            i += 1
            if verbose:
                t0 = time.time()
            batch = np.random.choice(n, size=self.batch_size)
            # grab
            if grab is not None:
                if max_batches is None or i <= max_batches:
                    batch_kwargs.update(self.grab_batch_init(grab, sample, i, batch, **kwargs))
                    if 1 < i and \
                            (max_frames is None or \
                                (max_batches is not None and \
                                    max_frames <= (i-1)*self.batch_size)):
                        batch_kwargs = {}
                else:
                    batch_kwargs = {}
            #
            if self.knn:
                r = self.batch_train(sample[batch], eta_square[batch], radius[batch], **batch_kwargs)
            else:
                r = self.batch_train(sample[batch], **batch_kwargs)
            residuals += r
            l_prev = l
            l = self.size
            txt = l
            if self.collapse_below:
                self.collapse()
                dl = l - self.size
                if verbose and 0 < dl:
                    txt = '{} (-{})'.format(l, dl)
            if verbose:
                if i == 1:
                    if self.collapse_below:
                        print("\t#nodes (-#collapsed)")
                    else:   print("\t#nodes")
                ti = time.time() - t0
                t.append(ti)
                print("{}\t{}\tElapsed: {:.0f} ms".format(i, txt, ti * 1e3))
            # enforce some stopping or continuing conditions
            if pass_count:
                k = i * self.batch_size
                if pass_count[0] and k < pass_count[0] * k1:
                    #print(('pass_count', k, pass_count[0] * k1))
                    continue
                elif pass_count[1] and pass_count[1] * k1 <= k:
                    if verbose:
                        print('stopping criterion: upper bound for `pass_count` reached')
                    break # do = False
            if residual_max:
                error_count = len([ residual for residual in r
                        if residual_max < residual ])
                if error_count_tol < error_count:
                    #print(('error_count_tol', error_count_tol, error_count))
                    continue
            if min_growth is not None:
                growth = float(l - l_prev) / float(l_prev)
                if growth < min_growth:
                    if verbose:
                        print('stopping criterion: relative growth: {:.0f}%'.format(growth * 100))
                    break
            if self.collapse_below and collapse_tol:
                collapse_rel = float(dl) / float(l)
                if collapse_tol < collapse_rel:
                    if verbose:
                        print('stopping criterion: relative collapse: {:.0f}%'.format(collapse_rel * 100))
                    break
            # exclusive criteria
            if stopping_criterion == 1:
                regression = sm.OLS(np.array(r), \
                        sm.add_constant(np.linspace(0,1,len(r)))\
                    ).fit()
                if 0 < regression.params[1]:
                    fit = 0
                else:
                    fit = 1/(1+exp((.1-regression.pvalues[1])/.01)) # invert p-value
                do = tolerance < fit
            elif stopping_criterion == 2:
                residual_prev = residual_mean
                #_, r = self.eval(sample[np.random.choice(n, size=self.validation_batch_size),:])
                residual_mean = np.mean(r)
                residual_std  = np.std(r)
                do = (residual_mean - residual_prev) / residual_std < 0
        if verbose:
            t = np.asarray(t)
            print('Elapsed:  mean: {:.0f} ms  std: {:.0f} ms'.format(np.mean(t) * 1e3, \
                                    np.std(t) * 1e3))
        if grab is not None and (max_frames is None or max_frames < sample.shape[0]):
            self.grab_completion(grab, sample, **kwargs)
        return residuals

    def boxed_radius(self, sample, knn, rmin, rmax, verbose=False, plot=False):
        #plot = True
        if plot:
            import matplotlib.pyplot as plt
        d = sample.shape[1]
        d = int(d) # PY2
        #if d == 2:
        #       return self.boxed_radius2d(sample, knn, rmin, rmax, plot) # faster
        if verbose:
            t0 = 0
            t = time.time()
        # bounding box(es)
        sample = np.asarray(sample)
        smin = sample.min(axis=0)
        smax = sample.max(axis=0)
        unit = rmin / sqrt(float(d)) # min radius will be the diagonal of the hypercube
        # volume ratio of the unit hypercube (1) and the inscribed d-ball
        dim_penalty = 1 / (pi ** (float(d) / 2.0) / gamma(float(d) / 2.0 + 1.0) * 0.5 ** d)
        max_n_units = ceil(2.0 * rmax / unit * dim_penalty) # max number of "circles" where to look for neighbors
        bmin, bmax = smin, smax # multiple names because of copying/pasting
        dim = np.ceil((bmax - bmin) / unit) - 1
        adjusted_bmax = bmin + dim * unit # bmin and adjusted_bmax are the lower vertices (origins) of the end cubes
        dim = [ int(_d) + 1 for _d in dim ] # _d for PY2
        # partition
        cell = dict()
        cell_indices = dict()
        count = np.zeros(dim, dtype=int)
        grid = Dichotomy(sample, base_edge=unit, origin=bmin, max_level=0)
        grid.split()
        for j in grid.cell:
            i, _, k = grid.cell[j]
            ids, pts = grid.subset[k]
            if ids.size:
                i = tuple([ int(_n) for _n in np.round((i - bmin) / unit) ])
                cell[i] = pts
                if ids.size < knn + 1:
                    cell_indices[i] = ids
                try:
                    count[i] = ids.size
                except IndexError:
                    print((count.shape, i))
                    raise
        # counts are density estimate; check how good/poor they are
        if d == 2 and plot:
            if verbose:
                t0 += time.time() - t
            plt.imshow(count.T)
            plt.gca().invert_yaxis()
            plt.show()
            if verbose:
                t = time.time()
        # count_threshold is useful for saving computation time,
        # skipping poorly populated cells with populated neighbor cells.
        # lower threshold means higher skip frequency (more approximations)
        count_threshold = np.zeros_like(count)
        for i in range(d):
            a = np.swapaxes(count_threshold, 0, i) # view (numpy >= 1.10)
            b = np.swapaxes(count, 0, i) # view
            a[:-1,...] += b[1::,...]
            a[1::,...] += b[:-1,...]
        count_threshold = (knn + 1) - count_threshold / (d * 2) # average number of points in neighbor cells
        # count_threshold should depend on space dimension
        count_threshold *= 2 # hard-coded, empirical, actually not even adjusted
        # adjust the requested number of neighbors (k)
        k = int(round(float(knn) * dim_penalty)) # request more neighbors in the cube so that the inscribed ball should contain knn as expected (hypothesis: that points are homogeneously distributed
        k *= 2 # gwr actually works better with higher k; hard-coded factor; empirical
        scale = {}
        n = sample.shape[0]
        radius = rmin * np.ones(n, dtype=sample.dtype)
        for i in cell_indices:
            if count_threshold[i] < count[i]:
                continue # consider that local radius are minimal (saves time)
            m = 0
            while True: # do..while
                m += 1
                I = [ np.arange(max(0, _k - m), min(_k + m + 1, _d)) \
                    for _k, _d in zip(i, dim) ] # _k, _d for PY2
                I = np.meshgrid(*I, indexing='ij')
                I = np.column_stack([ _i.flatten() for _i in I ]) # _i for PY2
                p = np.sum(count[tuple(I.T)])
                if not (p < k + 1 and m < max_n_units):
                    break
            scale[i] = m
            j = cell_indices[i]
            if p < k + 1:
                # `m` has reached `max_n_units`
                radius[j] = rmax
                continue
            X = []
            for _i in I:
                try:
                    _x = cell[tuple(_i)]
                except KeyError:
                    pass
                else:
                    X.append(_x)
            #X = [ cell[tuple(_i)] for _i in I if tuple(_i) in cell ] # _i for PY2
            X = np.concatenate(X, axis=0)
            x = cell[i]
            # dist with SciPy
            D = cdist(X, x)
            D.sort(axis=0)
            r = D[k]
            # and without SciPy (better for bigger matrices)
            #D = np.dot(X, x.T)
            #x2 = np.sum(x * x, axis=1, keepdims=True)
            #D -= 0.5 * x2.T
            #X2 = np.sum(X * X, axis=1, keepdims=True)
            #D -= 0.5 * X2
            #D.sort(axis=0)
            #r = D[-(k+1)]
            #r = np.sqrt(-2.0 * r)
            #r[np.isnan(r)] = 0.0 # numeric precision errors => negative square distances => NaN distances
            #
            del D
            radius[j] = r
        if verbose:
            print('estimating density: elapsed time {:d} ms'.format(int(round((t0 + time.time() - t) * 1000)))) # int for PY2
        # plot local radius
        #plot = True
        if d == 2 and plot:
            color = (radius - rmin) / (rmax - rmin)
            r = self.collapse_below * 0.5 # collapse_below defines a diameter
            cmap = plt.get_cmap('RdPu')
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            for x in np.arange(smin[0], smax[0], unit):
                plt.plot([x, x], [smin[1], smax[1]], 'y-')
            for y in np.arange(smin[1], smax[1], unit):
                plt.plot([smin[0], smax[0]], [y, y], 'y-')
            for i in range(n):
                c = color[i]
                if rmax < radius[i]:
                    ax.add_artist(patches.Circle(sample[i], r * 4, \
                        color=cmap(0.0), alpha=0.1))
                elif radius[i] < rmin - np.spacing(1):
                    ax.add_artist(patches.Circle(sample[i], r * 0.25, \
                        color=cmap(1.0)))
                elif rmin + np.spacing(1) < radius[i]:
                    ax.add_artist(patches.Circle(sample[i], r, \
                        color=cmap(1.0 - c)))
            plt.plot(sample[:,0], sample[:,1], 'k+')
            for i, x in cell.items():
                center = smin + (np.asarray(list(i)) + 0.5) * unit
                count = x.shape[0]
                if count < knn + 1:
                    color = 'm'
                    txt = "{:d}\n{:d}".format(count, scale.get(i, 0))
                else:
                    color = 'c'
                    txt = str(count)
                plt.text(center[0], center[1], txt, color=color, \
                    horizontalalignment='center', verticalalignment='center')
            try:
                plt.show()
            except AttributeError: # window has been closed; go on
                pass
        return radius


    def boxed_radius2d(self, sample, knn, rmin, rmax, plot=False):
        # deprecated: too many approximations here
        if plot:
            import matplotlib.pyplot as plt
        t = time.time()
        n = sample.shape[0]
        radius = rmin * np.ones(n, dtype=sample.dtype)
        # bounding box(es)
        smin = sample.min(axis=0)
        smax = sample.max(axis=0)
        unit = rmin * sqrt(2.0)
        max_n_units = ceil(2.0 * rmax / unit)
        ix = np.arange(n)
        # grid
        bounding_box = [(smin, smax), \
            (smin + unit * 0.25, smax - unit * 0.75), \
            (smin + unit * 0.5,  smax - unit * 0.5 ), \
            (smin + unit * 0.75, smax - unit * 0.25)]
        for bmin, bmax in bounding_box:
            bmax_ = np.ceil(bmax / unit) * unit + np.spacing(1)
            x_grid = np.arange(bmin[0], bmax_[0], unit)
            y_grid = np.arange(bmin[1], bmax_[1], unit)
            # partition strategy
            if x_grid.size <= y_grid.size:
                grid0 = x_grid
                grid1 = y_grid
                x0 = sample[:, 0]
                x1 = sample[:, 1]
            else:
                grid0 = y_grid
                grid1 = x_grid
                x0 = sample[:, 1]
                x1 = sample[:, 0]
            n0 = grid0.size - 1
            n1 = grid1.size - 1
            # partition
            cell = dict()
            icell = dict()
            undefined = []
            lower0 = np.ones(n, dtype=np.bool)
            for i in range(n0):
                upper0 = x0 < grid0[i+1]
                mask0 = np.logical_and(lower0, upper0)
                xi = x1[mask0]
                ix0 = ix[mask0]
                lower0 = np.logical_not(upper0)
                ni = xi.shape[0]
                lower1 = np.ones(ni, dtype=np.bool)
                for j in range(n1):
                    upper1 = xi < grid1[j+1]
                    ix1 = ix0[np.logical_and(lower1, upper1)]
                    xj = sample[ix1]
                    cell[(i,j)] = xj
                    if 0 < xj.size and xj.shape[0] < knn:
                        undefined.append((i,j))
                        icell[(i,j)] = ix1
                    lower1 = np.logical_not(upper1)
            #print(len(undefined))
            # compute radius where point density is low
            for i, j in undefined:
                nij = 0
                m = 0
                while nij < knn:
                    m += 1
                    Xij = [ cell[(k,l)] for k in range(i-m,i+m) for l in range(j-m,j+m) \
                        if 0 <= k and k < n0 and 0 <= l and l < n1 ]
                    nij = sum([ x.shape[0] for x in Xij ])
                Xij = np.concatenate(Xij, axis=0)
                xij = cell[(i,j)]
                D = np.dot(xij, Xij.T)
                x2 = np.sum(xij * xij, axis=1)
                x2.shape = (x2.size, 1)
                D -= 0.5 * x2
                X2 = np.sum(Xij * Xij, axis=1)
                X2.shape = (1, X2.size)
                D -= 0.5 * X2
                D.sort()
                r = D[:, (nij - 1) - knn]
                del D
                r = np.sqrt(-2.0 * r)
                r[np.isnan(r)] = 0.0
                #if any(r < rmin):
                #       print(r)
                radius[icell[(i,j)]] = np.maximum(radius[icell[(i,j)]], r)
        print('elapsed time {:d} ms'.format(round((time.time() - t) * 1000)))
        # plot local radius
        if plot:
            color = np.minimum((radius - rmin) / (rmax - rmin), 1.0) * 0.9
            cmap = plt.get_cmap('RdPu')
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            for i in range(n):
                c = color[i]
                if np.spacing(1) < c:
                    ax.add_artist(patches.Circle(sample[i], radius[i], \
                        color=cmap(1 - c), alpha=0.2))
            plt.plot(sample[:,0], sample[:,1], 'k+')
            plt.show()
        return radius

    def exact_radius(self, sample, eta_square, knn, step, *vargs):
        # deprecated: extremely slow
        n = sample.shape[0]
        w = sample.astype(np.float32)
        w2 = eta_square.astype(np.float32) * 0.5
        w2.shape = (w2.size, 1)
        D = []
        for i in range(0, n, step):
            j = range(i, min(i + step, n))
            t = time.time()
            d = np.dot(w[j], w.T)
            d -= w2[j]
            d -= w2.T
            print("Elapsed: {:.0f} ms".format((time.time() - t) * 1e3))
            t = time.time()
            d.sort() # along last axis if faster
            print("Elapsed: {:.0f} ms".format((time.time() - t) * 1e3))
            d = d[:, (n - 1) - (self.knn + 1)].astype(sample.dtype)
            d = np.sqrt(-2.0 * d)
            D.append(d)
        radius = np.concatenate(D)
        return radius


    def boxed_average(self, locations, displacements, knn, rmin, rmax, verbose=False, plot=False):
        #plot = True
        if plot:
            import matplotlib.pyplot as plt
        if not locations.shape == displacements.shape:
            raise ValueError('locations and displacements do not match in shape')
        d = locations.shape[1]
        d = int(d) # PY2
        if verbose:
            t0 = 0
            t = time.time()
        # bounding box(es)
        sample = np.asarray(locations)
        displacements = np.asarray(displacements)
        smin = sample.min(axis=0)
        smax = sample.max(axis=0)
        unit = rmin / sqrt(float(d)) # min radius will be the diagonal of the hypercube
        # volume ratio of the unit hypercube (1) and the inscribed d-ball
        dim_penalty = 1 / (pi ** (float(d) / 2.0) / gamma(float(d) / 2.0 + 1.0) * 0.5 ** d)
        max_n_units = ceil(2.0 * rmax / unit * dim_penalty) # max number of "circles" where to look for neighbors
        bmin, bmax = smin, smax # multiple names because of copying/pasting
        dim = np.ceil((bmax - bmin) / unit) - 1
        adjusted_bmax = bmin + dim * unit # bmin and adjusted_bmax are the lower vertices (origins) of the end cubes
        dim = [ int(_d) + 1 for _d in dim ] # _d for PY2
        # partition
        cell = dict()
        cell_indices = dict()
        count = np.zeros(dim, dtype=int)
        grid = Dichotomy(sample, base_edge=unit, origin=bmin, max_level=0)
        grid.split()
        for j in grid.cell:
            i, _, k = grid.cell[j]
            ids, pts = grid.subset[k]
            if ids.size:
                i = tuple([ int(_n) for _n in np.round((i - bmin) / unit) ])
                cell[i] = pts
                if ids.size < knn + 1:
                    cell_indices[i] = ids
                try:
                    count[i] = ids.size
                except IndexError:
                    print((count.shape, i))
                    raise
        # counts are density estimate; check how good/poor they are
        if d == 2 and plot:
            if verbose:
                t0 += time.time() - t
            plt.imshow(count.T)
            plt.gca().invert_yaxis()
            plt.show()
            if verbose:
                t = time.time()
        # count_threshold is useful for saving computation time,
        # skipping poorly populated cells with populated neighbor cells.
        # lower threshold means higher skip frequency (more approximations)
        count_threshold = np.zeros_like(count)
        for i in range(d):
            a = np.swapaxes(count_threshold, 0, i) # view (numpy >= 1.10)
            b = np.swapaxes(count, 0, i) # view
            a[:-1,...] += b[1::,...]
            a[1::,...] += b[:-1,...]
        count_threshold = (knn + 1) - count_threshold / (d * 2) # average number of points in neighbor cells
        # count_threshold should depend on space dimension
        count_threshold *= 2 # hard-coded, empirical, actually not even adjusted
        # adjust the requested number of neighbors (k)
        k = int(round(float(knn) * dim_penalty)) # request more neighbors in the cube so that the inscribed ball should contain knn as expected (hypothesis: points are homogeneously distributed)
        k *= 2 # gwr actually works better with higher k; hard-coded factor; empirical
        scale = {}
        n = sample.shape[0]
        dr = displacements
        displacement_length = np.sqrt(np.sum(dr*dr, axis=1))
        avg_length = np.zeros(n, dtype=sample.dtype)
        for i in cell_indices:
            j = cell_indices[i]
            if count_threshold[i] < count[i]:
                avg_length[j] = np.median(displacement_length[j], keepdims=True)
                continue # consider more measurements than wished
            m = 0
            while True:
                m += 1
                I = [ np.arange(max(0, _k - m), min(_k + m + 1, _d)) \
                    for _k, _d in zip(i, dim) ] # _k, _d for PY2
                I = np.meshgrid(*I, indexing='ij')
                I = np.column_stack([ _i.flatten() for _i in I ]) # _i for PY2
                p = np.sum(count[tuple(I.T)])
                if not (p < k + 1 and m < max_n_units):
                    break
            scale[i] = m
            if p < k + 1:
                # `m` has reached `max_n_units`; consider less measurements than wished
                avg_length[j] = np.median(displacement_length[j], keepdims=True)
                continue
            X = []
            L = []
            for _i in I:
                try:
                    _len = displacement_length[cell_indices[tuple(_i)]]
                    _x = cell[tuple(_i)]
                except KeyError:
                    pass
                else:
                    L.append(_len)
                    X.append(_x)
            #X = [ cell[tuple(_i)] for _i in I if tuple(_i) in cell ] # _i for PY2
            X = np.concatenate(X, axis=0)
            L = np.concatenate(L, axis=0)
            x = cell[i]
            # dist with SciPy
            D = cdist(X, x)
            I = np.argsort(D, axis=0)[:k]
            # and without SciPy (better for bigger matrices)
            #D = np.dot(X, x.T)
            #x2 = np.sum(x * x, axis=1, keepdims=True)
            #D -= 0.5 * x2.T
            #X2 = np.sum(X * X, axis=1, keepdims=True)
            #D -= 0.5 * X2
            #D.sort(axis=0)
            #r = D[-(k+1)]
            #r = np.sqrt(-2.0 * r)
            #r[np.isnan(r)] = 0.0 # numeric precision errors => negative square distances => NaN distances
            #
            del D
            avg_length[j] = np.median(L[I], axis=0)
        if verbose:
            print('estimating density: elapsed time {:d} ms'.format(int(round((t0 + time.time() - t) * 1000)))) # int for PY2
        # plot local radius
        #plot = True
        if d == 2 and plot:
            color = (avg_length - rmin) / (rmax - rmin)
            r = self.collapse_below * 0.5 # collapse_below defines a diameter
            cmap = plt.get_cmap('RdPu')
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            for x in np.arange(smin[0], smax[0], unit):
                plt.plot([x, x], [smin[1], smax[1]], 'y-')
            for y in np.arange(smin[1], smax[1], unit):
                plt.plot([smin[0], smax[0]], [y, y], 'y-')
            for i in range(n):
                c = color[i]
                if rmax < avg_length[i]:
                    ax.add_artist(patches.Circle(sample[i], r * 4, \
                        color=cmap(0.0), alpha=0.1))
                elif avg_length[i] < rmin - np.spacing(1):
                    ax.add_artist(patches.Circle(sample[i], r * 0.25, \
                        color=cmap(1.0)))
                elif rmin + np.spacing(1) < avg_length[i]:
                    ax.add_artist(patches.Circle(sample[i], r, \
                        color=cmap(1.0 - c)))
            plt.plot(sample[:,0], sample[:,1], 'k+')
            for i, x in cell.items():
                center = smin + (np.asarray(list(i)) + 0.5) * unit
                count = x.shape[0]
                if count < knn + 1:
                    color = 'm'
                    txt = "{:d}\n{:d}".format(count, scale.get(i, 0))
                else:
                    color = 'c'
                    txt = str(count)
                plt.text(center[0], center[1], txt, color=color, \
                    horizontalalignment='center', verticalalignment='center')
            try:
                plt.show()
            except AttributeError: # window has been closed; go on
                pass
        return avg_length


    def collapse(self):
        for n in list(self.iter_nodes()):
            if self.has_node(n):
                neighbors = [ neighbor for neighbor in self.iter_neighbors(n)
                            if n < neighbor ]
                if neighbors:
                    dist = la.norm(self.get_weight(n) - \
                        np.vstack([ self.get_weight(n)
                            for n in neighbors ]), axis=1)
                    k = np.argmin(dist)
                    neighbor = neighbors[k]
                    if dist[k] < self.collapse_threshold(n, neighbor):
                        #print((n, neighbor))
                        self.collapse_nodes(n, neighbor)

    def collapse_nodes(self, n1, n2):
        self.disconnect(n1, n2)
        w1 = self.get_weight(n1)
        w2 = self.get_weight(n2)
        self.set_weight(n1, (w1 + w2) / 2)
        for e2, n in list(self.iter_edges_from(n2)):
            a2 = self.get_edge_attr(e2, 'age')
            e1 = self.find_edge(n1, n)
            if e1 is None:
                self.connect(n1, n, age=a2)
            else:
                a1 = self.get_edge_attr(e1, 'age')
                self.set_edge_attr(e1, age=min(a1, a2))
            #self.disconnect(n2, n, e2)
        self.del_node(n2)

