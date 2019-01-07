
from math import *
import numpy as np
import scipy.optimize.linesearch as ls
import time
import scipy.sparse as sparse
from collections import namedtuple, defaultdict, deque
import traceback
import warnings


BFGSResult = namedtuple('BFGSResult', ('x', 'H', 'err', 'f', 'projg', 'cumtime', 'diagnosis'))


def sdfunc(func, yy, ids, components, _sum, h, args=(), kwargs={}, per_component_diff=None):
    """
    Deprecated!

    Compute the derivative of a function.

    Borrowed in large parts from *Numerical Recipes*

    Gradient is calculated as ``(_sum([ func(x+dx_i) for i in ids ]) - _sum([ func(x-dx_i) for i in ids ])) / (2 * h)``
    where ``h = norm(dx_i)``,
    unless `per_component_diff` is defined.
    In this later case, the gradient is ``_sum([ per_component_diff(func(x-dx_i), func(x+dx_i)) for i in ids ]) / (2 * h)``.
    This may moderate numerical precision errors.
    """
    SAFE, CON = 2., 1.4
    CON2 = CON * CON
    ans = []
    any_ok = False
    a = np.zeros((h.size, h.size), dtype=float)
    for d in ids:
        if callable(components):
            js = components(d)
        else:
            js = components
        ans_d, err = None, np.inf
        for i, hh in enumerate(h):
            yy0, yy1 = np.array(yy), np.array(yy) # copies
            yy0[d] -= hh
            yy1[d] += hh
            try:
                if per_component_diff:
                    a[i,0] = a_ip \
                        = _sum([ per_component_diff( \
                                func(j, yy0, *args, **kwargs), \
                                func(j, yy1, *args, **kwargs)) \
                             for j in js ]) \
                        / (2. * hh)
                else:
                    a[i,0] = a_ip \
                        = (_sum([ func(j, yy1, *args, **kwargs) \
                              for j in components ]) \
                        -  _sum([ func(j, yy0, *args, **kwargs) \
                              for j in js ])) \
                        / (2. * hh)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                #traceback.print_exc()
                #raise
                continue
            if i == 0:
                continue
            fac = CON2
            for j in range(1, a.shape[1]):
                a_pp = a[i-1,j-1] # pp stands for [previous, previous]
                a[i,j] = a_ij = (a_ip * fac - a_pp) / (fac - 1.)
                fac *= CON2
                err_ij = max(abs(a_ij - a_ip), abs(a_ij - a_pp))
                if err_ij <= err:
                    err, ans_d = err_ij, a_ij
                a_ip = a_ij
            if SAFE * err <= abs(a[i,i] - a[i-1,i-1]):
                break
        if ans_d is None:
            print('dfunc failed at column {}'.format(d))
            #return None
            ans_d = 0.
        else:
            any_ok = True
        ans.append(ans_d)
    if any_ok:
        return np.array(ans)
    else:
        return None


def minimize_range_sbfgs(ncomponents, minibatch, fun, x0, *args, **kwargs):
    """
    Deprecated!

    Arguments:

        ncomponents (int): number of components; components are expected to be in ``range(ncomponents)``.

        minibatch (callable): takes a component index and returns a set of minibatch 'rows' and
            'columns'.

        ...

        col2rows (callable): keyworded-only.

    See also :func:`minimize_sbfgs`.
    """
    col2rows = kwargs.pop('col2rows', None)
    if minibatch is None:
        rows = np.arange(ncomponents)
        if col2rows:
            rows = (rows, col2rows)
        def __minibatch__(i, x):
            return rows,
    else:
        components = np.arange(ncomponents)
        def _minibatch(i, x):
            _i = i % ncomponents
            if _i == 0:
                np.random.shuffle(components) # in-place
            _i = components[_i]
            return _i, minibatch(_i, x)
        if col2rows:
            def __minibatch__(i, x):
                _i, _x = _minibatch(i, x)
                if _x is None:
                    return None
                elif isinstance(_x, tuple):
                    _rows = _x[0]
                    _rows = (_rows, col2rows)
                    return (_i,_rows)+_x[1:]
                else:
                    _rows = _x
                    _rows = (_rows, col2rows)
                    return (_rows,)
        else:
            def __minibatch__(i, x):
                _i, _x = _minibatch(i, x)
                if _x is None:
                    return None
                else:
                    #return _i, *_x
                    return (_i,)+_x
        if kwargs.get('epoch', None) is None:
            kwargs['epoch'] = ncomponents
        if kwargs.get('gcount', None) is None:
            kwargs['gcount'] = .95
    return minimize_sbfgs(__minibatch__, fun, x0, *args, **kwargs)


def minimize_sbfgs(minibatch, fun, x0, args=(), bounds=None, eta0=1., c=1., l=0., eps=1e-6,
    gtol=1e-10, gcount=10, maxiter=None, maxcor=50, tau=None, epoch=None, step_max=None,
    covariates=None, _sum=np.sum, per_component_diff=None, memory=10,
    diagnosis=None, check_gradient=None, fobj=None, independent_components=False,
    iter_kwarg=None, epoch_kwarg=None, alt_fun=None, error=None, verbose=False, col2rows=None):
    """
    Deprecated!

    Sparse variant of the BFGS minimization algorithm.

    Designed for objective functions in the shape :math:`\sum_{i} f(i, \theta)`
    where :math:`\sum` is sparse, i.e. considers few component indices :math:`i`.

    Arguments:

        minibatch (callable): takes the iteration number and the parameter vector
            and returns component indices (referred to as 'row' indices),
            or row indices and indices in :math:`theta` (referred to as 'column' indices),
            or component index (referred to as 'explicit' component index),
            row indices and column indices.

        fun (callable): partial function :math:`f`;
            takes a component index and the full parameter vector, and returns a float.

        x0 (numpy.ndarray): initial parameter vector.

        args (tuple): extra positional arguments for :func:`fun`.

        bounds (list): list of (lower bound, upper bound) couples.

        eta0 (float): see `eta_max` in :func:`slnsrch`.

        c (float):

        l (float):

        eps (float):

        gtol (float):

        gcount (int or float):
            if `epoch` is defined and `gcount` is float and less than or equal to 1,
            `gcount` is multiplied by `epoch`.

        maxiter (int):

        maxcor (int):

        tau (float):

        epoch (int):

        covariates (2-element tuple or scipy.sparse.csr_matrix):
            parameter association matrix;
            either a (`indices`, `indptr`) couple
            or CSR sparse matrix with attributes `indices` and `indptr`;
            expected if `independent_components` is ``False``.

        _sum (callable): sum function for values returned by `fun`; see also :func:`sdfunc`.

        per_component_diff (callable): see :func:`sdfunc`.

        memory (int): requires ``independent_components=True``.

        diagnosis (callable): called whenever a step has failed;
            takes the parameter vector and all the output arguments from `minibatch` as input arguments
            and optionally returns any output argument;
            the returned objects are collected and stored in the output `BFGSResult` object.

        check_gradient (callable):

        fobj (float): stop when ``f <= fobj``.

        independent_components (bool):

        iter_kwarg (str): keyword for passing the iteration number to :func:`fun`.

        epoch_kwarg (str): keyword for passing the epoch-wise baseline parameter vector.

        alt_fun (callable): alternative function for :func:`fun` that is evaluated on a minibatch
            at the beginning and end of each iteration;
            :func:`alt_fun` does not admit `iter_kwarg` and `epoch_kwarg`.

        error (callable): takes the parameter vector and returns a scalar error measurement.

        verbose (bool): at each iteration print the value of the local objective and its variation;
            if available, uses :func:`alt_fun` instead of :func:`fun`.
    """
    if verbose:
        component_strlen = [0]
        def format_component(_c, _f=True):
            if _c is None:
                return 'f' if _f else ''
            _c = str(_c)
            _strlen = len(_c)
            if component_strlen[0] <= _strlen:
                component_strlen[0] = _strlen
                _sp = ''
            else:
                _sp = ' ' * (component_strlen[0] - _strlen)
            return 'f({}{})'.format(_sp, _c)
        def msg0(_i, _c, _f, _dg):
            _i = str(_i)
            return 'At iterate {}{}\t{}= {}{:E} \tproj g = {:E}\n'.format(
                ' ' * max(0, 3 - len(_i)), _i,
                format_component(_c),
                ' ' if 0 <= _f else '', _f, _dg)
        def msg1(_i, _c, _f0, _f1, _dg):
            _i = str(_i)
            _df = _f1 - _f0
            return 'At iterate {}{}\t{}= {}{:E} \tdf= {}{:E} \tproj g = {:E}\n'.format(
                ' ' * max(0, 3 - len(_i)), _i,
                format_component(_c),
                ' ' if 0 <= _f1 else '', _f1,
                ' ' if 0 <= _df else '', _df, _dg)
        def msg2(_i, _c, *_args):
            _c = format_component(_c)
            if len(_args) == 3 and isinstance(_args[1], str) and isinstance(_args[-1], (tuple, list)):
                _exc_type, _exc_msg, _exc_args = _args
                try:
                    _exc_msg = _exc_msg.replace('%s', '{}').format(*_exc_args)
                except (SystemExit, KeyboardInterrupt):
                    raise
                except:
                    pass
                else:
                    _args = (_exc_type, _exc_msg)
            if _c:
                _args = (_c,) + _args
            msg = ''.join(('At iterate {}{}\t', ':  '.join(['{}'] * len(_args)), '\n'))
            _i = str(_i)
            return msg.format(' ' * max(0, 3 - len(_i)), _i, *_args)
        cumt = 0.
        fx_history = []
        t0 = time.time()
    proj_history = []
    # precompute arguments for dfunc
    h = 1e-8
    CON = 1.4
    dfunc_h = h * ((1./CON) ** np.arange(maxcor))
    #dfunc_h = np.outer(h, (1./CON) ** np.arange(maxcor))
    if per_component_diff is None and _sum in (sum, np.sum, np.mean):
        per_component_diff = lambda a, b: b - a
    if diagnosis:
        diagnoses = defaultdict(list)
    #
    fargs = args
    fkwargs = {}
    if iter_kwarg:
        fkwargs[iter_kwarg] = 0
    if epoch_kwarg:
        if not epoch:
            raise ValueError('`epoch` undefined')
        fkwargs[epoch_kwarg] = x0
    def f(_x, _rows):
        return _sum([ fun(_j, _x, *fargs, **fkwargs) for _j in _rows ])
    _local_f = {}
    _total_f = [0] # mutable
    def eval_f_and_update(_x, _rows):
        _f = []
        for _j in _rows:
            _fj = fun(_j, _x, *fargs, **fkwargs)
            _fj_old = _local_f.get(_j, 0)
            _local_f[_j] = _fj
            _total_f[0] += _fj - _fj_old
            _f.append(_fj)
        return _sum(_f)
    if alt_fun:
        def alt_f(_x, _rows):
            return _sum([ alt_fun(_j, _x, *fargs) for _j in _rows ])
    #
    t = 0
    while True:
        rows = minibatch(t, x0)
        if rows is None:
            t += 1
        else:
            break
    twoways = False
    component = None
    explicit_components = False
    if isinstance(rows, tuple):
        if rows[1:]:
            twoways = True
            if rows[2:]:
                try:
                    component, rows, cols = rows
                except ValueError:
                    raise ValueError('too many values to unpack from minibatch')
                explicit_components = True
            else:
                rows, cols = rows
            def df(_x, _rows, _cols):
                return sdfunc(fun, _x, _cols, _rows, _sum, dfunc_h, fargs, fkwargs, per_component_diff)
            if alt_fun:
                def alt_df(_x, _rows, _cols):
                    return sdfunc(alt_fun, _x, _cols, _rows, _sum, dfunc_h, fargs, {}, per_component_diff)
        else:
            rows, = rows
    if isinstance(rows, tuple):
        rows_f, rows_df = rows
    else:
        rows_f = rows_df = rows
    #assert isinstance(rows[0], (int, np.int_))
    if twoways:
        if isinstance(cols, tuple):
            cols_f, cols_df = cols
        else:
            cols_f = cols_df = cols
        g = df(x0, rows_df, cols_df)
    else:
        cols = cols_f = cols_df = None
        def df(_x, _rows):
            return sdfunc(fun, _x, range(_x.size), _rows, _sum, dfunc_h, fargs, fkwargs, per_component_diff)
        if alt_fun:
            def alt_df(_x, _rows):
                return sdfunc(alt_fun, _x, range(_x.size), _rows, _sum, dfunc_h, fargs, {}, per_component_diff)
        g = df(x0, rows_df)
    if g is None:
        raise RuntimeError('gradient calculation failed')
    p = -eps * g
    # project descent direction onto descent subspace
    if cols_df is not cols_f:
        _gs2ds = [ (cols_df==_c).nonzero()[0][0] for _c in cols_f ] # gradient subspace to descent subspace
        p = p[ _gs2ds ]
    # precompute argument(s) for slnsrch
    if step_max is None:
        pass
        #STPMX = 500.
        #step_max = STPMX * max(np.sqrt(np.sum(p * p)), p.size) # assume p.size is constant
    ls_maxiter = None#10
    # bounds
    if bounds:
        lower_bound, upper_bound = np.full_like(x0, -np.inf), np.full_like(x0, np.inf)
        any_lower_bound = any_upper_bound = False
        for i, _bs in enumerate(bounds):
            _lb, _ub = _bs
            if _lb not in (None, -np.inf, np.nan):
                any_lower_bound = True
                lower_bound[i] = _lb
            if _ub not in (None, np.inf, np.nan):
                any_upper_bound = True
                upper_bound[i] = _ub
        if not any_lower_bound:
            lower_bound = None
        if not any_upper_bound:
            upper_bound = None
        if any_lower_bound or any_upper_bound:
            bounds = (lower_bound, upper_bound)
        else:
            bounds = None
        if bounds and c < 1:
            warnings.warn('c < 1; bounds may be violated')
    _sanity_check(x0, bounds)
    #
    eta, p = subspace_search(f, rows_f, cols_f, x0, p, g, df, rows_df, bounds=bounds, eps=eps,
            eta0=eta0, c=c, ls_maxiter=ls_maxiter, step_max=step_max, f0=eval_f_and_update)
    #eta = slnsrch(f, rows_f, cols_f, x0, p, g, eta0, ls_maxiter, step_max, bounds, c)
    s = eta / c * p
    if cols is None:
        x = x0 + s
        h = df(x, rows_df)
    else:
        x = np.array(x0) # copy
        x[cols_f] += s
        h = df(x, rows_df, cols_df)
    _sanity_check(x, bounds)
    if h is None:
        raise RuntimeError('gradient calculation failed')
    y = h - g
    # express descent vector in gradient subspace
    if cols_df is not cols_f:
        _s = s
        s = np.zeros_like(y)
        s[_gs2ds] = _s
    if l:
        y += l * s
    proj = np.dot(s, y)
    proj_history.append(proj)
    if _local_f:
        if verbose:
            suspend = fobj is None
            if suspend:
                cumt += time.time() - t0
        eval_f_and_update(x, rows_f)
        if verbose:
            f1 = _total_f[0]
            print(msg0(t, component, f1, proj))
            fx_history.append(f1)
            if suspend:
                t0 = time.time()
    if error is None:
        errs = None
    else:
        errs = [error(x)]
    if epoch:
        if isinstance(gcount, float) and gcount <= 1:
            gcount = float(epoch) * gcount
    # make B sparse and preallocate the nonzeros
    gamma = proj / np.dot(y, y)
    if independent_components and memory:
        B0 = deque([], memory)
        rho = 1. / proj
        B0.appendleft((s, y, rho, gamma))
    else:
        memory = False
        B0 = gamma
    if independent_components:
        i = component if explicit_components else tuple(rows_f)
        B = {}
        B[i] = B0
    elif twoways and (sparse.issparse(covariates) or covariates):
        if isinstance(covariates, tuple):
            indices, indptr = covariates
        else:
            indices, indptr = covariates.indices, covariates.indptr
        B = sparse.csr_matrix((np.zeros(indices.size, dtype=x0.dtype), indices, indptr),
            shape=(x0.size, x0.size))
        B[np.ix_(cols_df, cols_df)] = B0
        covariates = True
    elif np.isscalar(B0):
        B = np.diag(np.full(x.size, B0))
    elif B0.shape == ( x0.size, x0.size ):
        B = B0
    else:
        raise NotImplementedError('fixme')
    x2 = x0
    k1 = k2 = 0
    resolution = None
    while True:
        t += 1
        if maxiter and maxiter < t:
            break
        try:
            if iter_kwarg:
                fkwargs[iter_kwarg] = t
            #if epoch_kwarg and t % epoch == 0:
            #    fkwargs[epoch_kwarg] = x

            ## step (pre-a) ##
            bt = minibatch(t, x)
            if bt is None:
                continue
            if twoways:
                if explicit_components:
                    component, rows, cols = bt
                else:
                    rows, cols = bt
                if isinstance(cols, tuple):
                    cols_f, cols_df = cols
                else:
                    cols_f = cols_df = cols
            else:
                if isinstance(bt, tuple):
                    rows, = bt
                else:
                    rows = bt
                    bt = (rows,)
            if isinstance(rows, tuple):
                rows_f, rows_df = rows
            else:
                rows_f = rows_df = rows
            if cols is None:
                g = df(x, rows_df)
            else:
                g = df(x, rows_df, cols_df) # computation of `g` is part of step (a)

            #
            if g is None or np.all(g == 0):
                x = x2
                if verbose:
                    print(msg2(t, component, 'GRADIENT CALCULATION FAILED (t)'))
                if diagnosis:
                    _d = diagnosis(x, *bt)
                    if _d is not None:
                        _d = ('GRADIENT CALCULATION FAILED', _d)
                        if explicit_components:
                            _d = (component,) + _d
                        diagnoses[t].append(_d)
                continue
                resolution = 'GRADIENT CALCULATION FAILED (t)'
                break
            #if verbose:
            #    if alt_fun:
            #        f0 = alt_f(x, rows_f)
            #    else:
            #        try:
            #            f0 = f(x, rows_f)
            #        except ValueError as e:
            #            f0 = e

            ## step (a) ##
            if independent_components:
                i = component if explicit_components else tuple(rows_f)
                try:
                    Bi = B[i]
                except KeyError:
                    Bi = None
            elif cols is None:
                Bi = B
            else:
                Bi = B[np.ix_(cols_df, cols_df)]
                if covariates:
                    if sparse.issparse(Bi):
                        firsttime = Bi.count_nonzero() == 0
                    else:
                        firsttime = np.all(Bi == 0)
                    if firsttime:
                        Bi = None
            if Bi is None:
                p = -eps * g
            elif memory:
                _B = []
                p = np.array(g) # copy
                for _s, _y, _rho, _ in Bi: # from k-1 to k-m
                    _alpha = _rho * np.dot(_s, p)
                    p -= _alpha * _y
                    _B.append((_s, _y, _rho, _alpha))
                _, _, _, gamma = Bi[0]
                p *= -gamma # gamma_{k-1}
                for _s, _y, _rho, _alpha in _B[::-1]: # from k-m to k-1
                    _beta = _rho * np.dot(_y, p)
                    p += (_alpha - _beta) * _s
                print((g, p, _s, _y, _rho, gamma, _alpha, _beta))
            else:
                p = -np.dot(Bi, g)
                if sparse.issparse(p):
                    p = p.todense()
                if p.shape[1:]:
                    p = np.ravel(p)
                if 0 <= np.sum(g * p):
                    # line search will fail; make a small update instead
                    if verbose:
                        print(msg2(t, component, 'IGNORING THE HESSIAN MATRIX'))
                    p = -eps * g

            # project descent direction onto descent subspace
            if cols_df is not cols_f:
                _gs2ds = [ (cols_df==_c).nonzero()[0][0] for _c in cols_f ] # gradient subspace to descent subspace
                p = p[ _gs2ds ]

            ## step (b) ##
            #epoch = ncomps
            #if epoch:
            #    if t % epoch == 1:
            #        eta = tau / (tau + float(t / epoch)) * eta0
            #else:
            #    eta = tau / (tau + float(t)) * eta0
            eta, p = subspace_search(f, rows_f, cols_f, x, p, g, df, rows_df, B=Bi, bounds=bounds, eps=eps,
                    eta0=eta0, c=c, ls_maxiter=ls_maxiter, step_max=step_max, f0=eval_f_and_update)
            #eta = slnsrch(f, rows_f, cols, x, p, g, eta0, ls_maxiter, step_max, bounds)
            if eta is None or (bounds is None and eta == 0):
                if diagnosis:
                    _d = diagnosis(x, *bt)
                    if _d is not None:
                        _d = ('LINE SEARCH FAILED', _d)
                        if explicit_components:
                            _d = (component,) + _d
                        diagnoses[t].append(_d)
                #x = x2
                if verbose:
                    print(msg2(t, component, 'LINE SEARCH FAILED'))
                    #print(B[component])
                continue
                #resolution = 'LINE SEARCH FAILED'
                #break
            elif eta:
                if tau:
                    if epoch:
                        eta *= 1. - tau * fmod(t, epoch) / epoch
                        #eta *= tau / (tau + float(t / epoch))
                    else:
                        eta *= tau / (tau + float(t))
            else:#if bounds is not None and eta == 0
                if verbose:
                    print(msg2(t, component, 'SATURATED CONSTRAINT'))
                continue

            ## step (c) ##
            s = eta / c * p

            _update_B = True

            ## step (d) ##
            if cols is None:
                x1 = x + s
            else:
                x1 = np.array(x)
                x1[cols_f] += s
            _sanity_check(x1, bounds)
            #if _fobj and _fobj(x1):
            #    resolution = 'CONVERGENCE: F* <= FOBJ  (f*= approximated total)'
            #    break
            if cols is None:
                h = df(x1, rows_df)
            else:
                h = df(x1, rows_df, cols_df) # computation of `h` is part of step (e)

            #
            if h is None:
                if verbose:
                    print(msg2(t, component, 'GRADIENT CALCULATION FAILED (t+1)'))
                    if alt_fun:
                        print('calling sdfunc with alt_fun:')
                        if twoways:
                            h = alt_df(x1, rows_df, cols_df)
                        else:
                            h = alt_df(x1, rows_df)
                        assert h is None
                if diagnosis:
                    _d = diagnosis(x1, *bt)
                    if _d is not None:
                        _d = ('GRADIENT CALCULATION FAILED', _d)
                        if explicit_components:
                            _d = (component,) + _d
                        diagnoses[t].append(_d)
                # option 1. step back
                x = x2
                continue
                # option 2. admit the update and continue, just skip the inverse Hessian calculations
                _update_B = False
                # option 3. stop here
                #resolution = 'GRADIENT CALCULATION FAILED (t+1)'
                #break
            elif np.allclose(g, h):
                if np.all(s == 0):
                    warnings.warn('null update')
                if diagnosis:
                    _d = diagnosis(x1, *bt)
                    if _d is not None:
                        _d = ('NO CHANGE IN THE GRADIENT', _d)
                        if explicit_components:
                            _d = (component,) + _d
                        diagnoses[t].append(_d)
                #if not l:
                #    warnings.warn('no change in the gradient; setting `l` greater than 0')
                #    l = .1
                if verbose:
                    print(msg2(t, component, 'NO CHANGE IN THE GRADIENT'))
                _update_B = False
                #continue # B cannot be properly updated

            ## step (e) ##
            y = h - g

            # express descent vector in gradient subspace
            if cols_df is not cols_f:
                _s = s
                s = np.zeros_like(y)
                s[_gs2ds] = _s

            if l:
                y += l * s

            #
            proj = np.dot(s, y)
            proj_history.append(proj)

            if proj <= 0:
                if verbose:
                    print(msg2(t, component, 'PROJ G <= 0'))
                continue

            x2, x = x, x1

            if error is not None:
                errs.append(error(x))

            if epoch:
                if gtol is not None and proj < gtol:
                    k1 += 1
                if t % epoch == 0:
                    if gcount <= k1:
                        resolution = 'CONVERGENCE: PROJ G < GTOL'
                        break
                    else:
                        k1 = 0
            else:
                if gtol is not None and abs(proj) < gtol:
                    k1 += 1
                    if gcount <= k1:
                        resolution = 'CONVERGENCE: PROJ G < GTOL'
                        break
                else:
                    k1 = 0

            if not _update_B:
                continue

            # step (f)
            if Bi is None:
                if memory:
                    Bi = deque([], memory)
                else:
                    Bi = proj / np.dot(y, y)
                if independent_components:
                    B[i] = Bi
                elif twoways:
                    B[np.ix_(cols_df, cols_df)] = Bi
                else:
                    B[...] = Bi

            #
            if _local_f:
                if verbose and suspend:
                    cumt += time.time() - t0
                f0 = _total_f[0]
                eval_f_and_update(x, rows_f)
                f1 = _total_f[0]
                if verbose:
                    print(msg1(t, component, f0, f1, proj))
                    fx_history.append((f0, f1))
                    if suspend:
                        t0 = time.time()
                if fobj is not None and f1 <= fobj:
                    resolution = 'CONVERGENCE: F <= FOBJ'
                    break

            ## step (g) ##
            if proj <= 0:
                continue
            rho = 1. / proj

            if memory:
                gamma = proj / np.dot(y, y)
                Bi.appendleft((s, y, rho, gamma))
                continue

            #
            # no first time or small step computation for `q`
            q = -np.dot(Bi, y)
            if sparse.issparse(q):
                q = q.todense()
            if q.shape[1:]:
                q = np.ravel(q)

            ## step (h) ##
            # B = (I - rho s y.T) B ( I - rho y s.T) + c rho s s.T
            phi = c - rho * np.dot(y, q) # rho s y.T B y s.T rho + c rho s s.T = rho phi s s.T
            deltaB = rho * (np.outer(q, s) + np.outer(s, q) + phi * np.outer(s, s))
            if independent_components:
                B[i] = B[i] + deltaB
            elif twoways:
                B[np.ix_(cols_df, cols_df)] += deltaB
            else:
                B += deltaB

        except KeyboardInterrupt:
            #x = x2
            # some variables may not be valid, e.g. t0
            resolution = 'INTERRUPTED'
            break

    if not resolution:
        resolution = 'MAXIMUM ITERATION REACHED'
    if verbose:
        cumt += time.time() - t0
        print('           * * *\n\n{}\n'.format(resolution))
        minute = floor(cumt / 60.)
        second = cumt - minute * 60.
        if minute:
            print('Elapsed time = {:d}m{:.3f}s\n'.format(minute, second))
        else:
            print('Elapsed time = {:.3f}s\n'.format(second))
        args = [fx_history, proj_history, cumt]
    else:
        args = [None, proj_history, None]
    if not (diagnosis and diagnoses):
        diagnoses = None
    return BFGSResult(x, B, errs, diagnoses, *args)



def wolfe_line_search(f, x, p, g, subspace=None, args_f=(), args_g=None, args=None, f0=None, g0=None,
        bounds=None, weight_regul=None, step_regul=None,
        eta_max=1., iter_max=10, step_max=None, c0=.5, c1=1e-4, c2=.9, c3=.9, c4=.1, c5=1e-10,
        armijo_max=None):
    """
    Wolfe line search along direction `p` possibly restricted to a subspace.

    If defined, argument `subspace` is passed to gradient function `g` as last positional argument,
    unless already present in `args_g`.

    `args` is an alias for `args_f`.
    If `args_g` is ``None``, then it falls back to `args_f`.

    .. code-block:: math

        f(x_k + eta_k p_k) <= f(x_k) + c1 eta_k p_k^T g(x_k) # Armijo rule
        c2 * p_k^T g(x_k) <= -p_k^T g(x_k + eta_k p_k) <= -c3 p_k^T g(x_k) # weak (c2=None) or strong (c2=c3) Wolfe condition
        c4 eta_k <= eta_{k+1} <= c0 eta_k
        eta_min = c5 eta_max

    """
    if args is not None:
        if args is args_f:
            pass
        elif args_f is ():
            args_f = args
        else:
            raise ValueError('both `args` and `args_f` are defined')
    if not isinstance(args_f, (tuple, list)):
        args_f = (args_f,)
    if args_g is None:
        args_g = args_f
    elif not isinstance(args_g, (tuple, list)):
        args_g = (args_g,)
    if f0 is None:
        f0 = f(x, *args_f)
    if subspace is not None and subspace not in args_g:
        args_g = list(args_g)
        args_g.append(subspace)
    if g0 is None:
        g0 = g(x, *args_g)
    if subspace is None:
        x0 = x
    else:
        x0 = x[subspace]
    x = np.array(x)
    #
    slope = np.dot(p, g0)
    if 0 <= slope:
        return None#raise ValueError('not a descent direction')
    if step_max:
        norm = np.max(np.abs(p))
        if step_max < norm * eta_max:
            eta_max *= step_max / norm
    f_prev = f0
    if weight_regul:
        f0 += weight_regul * np.dot(x0, x0)
    if bounds:
        lb, ub = bounds
        if subspace is not None:
            if lb is not None and len(lb) == x.size:
                lb = lb[subspace]
            if ub is not None and len(ub) == x.size:
                ub = ub[subspace]
    else:
        assert bounds is None
        bounds = lb = ub = None
        _p, _slope = p, slope
    if c5 is None:
        eta_min = -np.inf
    else:
        eta_min = eta_max * c5
    eta = eta_prev = eta_max
    #eta_hist, f_hist, norm_p_hist, df_hist, armijo_hist = [], [], [], [], []
    last_valid_eta = None
    last_valid_x = np.array(x0)
    i = 0
    while True:
        #eta_hist.append(eta)
        if eta < eta_min:
            print('eta < eta_min={}'.format(eta_min))
            break
        _x = x0 + eta * p
        if bounds:
            _x = _proj(_x, lb, ub)
            _p = _x - x0
        else:
            _slope = eta * slope
        #norm_p_hist.append(sqrt(np.dot(_p, _p)))
        if subspace is None:
            x = _x
        else:
            x[subspace] = _x
        try:
            fx = f(x, *args_f)
        except ValueError:
            fx = None
        #f_hist.append(fx)
        if fx is None:
            eta *= c0
        else:
            if weight_regul:
                fx += weight_regul * np.dot(_x, _x)
            if step_regul:
                fx += step_regul + np.dot(_p, _p)
            df = fx - f0
            #df_hist.append(df)
            if bounds:
                _slope = np.dot(_p, g0)
            #armijo_hist.append(c1 * _slope)
            if _slope < 0:
                armijo_threshold = c1 * _slope
                if armijo_max and armijo_threshold < -armijo_max:
                    armijo_threshold = -armijo_max
                armijo_rule = df <= armijo_threshold
                #print((f0, eta, fx, armijo_rule))
                if armijo_rule:
                    if c2 is None and c3 is None:
                        # do not test Wolfe condition; stop here
                        p = _x - x0
                        return p
                    try:
                        gx = g(x, *args_g)
                    except ValueError:
                        gx = None
                    if gx is not None:
                        new_slope = np.dot(_p, gx)
                        if c3 is not None:
                            curvature_condition = c3 * _slope <= new_slope # new_slope less negative than slope (can be positive!)
                        else:
                            curvature_condition = True
                        if curvature_condition and c2 is not None:
                            curvature_condition = new_slope <= -c2 * _slope
                        #print((_slope, new_slope, curvature_condition))
                        if curvature_condition:
                            p = _x - x0
                            return p
            if eta == eta_max:
                eta1 = -slope / (2. * (df - slope))
            else:
                df_prev = f_prev - f0
                rhs1 = df - eta * slope
                rhs2 = df_prev - eta_prev * slope
                eta2 = eta_prev
                rhs1 /= eta * eta
                rhs2 /= eta2 * eta2
                a = (rhs1 - rhs2) / (eta - eta2)
                b = (eta * rhs2 - eta2 * rhs1) / (eta - eta2)
                if a == 0:
                    eta1 = -slope / (2. * b)
                else:
                    discr = b * b - 3. * a * slope
                    if discr < 0:
                        eta1 = eta * c0
                    elif b <= 0:
                        eta1 = (np.sqrt(discr) - b) / (3. * a)
                    else:
                        eta1 = -slope / (np.sqrt(discr) + b)
            eta_prev, f_prev = eta, fx
            eta = max(c4 * eta, min(eta1, c0 * eta)) # c3 * eta{k} <= eta{k+1} <= c0 * eta{k}
        i += 1
        if iter_max and i == iter_max:
            #print('iter_max reached')
            break
    #if True:
    #    print((eta_hist, norm_p_hist, f_hist, df_hist, armijo_hist))
    return None # line search failed

def slnsrch(f, rows, cols, x, p, g, eta_max=1., iter_max=10, step_max=None, bounds=None, c=1.,
        fold=None, active_set=False):
    """
    Deprecated!

    Line search.

    Borrowed in large parts from *Numerical Recipes*.
    """
    #if cols is None and bounds is None:
    #    eta, f_count, fnew =  ls.line_search_armijo(f, x, p, g, f(x, rows), (rows,)) # crashes
    #    return eta
    #
    TOLX = 1e-8
    ALF = 1e-4
    #
    slope = np.sum(g * p)
    if 0 <= slope:
        #raise ValueError('positive slope: {}'.format(slope))
        return None
    if step_max:
        norm = np.sqrt(np.sum(p * p))
        if step_max < norm * eta_max:
            eta_max *= step_max / norm
    if fold is None:
        fold = f(x, rows)
    f2 = fold
    if cols is None:
        xold = x
    else:
        xold = x[cols]
    xnew = np.array(x)
    if bounds:
        lb, ub = bounds
        if cols is not None:
            lb, ub = None if lb is None else lb[cols], None if ub is None else ub[cols]
    else:
        assert bounds is None
        bounds = lb = ub = None
    eta_min = min(eta_max * 1e-3, TOLX / np.max(np.abs(p) / np.maximum(np.abs(xold), 1.)))
    eta = eta2 = eta_max
    last_valid_eta = None
    last_valid_x = np.array(x)
    i = 0
    while True:
        #if i == 0:
        #    eta = _update(xold, eta, p, bounds, c)
        #    if eta == 0:
        #        return eta
        _x = xold + eta * p
        _x = _proj(_x, lb, ub)
        if cols is None:
            xnew = _x
        else:
            xnew[cols] = _x
        try:
            fnew = f(xnew, rows)
        except ValueError:
            #print((xold, xnew[cols], i, bounds, eta, p))
            #raise
            eta /= 2.
        else:
            last_valid_eta = eta
            last_valid_x = _x
            if eta < eta_min:
                #print('eta < eta_min={}'.format(eta_min))
                break
            df = fnew - fold
            if df <= ALF * eta * slope:
                #print('df <= ALF * eta * slope = {} * {} * {}'.format(ALF, eta, slope))
                break
            if eta == eta_max:
                eta1 = -slope / (2. * (df - slope))
                assert 0 <= eta1
            else:
                rhs1 = df - eta * slope
                rhs2 = f2 - fold - eta2 * slope
                rhs1 /= eta * eta
                rhs2 /= eta2 * eta2
                a = (rhs1 - rhs2) / (eta - eta2)
                b = (eta * rhs2 - eta2 * rhs1) / (eta - eta2)
                if a == 0:
                    eta1 = -slope / (2. * b)
                    assert 0 <= eta1
                else:
                    discr = b * b - 3. * a * slope
                    if discr < 0:
                        eta1 = eta / 2.
                    elif b <= 0:
                        eta1 = (np.sqrt(discr) - b) / (3. * a)
                        assert 0 <= eta1
                    else:
                        eta1 = -slope / (np.sqrt(discr) + b)
                        assert 0 <= eta1
                eta1 = min(eta1, eta / 2.)
            eta2, f2 = eta, fnew
            eta = max(eta1, .1 * eta)
        i += 1
        if iter_max and i == iter_max:
            #print('iter_max reached')
            break
    if last_valid_eta:
        pnew = _x - xold
        if active_set:
            return pnew, _active_set(_x, lb, ub, cols)
        else:
            return pnew
    else:
        if active_set:
            return None, []
        else:
            return None

def _update(x, eta0, p, bounds, c):
    """Deprecated!"""
    if bounds:
        lower_bound, upper_bound = bounds
        eta_lb = eta_ub = None
        ok = []
        if lower_bound:
            bound = [ (i, b) for i, b in enumerate(lower_bound) if b is not None ]
            if bound:
                bounded, bound = zip(*bound)
                bounded, bound = np.array(bounded), np.array(bound)
                eta = (bound - x[bounded]) / p[bounded]
                ok = bounded[0 <= eta].tolist() # in the case `upper_bound` is defined
                if ok:
                    eta_lb = np.min(eta[0 <= eta]) * c
                else:
                    ok = []
        if upper_bound:
            bounded, bound = [], []
            for i, b in enumerate(upper_bound):
                if not (b is None or i in ok): # here `ok` must be a sequence
                    bounded.append(i)
                    bound.append(b)
            if bound:
                bounded, bound = np.array(bounded), np.array(bound)
                eta = (bound - x[bounded]) / p[bounded]
                eta = eta[0 <= eta]
                if eta.size:
                    eta_ub = np.min(eta) * c
                    ok = True
        if ok:
            if eta_lb is None:
                eta = eta_ub
            elif eta_ub is None:
                eta = eta_lb
            else:
                eta = min(eta_lb, eta_ub)
            #if eta < eta0:
            #    print('bound met at eta= {}'.format(eta))
            return min(eta0, eta)
        else:
            return eta0
    else:
        return eta0


def _proj(x, lb, ub):
    if lb is not None:
        x = np.maximum(lb, x)
    if ub is not None:
        x = np.minimum(x, ub)
    return x


def _active_set(x, lb, ub, cols=None):
    """Deprecated!"""
    _as = set()
    if lb is not None:
        _lb = np.isclose(x, lb).nonzero()
        _as += set(_lb.tolist())
    if ub is not None:
        _ub = np.isclose(x, ub).nonzero()
        _as += set(_ub.tolist())
    if _as:
        _as = np.array(list(_as))
        if cols:
            _as = cols[_as]
    return _as


def subspace_search(f, rows_f, cols, x, p, g, df, rows_df=None, B=None,
        eta0=1., c=1., ls_maxiter=None, f0=None, eps=None, bounds=None, **kwargs):
    """Deprecated!"""
    assert c == 1
    #if B is None and eps is not None:
    #    kwargs['c2'] = 1. - eps
    if bounds is not None:
        kwargs['c2'] = None # no Wolfe condition
    fold = f0(x, rows_f) if f0 else None
    rows_pnew = wolfe_line_search(f, x, p, df, cols, rows_f, rows_df, f0=fold,# g0=g,
            eta_max=eta0, iter_max=ls_maxiter, bounds=bounds, **kwargs)
    if pnew is None:
        return None, p
    #elif bounds:
    #    pnew = wolfe_line_search(f, rows_f, cols, x, pnew, df, f0=fold, rows_g=rows_df,# g0=g,
    #        eta_max=1., iter_max=ls_maxiter, **kwargs)
    return 1., pnew

def _sanity_check(x, bounds):
    """Deprecated!"""
    if bounds:
        lb, ub = bounds
        if lb is not None:
            fail, = np.nonzero(x < lb - 1e-8)
            if fail.size:
                for i in fail:
                    print('at column {}: {} < {}'.format(i, x[i], lb[i]))
                raise ValueError
        assert ub is None



def minimize_sparse_bfgs(fun, x0, component, covariate, gradient_subspace, descent_subspace,
        args=(), bounds=None, _sum=np.sum, gradient_sum=None, gradient_covariate=None,
        memory=10, eps=1e-6, ftol=1e-6, gtol=1e-10, s_scale=1.,
        max_iter=None, regul=None, regul_decay=1e-5, ls_regul=None, ls_step_max=None, ls_iter_max=None,
        ls_armijo_max=None, ls_wolfe=None,
        independent_components=True, newton=True, verbose=False):
    """
    Let the objective function :math:`f(x) = \sum_{i \in C} f_{i}(x) \forall x in \Theta`
    be a linear function of sparse components :math:`f_{i}` such that
    :math:`\forall j \notin G_{i}, \forall x in \Theta, {\partial f_{i}}{\partial x_{j}}(x) = 0`.

    Let the components also covary sparsely:
    :math:`\forall i in C,
    \exists C_{i} \subset C | i \in C_{i},
    \exists D_{i} \subset G_{i},
    \forall j in D_{i},
    \forall x in \Theta,
    \frac{\partial f}{\partial x_{j}}(x) =
    \sum_{i' \in C_{i}} \frac{\partial f_{i'}}{\partial x_{j}}(x)`.

    We may additionally need that
    :math:`\forall i \in C, D_{i} = \bigcap_{i' \in C_{i}} G_{i'}`,
    :math:`\bigcup_{i \in C} D_{i} = J` and :math:`D_{i} \cap D_{j} = \emptyset \forall i, j \in C^{2}`
    with :math:`J` the indices of parameter vector :math:`x = \lvert x_{j}\rvert_{j \in J}`
    (to be checked).

    At iteration :math:`k`, let choose component :math:`i`
    and minimize :math:`f` wrt parameters :math:`\{x_{j} | j \in D_{i}\}`.

    Compute gradient :math:`g_{i}(x) = \lvert\frac{\partial f}{\partial x_{j}}(x)\rvert_{j} =
    \sum_{i' \in C_{i}} \lvert\frac{\partial f_{i'}}{\partial x_{j}}(x) \rvert_{j}`
    with (again) :math:`g_{i}(x)\rvert_{j} = 0 \forall j \notin G_{i}`.

    Perform a Wolfe line search along descent direction restricted to subspace :math:`D_{i}`.
    Update :math:`x` and compute the gradient again.

    The inverse Hessian matrix must also be updated twice: before and after the update.

    x0 is modified inplace.
    """
    _all = None
    if not callable(fun):
        raise TypeError('fun is not callable')
    if not isinstance(x0, np.ndarray):
        raise TypeError('x0 is not a numpy.ndarray')
    if not callable(component):
        raise TypeError('component is not callable')
    if not callable(covariate):
        raise TypeError('covariate is not callable')
    if gradient_subspace is None:
        gradient_subspace = lambda i: _all
    elif not callable(gradient_subspace):
        raise TypeError('gradient_subspace is not callable')
    if descent_subspace is None:
        descent_subspace = lambda i: _all
    elif not callable(descent_subspace):
        raise TypeError('descent_subspace is not callable')
    if not callable(_sum):
        raise TypeError('_sum is not callable')
    if gradient_sum is None:
        gradient_sum = _sum
    elif not callable(gradient_sum):
        raise TypeError('gradient_sum is not callable')
    # logging
    if verbose:
        component_strlen = [0]
        def format_component(_c, _f=True):
            if _c is None:
                return 'f' if _f else ''
            _c = str(_c)
            _strlen = len(_c)
            if component_strlen[0] <= _strlen:
                component_strlen[0] = _strlen
                _sp = ''
            else:
                _sp = ' ' * (component_strlen[0] - _strlen)
            return 'f({}{})'.format(_sp, _c)
        def msg0(_i, _c, _f, _dg):
            _i = str(_i)
            return 'At iterate {}{}\t{}= {}{:E} \tproj g = {:E}\n'.format(
                ' ' * max(0, 3 - len(_i)), _i,
                format_component(_c),
                ' ' if 0 <= _f else '', _f, _dg)
        def msg1(_i, _c, _f0, _f1, _dg=None):
            _i = str(_i)
            _df = _f1 - _f0
            return 'At iterate {}{}\t{}= {}{:E} \tdf= {}{:E}{}\n'.format(
                ' ' * max(0, 3 - len(_i)), _i,
                format_component(_c),
                ' ' if 0 <= _f1 else '', _f1,
                ' ' if 0 <= _df else '', _df,
                '' if _dg is None else ' \tproj g = {:E}'.format(_dg))
        def msg2(_i, _c, *_args):
            _c = format_component(_c)
            if len(_args) == 3 and isinstance(_args[1], str) and isinstance(_args[-1], (tuple, list)):
                _exc_type, _exc_msg, _exc_args = _args
                try:
                    _exc_msg = _exc_msg.replace('%s', '{}').format(*_exc_args)
                except (SystemExit, KeyboardInterrupt):
                    raise
                except:
                    pass
                else:
                    _args = (_exc_type, _exc_msg)
            if _c:
                _args = (_c,) + _args
            msg = ''.join(('At iterate {}{}\t', ':  '.join(['{}'] * len(_args)), '\n'))
            _i = str(_i)
            return msg.format(' ' * max(0, 3 - len(_i)), _i, *_args)
        cumt = 0.
        t0 = time.time()
    #
    n = x0.size
    # working copy of the parameter vector
    x = x0
    x = np.array(x0)
    # bounds
    if bounds:
        lower_bound, upper_bound = np.full_like(x0, -np.inf), np.full_like(x0, np.inf)
        any_lower_bound = any_upper_bound = False
        for i, _bs in enumerate(bounds):
            _lb, _ub = _bs
            if _lb not in (None, -np.inf, np.nan):
                any_lower_bound = True
                lower_bound[i] = _lb
            if _ub not in (None, np.inf, np.nan):
                any_upper_bound = True
                upper_bound[i] = _ub
        if not any_lower_bound:
            lower_bound = None
        if not any_upper_bound:
            upper_bound = None
        if any_lower_bound or any_upper_bound:
            bounds = (lower_bound, upper_bound)
        else:
            bounds = None
        #if bounds and c < 1:
        #    warnings.warn('c < 1; bounds may be violated')
    # component
    class _defaultdict(object):
        __slots__ = ('_dict', 'arg', 'factory', 'args', 'kwargs')
        def __init__(self, factory, *args, **kwargs):
            self._dict = {}
            self.factory = factory
            self.arg = kwargs.pop('init_argument', None)
            self.args = args
            self.kwargs = kwargs
        def __make__(self, arg):
            return self.factory(arg, *self.args, **self.kwargs)
        def __nonzero__(self):
            return bool(self._dict)
        def __getitem__(self, i):
            try:
                item = self._dict[i]
            except KeyError:
                if self.arg is None:
                    arg = i
                elif callable(self.arg):
                    arg = self.arg(i)
                else:
                    arg = self.arg[i]
                item = self.__make__(arg)
                self._dict[i] = item
            return item
        def __setitem__(self, i, item):
            self._dict[i] = item
        def __len__(self):
            return len(self._dict)
        def __iter__(self):
            return iter(self._dict)

    # parameter singleton
    class Parameters(object):
        def __init__(self, x, covariate, gradient_subspace, descent_subspace,
                eps, fun, _sum, args, regul):
            self.x = x
            self.covariate = covariate
            self.gradient_subspace = gradient_subspace
            self.descent_subspace = descent_subspace
            self.eps = eps
            self.fun = fun
            self.sum = _sum
            self.args = args
            self.regul = regul
    def define_component(__global__, independent_components, memory, newton, gradient_covariate):
        # base classes
        class LocalSubspace(object):
            """
            Requires `x`, `covariate`, `gradient_subspace` and `descent_subspace`.
            """
            __slots__ = ('_i', '_subspace_map', '_gradient_subspace_size', '_descent_subspace_size')
            def __init__(self, i=None):
                self._i = i
                self._subspace_map = None
                self._gradient_subspace_size = None
                self._descent_subspace_size = None
            @property
            def _size_error(self):
                return ValueError('vector size does not match any (sub)space')
            @property
            def n(self):
                return __global__.x.size
            @property
            def i(self):
                if self._i is None:
                    raise ValueError('component attribute `i` is not set')
                return self._i
            @i.setter
            def i(self, i):
                if (self._i is None) != (i != self._i): # _i is None xor i != _i
                    self._i = i
                    self._subspace_map = None
                    self._gradient_subspace_size = None
                    self._descent_subspace_size = None
            @property
            def covariate(self):
                return __global__.covariate(self.i)
            @property
            def gradient_subspace(self):
                return __global__.gradient_subspace(self.i)
            @property
            def gradient_subspace_size(self):
                if self._gradient_subspace_size is None:
                    try:
                        g = self.g
                    except AttributeError:
                        g = None
                    if g is None:
                        j = self.gradient_subspace
                        if j is None: # full space
                            self._gradient_subspace_size = self.n
                        else:
                            self._gradient_subspace_size = len(j)
                    else:
                        self._gradient_subspace_size = len(g)
                return self._gradient_subspace_size
            @property
            def descent_subspace(self):
                j = __global__.descent_subspace(self.i)
                if j is None: # falls back onto gradient subspace
                    j = self.gradient_subspace
                return j
            @property
            def descent_subspace_size(self):
                if self._descent_subspace_size is None:
                    j = self.descent_subspace
                    if j is None: # full space
                        self._descent_subspace_size = self.n
                    else:
                        self._descent_subspace_size = len(j)
                return self._descent_subspace_size
            def in_full_space(self, vec, copy=False):#, working_copy=x
                if vec is None:
                    return None
                if vec.size == self.n:
                    if copy:
                        vec = np.array(vec)
                    return vec
                if vec.size == self.gradient_subspace_size:
                    j = self.gradient_subspace
                elif vec.size == self.descent_subspace_size:
                    j = self.descent_subspace
                else:
                    raise self._size_error
                working_copy = __global__.x
                if copy:
                    working_copy = np.array(working_copy)
                working_copy[j] = vec
                return working_copy
            def in_gradient_subspace(self, vec, copy=False):
                if vec is None:
                    return None
                if vec.size == self.n:
                    j = self.gradient_subspace
                    if j is not None:
                        vec = vec[j]
                    elif copy:
                        vec = np.array(vec)
                elif vec.size == self.gradient_subspace_size:
                    if copy:
                        vec = np.array(vec)
                elif vec.size == self.descent_subspace_size:
                    _vec = np.zeros(self.gradient_subspace_size, dtype=vec.dtype)
                    _vec[self.subspace_map] = vec
                    vec = _vec
                else:
                    raise self._size_error
                return vec
            def in_descent_subspace(self, vec, copy=False):
                if vec is None:
                    return None
                if vec.size == self.n:
                    j = self.descent_subspace
                    if j is not None:
                        vec = vec[j]
                    elif copy:
                        vec = np.array(vec)
                elif vec.size == self.descent_subspace_size:
                    if copy:
                        vec = np.array(vec)
                elif vec.size == self.gradient_subspace_size:
                    vec = vec[self.subspace_map]
                else:
                    raise self._size_error
                return vec
            @property
            def subspace_map(self):
                if self._subspace_map is None:
                    jg = self.gradient_subspace
                    assert jg is not None
                    jd = self.descent_subspace
                    assert jd is not None
                    self._subspace_map = [ (jg==j).nonzero()[0][0] for j in jd ]
                return self._subspace_map
        class LocalSubspaceProxy(object):
            __slots__ = ('__proxied__',)
            def __init__(self, i):
                if isinstance(i, LocalSubspaceProxy):
                    self.__proxied__ = i.__proxied__
                elif isinstance(i, LocalSubspace):
                    self.__proxied__ = i
                else:
                    self.__proxied__ = LocalSubspace(i)
            @property
            def n(self):
                return self.__proxied__.n
            @property
            def i(self):
                return self.__proxied__.i
            @i.setter
            def i(self, i):
                raise AttributeError('the `i` property is read-only')
                self.__proxied__.i = i
            @property
            def covariate(self):
                return self.__proxied__.covariate
            @property
            def gradient_subspace(self):
                return self.__proxied__.gradient_subspace
            @property
            def gradient_subspace_size(self):
                return self.__proxied__.gradient_subspace_size
            @property
            def descent_subspace(self):
                return self.__proxied__.descent_subspace
            @property
            def descent_subspace_size(self):
                return self.__proxied__.descent_subspace_size
            @property
            def subspace_map(self):
                return self.__proxied__.subspace_map
            def in_full_space(self, *args, **kwargs):
                return self.__proxied__.in_full_space(*args, **kwargs)
            def in_gradient_subspace(self, *args, **kwargs):
                return self.__proxied__.in_gradient_subspace(*args, **kwargs)
            def in_descent_subspace(self, *args, **kwargs):
                return self.__proxied__.in_descent_subspace(*args, **kwargs)
        # inverse Hessian matrix
        Pair = namedtuple('Pair', ('s', 'y', 'rho', 'gamma'))
        class AbstractInverseHessianBlock(LocalSubspaceProxy):
            """
            Requires `eps`.

            May also expose the internal representation of a block as attribute or property `block`.
            """
            __slots__ = ()
            @property
            def eps(self):
                return __global__.eps
            def dot(self, g):
                return self.block.dot(g)
            def update(self, s, y, proj):
                if proj is None:
                    proj = np.dot(s, self.in_descent_subspace(y))
                Hy = self.dot(y)
                yHy = np.dot(y, Hy)
                assert 0 <= yHy
                self.block = self.block + (\
                        (1 + yHy) / proj * np.outer(s, s) - np.outer(Hy, s) - np.outer(s, Hy)
                    ) / proj
            def drop(self):
                raise NotImplementedError('abstract method')
        class GradientDescent(AbstractInverseHessianBlock):
            """Do not use Cauchy points."""
            def dot(self, g):
                return self.eps * g
            def update(self, *args):
                pass
            def drop(self):
                pass
        class InverseHessianBlockView(AbstractInverseHessianBlock):
            __slots__ = ('fresh', 'slice')
            def __init__(self, component):
                AbstractInverseHessianBlock.__init__(self, component)
                self.fresh = True
                self.slice = np.ix_(self.gradient_subspace, self.gradient_subspace)
            def drop(self):
                self.fresh = True
            @property
            def block(self):
                block = __global__.H[self.slice]
                if self.fresh:
                    block = self.eps * (block + sparse.identity(self.gradient_subspace_size, format='lil'))
                return block
            @block.setter
            def block(self, block):
                #if self.fresh:
                #    _block = __global__.H[self.slice]
                #    i, j, k = sparse.find(_block)
                #    block[np.ix_(i,j)] += k
                #    block[np.ix_(i,j)] /= 2
                __global__.H[self.slice] = block
                self.fresh = False
        class IndependentInverseHessianBlock(AbstractInverseHessianBlock):
            __slots__ = ('block',)
            def __init__(self, component):
                AbstractInverseHessianBlock.__init__(self, component)
                self.drop()
            def drop(self):
                self.block = self.eps * np.identity(self.gradient_subspace_size)
        class LimitedMemoryInverseHessianBlock(AbstractInverseHessianBlock):
            """Requires `memory`."""
            __slots__ = ('block',)
            def __init__(self, component):
                AbstractInverseHessianBlock.__init__(self, component)
                self.drop()
            def drop(self):
                self.block = deque([], __global__.memory)
            def dot(self, g):
                if self.block:
                    # `block` is nonempty
                    p = np.array(g) # copy
                    U = []
                    for u in self.block: # for each past block from k-1 to k-m
                        if self.gradient_subspace is None or len(u.s) == self.gradient_subspace_size:
                            alpha = u.rho * np.dot(u.s, p)
                        else:
                            alpha = u.rho * np.dot(u.s, self.in_descent_subspace(p))
                        p -= alpha * u.y
                        U.append((u.s, u.y, u.rho, alpha))
                    p *= -self.last.gamma # gamma_{k-1}
                    for s, y, rho, alpha in U[::-1]: # from k-m to k-1
                        beta = rho * np.dot(y, p)
                        if len(s) == len(p):
                            p += (alpha - beta) * s
                        else:
                            assert self.subspace_map is not None
                            p[self.subspace_map] += (alpha - beta) * s
                else:
                    p = -self.eps * g
                return -p
            def update(self, s, y, proj):
                rho = 1. / proj
                gamma = proj / np.dot(y, y)
                self.block.appendleft(Pair(s, y, rho, gamma))
            @property
            def last(self):
                return self.block[0]
        # choose an implementation
        if not newton:
            InverseHessianBlock = GradientDescent
        elif independent_components:
            if memory:
                __global__.memory = memory
                InverseHessianBlock = LimitedMemoryInverseHessianBlock
            else:
                InverseHessianBlock = IndependentInverseHessianBlock
        else:
            # just ignore
            #if memory:
            #    raise NotImplementedError('`memory` requires `indepdendent_components`')
            __global__.H = sparse.lil_matrix((x.size, x.size))
            InverseHessianBlock = InverseHessianBlockView
        # component
        class Component(LocalSubspaceProxy):
            """
            Requires `fun`, `_sum` and `args`.
            """
            __slots__ = ('_x', '_f', '_g', '_H')
            def __init__(self, i):
                LocalSubspaceProxy.__init__(self, i)
                self._x = None # gradient-active components only
                self._f = None
                self._g = None # gradient-active components only
                self._H = None
            @property
            def x(self):
                return self._x
            @x.setter
            def x(self, _x):
                assert _x is None
                self._x = self.in_gradient_subspace(_x)
                self.f = None
                self.g = None
                self.H = None
            def pull(self, _x):
                self._x = self.in_gradient_subspace(_x)
                #assert self._x is not _x # fails if not _stochastic
                self._f = self.__f__(_x)
                assert self._f is not None
                self._g = self.__g__(_x, update=True)
                assert self._g is not None
            def __f__(self, _x):
                #assert _x is x # check there is a single working copy
                return __global__.fun(self.i, _x, *args)
            @property
            def f(self):
                if self._f is None:
                    self._f = self.__f__(self.in_full_space(self.x))
                return self._f
            @f.setter
            def f(self, f):
                self._f = f
            def __g__(self, _x, subspace=None, covariate=None, update=False):
                #assert _x is x # check there is a single working copy
                if subspace is None:
                    subspace = self.gradient_subspace
                if covariate is None:
                    covariate = self.covariate
                _total_g, _partial_g = sparse_grad(__global__.fun, _x, covariate,
                        subspace, __global__.args, __global__.sum, __global__.regul)
                return _total_g
            @property
            def g(self):
                if self._g is None:
                    self._g = self.__g__(self.in_full_space(self.x), update=True)
                return self._g
            @g.setter
            def g(self, g):
                self._g = self.in_gradient_subspace(g)
            @property
            def H(self):
                if self._H is None:
                    self._H = InverseHessianBlock(self)
                return self._H
            @H.setter
            def H(self, H):
                self._H = H
            def commit(self, s):
                c = Component(self) # fails if `i` is not set
                if self.x is None:
                    raise ValueError('parameter vector `x` not defined')
                #if self.s is None:
                #    raise ValueError('parameter update `s` not defined')
                c._x = np.array(self.x)
                if c.descent_subspace is None:
                    if len(s) != self.n:
                        raise ValueError('wrong size for parameter update `s`')
                    c._x += s
                else:
                    if len(s) != c.descent_subspace_size:
                        raise ValueError('parameter update `s` is not in descent subspace')
                    c._x[c.subspace_map] += s
                return c
            def push(self, _x=None):
                if _x is None:
                    _x = __global__.x
                __x = self.x
                if __x is None:
                    raise RuntimeError('no parameters are defined')
                if __x.size == _x.size:
                    if __x is not _x:
                        _x[...] = __x
                elif __x.size == self.gradient_subspace_size:
                    _x[self.gradient_subspace] = __x
                else:
                    raise self._size_error
        if gradient_covariate is None:
            return Component
        else:
            class Component1(Component):
                __slots__ = ()
                def __g__(self, x, subspace=None, update=False):
                    return Component.__g__(self, x, subspace, gradient_covariate, update)
            return Component1
    __global__ = Parameters(x, covariate, gradient_subspace, descent_subspace, eps, fun, _sum, args, regul)
    Component = define_component(__global__, independent_components, memory, newton, gradient_covariate)
    C = _defaultdict(Component)
    def push(c1, H1=None):
        i = c1.i
        C[i] = c1
        c1.push(x) # update the full parameter vector
        if H1 is not None:
            c1.H.update(*H1)
    ls_kwargs = {}
    if ls_step_max:
        ls_kwargs['step_max'] = ls_step_max
    if ls_iter_max:
        ls_kwargs['iter_max'] = ls_iter_max
    if ls_armijo_max:
        ls_kwargs['armijo_max'] = ls_armijo_max
    if ls_wolfe:
        try:
            c2, c3 = ls_wolfe
        except TypeError:
            c2 = c3 = ls_wolfe
        ls_kwargs['c3'] = c3
    elif newton:
        c2 = .9
    else:
        c2 = .1
    ls_kwargs['c2'] = c2
    if regul:
        regul0 = regul
    if ls_regul:
        ls_regul0 = ls_regul
    if s_scale <= 0 or 1 < s_scale:
        raise ValueError('expected: 0 < s_scale <= 1')
    fcount = gcount = .9
    k0 = k1 = 0

    i_prev = None # previous component index
    k = 0 # iteration index
    while True:
        try:
            if max_iter and max_iter <= k:
                resolution = 'MAXIMUM ITERATION REACHED'
                break
            if regul_decay:
                _decay = max(1. - float(k) * regul_decay, 1e-10)
                if regul:
                    __global__.regul = regul = regul0 * _decay
                if ls_regul:
                    ls_regul = ls_regul0 * _decay

            # choose the target component (as a component index)
            i = component(k)
            # retrieve the corresponding component object
            c = C[i]

            # check for changes in the corresponding parameters since last iteration on component `i`
            if i != i_prev:
                # copy part of it for initial H update
                new_component = c.x is None
                if new_component:
                    c.pull(x0)
                x_prev, g_prev = c.x, c.g
                # update with current parameters
                c.pull(x)
                # update H with inter-iteration changes
                if np.allclose(x_prev, c.x):
                    pass
                elif newton:
                    s_ii = c.x - x_prev
                    y_ii = c.g - g_prev
                    proj_ii = np.dot(s_ii, y_ii)
                    if proj_ii <= 0:
                        if verbose:
                            print(msg2(k, i, 'PROJ G <= 0 (k)'))
                        c.H.drop()
                    else:
                        c.H.update(s_ii, y_ii, proj_ii)
                i_prev = i # do so before first `continue` may happen

            # estimate the local gradient
            g = c.g # g_{k}

            # retrieve the local inverse Hessian
            H = c.H # H_{k}

            # define the descent direction
            p = -H.dot(g) # p_{k} or q_{k}
            p = c.in_descent_subspace(p)

            # get the parameter update
            s = wolfe_line_search(
                    c.__f__, x, p, c.__g__, c.descent_subspace,
                    f0=c.f, g0=c.in_descent_subspace(g), bounds=bounds,
                    weight_regul=regul, step_regul=ls_regul,
                    **ls_kwargs) # s_{k}
            # sanity checks
            if s is None:
                if verbose:
                    print(msg2(k, i, 'LINE SEARCH FAILED'))
                push(c) # undo any change in the working copy of the parameter vector
                continue
            if np.all(s == 0):
                if verbose:
                    print(msg2(k, i, 'NULL UPDATE'))
                push(c) # undo any change in the working copy of the parameter vector
                continue

            s *= s_scale

            # update the parameter vector
            c1 = c.commit(s) # x_{k+1} = x_{k} + s_{k}

            # check for convergence based on f
            ncomponents = len(C)
            assert 0 < ncomponents
            if ftol is not None:
                df = c.f - c1.f
                if df < ftol:
                    k0 += 1
                if k and k % ncomponents == 0:
                    if fcount * ncomponents <= k0:
                        resolution = 'CONVERGENCE: DELTA F < FTOL'
                        push(c1)
                        break
                    else:
                        k0 = 0

            if gtol is None and not newton:
                if verbose:
                    print(msg1(k, i, c.f, c1.f))
                continue

            # estimate the gradient at x_{k+1}
            h = c1.g # g_{k+1}
            # sanity checks
            if h is None:
                if verbose:
                    print(msg2(k, i, 'GRADIENT CALCULATION FAILED (k+1)'))
                # drop c1
                push(c) # undo any change in the working copy of the parameter vector
                continue
            if np.allclose(g, h):
                if verbose:
                    print(msg2(k, i, 'NO CHANGE IN THE GRADIENT'))
                push(c1) # 'push' the parameter update...
                continue # ...but do not update H

            #
            y = h - g # y_{k} = g_{k+1} - g_{k}
            #
            proj = np.dot(s, c.in_descent_subspace(y)) # s_{k}^{T} . y_{k}
            if proj <= 0:
                if verbose:
                    print(msg2(k, i, 'PROJ G <= 0 (k+1)'))
                # either drop c1...
                #push(c) # undo any change in the working copy of the parameter vector
                # ... or drop the inverse Hessian
                c1.H.drop()
                push(c1)
                continue
            elif verbose:
                print(msg1(k, i, c.f, c1.f, proj))

            # check for convergence based on g
            if gtol is not None:
                if proj < gtol:
                    k1 += 1
                if k and k % ncomponents == 0:
                    if gcount * ncomponents <= k1:
                        resolution = 'CONVERGENCE: PROJ G < GTOL'
                        push(c1)
                        break
                    else:
                        k1 = 0

            # 'push' the parameter update together with H update
            H1 = (s, y, proj)
            c1.H.update(*H1)
            push(c1)

        except KeyboardInterrupt:
            resolution = 'INTERRUPTED'
            push(c)
            break

        finally:
            # loop
            k += 1

    if verbose:
        cumt += time.time() - t0
        print('           * * *\n\n{}\n'.format(resolution))
        minute = floor(cumt / 60.)
        second = cumt - minute * 60.
        if minute:
            print('Elapsed time = {:d}m{:.3f}s\n'.format(minute, second))
        else:
            print('Elapsed time = {:.3f}s\n'.format(second))
    H = {i: C[i].H for i in C}
    return BFGSResult(x, H, None, None, None, None, None)


def sparse_grad(fun, x, active_i, active_j, args=(), _sum=np.sum, regul=None):
    """
    Compute the derivative of a function.
    """
    SAFE, CON = 2., 1.4
    CON2 = CON * CON
    h0 = 1e-8
    H = h0 * ((1./CON) ** np.arange(10))
    if active_j is None:
        active_j = range(x.size)
    if not regul:
        penalty = 0.
    total_grad, partial_grad = [], {}
    any_ok = False
    a = np.zeros((H.size, H.size), dtype=float)
    for j in active_j:
        if callable(active_i):
            I = active_i(j)
        else:
            I = active_i
        total_grad_j, err = None, np.inf
        xj = x[j] # keep copy
        try:
            for u, h in enumerate(H):
                try:
                    x[j] = xj + h
                    f_a = np.array([ fun(i, x, *args) for i in I ])
                    x[j] = xj - h
                    f_b = np.array([ fun(i, x, *args) for i in I ])
                    partial_grad_j = (f_a - f_b) / (2. * h)
                    if regul:
                        penalty = regul * 2. * xj
                    a[u,0] = a_up = _sum(partial_grad_j) + penalty
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    #traceback.print_exc()
                    #raise
                    continue
                if u == 0:
                    continue
                fac = CON2
                for v in range(1, a.shape[1]):
                    a_pp = a[u-1,v-1] # pp stands for [previous, previous]
                    a[u,v] = a_uv = (a_up * fac - a_pp) / (fac - 1.)
                    fac *= CON2
                    err_uv = max(abs(a_uv - a_up), abs(a_uv - a_pp))
                    if err_uv <= err:
                        err, total_grad_j = err_uv, a_uv
                    a_up = a_uv
                if SAFE * err <= abs(a[u,u] - a[u-1,u-1]):
                    break
            if total_grad_j is None:
                print('sparse_grad failed at column {}'.format(j))
                total_grad_j = 0.
            else:
                any_ok = True
                partial_grad[j] = (I, np.array(partial_grad_j))
            total_grad.append(total_grad_j)
        finally:
            x[j] = xj # restore
    if any_ok:
        total_grad = np.array(total_grad)
        return total_grad, partial_grad
    else:
        return None, None


__all__ = [ 'BFGSResult', 'minimize_sbfgs', 'minimize_range_sbfgs',
        'sdfunc', 'slnsrch', 'subspace_search', 'minimize_sparse_bfgs' ]

