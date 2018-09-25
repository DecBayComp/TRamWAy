
from math import *
import numpy as np
import scipy.optimize.linesearch as ls
import time
import scipy.sparse as sparse
from collections import namedtuple
import traceback


BFGSResult = namedtuple('BFGSResult', ('x', 'B', 'err', 'f', 'projg', 'cumtime'))


def sdfunc(func, yy, ids, components, join, h, args=(), kwargs={}, per_component_diff=None):
    """
    Compute the derivative of a function.

    Borrowed in large parts from *Numerical Recipes*

    Gradient is calculated as ``(join([ func(x+dx_i) for i in ids ]) - join([ func(x-dx_i) for i in ids ])) / (2 * h)``
    where ``h = norm(dx_i)``,
    unless `per_component_diff` is defined.
    In this later case, the gradient is ``join([ per_component_diff(func(x-dx_i), func(x+dx_i)) for i in ids ]) / (2 * h)``.
    This may moderate numerical precision errors.
    """
    SAFE, CON = 2., 1.4
    CON2 = CON * CON
    ans = []
    a = np.zeros((h.size, h.size), dtype=float)
    #a = np.zeros((h.shape[1], h.shape[1]), dtype=float)
    for d in ids:
        ans_d, err = None, np.inf
        #for i, hh in enumerate(h[d]):
        for i, hh in enumerate(h):
            yy0, yy1 = np.array(yy), np.array(yy) # copies
            yy0[d] -= hh
            yy1[d] += hh
            try:
                if per_component_diff:
                    a[0,i] = a_ip \
                        = join([ per_component_diff( \
                                func(j, yy0, *args, **kwargs), \
                                func(j, yy1, *args, **kwargs)) \
                             for j in components ]) \
                        / (2. * hh)
                else:
                    a[0,i] = a_ip \
                        = (join([ func(j, yy1, *args, **kwargs) \
                              for j in components ]) \
                        -  join([ func(j, yy0, *args, **kwargs) \
                              for j in components ])) \
                        / (2. * hh)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                #traceback.print_exc()
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
            #print(a)
            return None
        ans.append(ans_d)
    return np.array(ans)



def minimize_sgbfgs(minibatch, fun, x0, args=(), join=np.sum, eta0=1., c=1., l=0., eps=1.,
    gtol=1e-10, gcount=10, maxiter=100000, maxcor=50, covariates=None, ncomps=None,
    verbose=False, iter_kwarg=None, per_component_diff=None, reference_parameters=None):
    """
    Sparse-gradient variant of the BFGS minimization algorithm.
    """
    if verbose:
        def msg1(_i, _f, _dg):
            _i = str(_i)
            return 'At iterate {}{}\tf= {}{:E} \tproj g = {:E}\n'.format(
                ' ' * max(0, 3 - len(_i)), _i, ' ' if 0 <= _f else '', _f, _dg)
        def msg2(_i, *_args):
            msg = ''.join(('At iterate {}{}\t', ': '.join(['{}'] * len(_args)), '\n'))
            _i = str(_i)
            return msg.format(' ' * max(0, 3 - len(_i)), _i, *_args)
        cumt = 0.
        fs = []
        t0 = time.time()
    # precompute arguments for dfunc
    h = 1e-8#np.maximum(1e-8, 1e-4 * np.abs(x0))
    CON = 1.4
    dfunc_h = h * ((1./CON) ** np.arange(maxcor))
    #dfunc_h = np.outer(h, (1./CON) ** np.arange(maxcor))
    if per_component_diff is None and join in (np.sum, np.mean):
        per_component_diff = lambda a, b: b - a
    #
    fargs = args
    fkwargs = {}
    def f(_x, _rows):
        return join( fun(_j, _x, *fargs, **fkwargs) for _j in _rows )
    #
    if iter_kwarg:
        fkwargs[iter_kwarg] = 0
    rows = minibatch(0, x0)
    twoways = False
    explicit_components = False
    if isinstance(rows, tuple):
        if rows[1:]:
            if len(rows) == 3:
                i, rows, cols = rows
                explicit_components = True
            elif len(rows) == 2:
                rows, cols = rows
            else:
                raise ValueError('too many values to unpack from minibatch')
            twoways = True
            def df(_x, _rows, _cols):
                return sdfunc(fun, _x, _cols, _rows, join, dfunc_h, fargs, fkwargs, per_component_diff)
            g = df(x0, rows, cols)
        else:
            rows, = rows
    if not twoways:
        cols = None
        def df(_x, _rows):
            return sdfunc(fun, _x, range(_x.size), _rows, join, dfunc_h, fargs, fkwargs, per_component_diff)
        g = df(x0, rows)
    if g is None:
        raise RuntimeError('gradient calculation failed')
    p = -eps * g
    # precompute argument(s) for slnsrch
    STPMX = 500.
    step_max = STPMX * max(np.sqrt(np.sum(p * p)), p.size) # assume p.size is constant
    ls_maxiter = None#10
    #
    eta = slnsrch(f, rows, cols, x0, p, g, eta0, ls_maxiter, step_max)
    s = eta / c * p
    if twoways:
        x = np.array(x0) # copy
        x[cols] += s
        h = df(x, rows, cols)
    else:
        x = x0 + s
        h = df(x, rows)
    if h is None:
        raise RuntimeError('gradient calculation failed')
    y = h - g + l * s
    proj = np.dot(s, y)
    projs = [proj]
    if verbose:
        cumt += time.time() - t0
        try:
            f1 = f(x, rows)
        except ValueError as e:
            print(msg2(0, 'ValueError', e.args[0]))
        else:
            print(msg1(0, f1, proj))
            fs.append(f1)
        t0 = time.time()
    if reference_parameters is None:
        errs = None
    else:
        err = x - reference_parameters
        errs = [np.sum(err * err)]
    # make B sparse and preallocate the nonzeros
    if explicit_components:
        B = {}
        B[i] = proj / np.dot(y, y)
    elif twoways and (sparse.issparse(covariates) or covariates):
        if isinstance(covariates, tuple):
            indices, indptr = covariates
        else:
            indices, indptr = covariates.indices, covariates.indptr
        B = sparse.csr_matrix((np.zeros(indices.size, dtype=x0.dtype), indices, indptr),
            shape=(x0.size, x0.size))
        B[cols, cols] = proj / np.dot(y, y)
        covariates = True
    else:
        raise NotImplementedError('fixme')
        B = np.diag(np.full(x.size, proj / np.dot(y, y)))
    x2 = x0
    k1 = k2 = 0
    resolution = None
    for t in range(1, int(maxiter)):
        try:
            if iter_kwarg:
                fkwargs[iter_kwarg] = t

            ## step (pre-a) ##
            if twoways:
                if explicit_components:
                    i, rows, cols = minibatch(t, x)
                else:
                    rows, cols = minibatch(t, x)
                g = df(x, rows, cols) # computation of `g` is part of step (a)
            else:
                rows = minibatch(t, x)
                g = df(x, rows)

            #
            if g is None:
                x = x2
                if verbose:
                    print(msg2(t, 'GRADIENT CALCULATION FAILED (t)'))
                continue
                resolution = 'GRADIENT CALCULATION FAILED (t)'
                break

            ## step (a) ##
            if explicit_components:
                try:
                    Bi = B[i]
                except KeyError:
                    p = -eps * g
                else:
                    p = -np.dot(Bi, g)
            elif twoways:
                if covariates:
                    firsttime = np.all(B[cols,cols] == 0)
                    if firsttime:
                        p = -eps * g
                    else:
                        p = np.ravel(-np.dot(B[np.ix_(cols, cols)].todense(), g))
                else:
                    p = -np.dot(B[np.ix_(cols, cols)], g)
                if not firsttime and 0 <= np.sum(g * p):
                    # line search will fail
                    #print(B[np.ix_(cols, cols)].todense())
                    # make a small update instead
                    p = -eps * g
            else:
                p = -np.dot(B, g)

            ## step (b) ##
            #epoch = ncomps
            #if epoch:
            #    if t % epoch == 1:
            #        eta = tau / (tau + float(t / epoch)) * eta0
            #else:
            #    eta = tau / (tau + float(t)) * eta0
            eta = slnsrch(f, rows, cols, x, p, g, eta0, ls_maxiter, step_max)
            if not eta:
                x = x2
                #if verbose:
                #    print(msg2(t, 'LINE SEARCH FAILED'))
                #continue
                resolution = 'LINE SEARCH FAILED'
                break

            ## step (c) ##
            s = eta / c * p

            ## step (d) ##
            if twoways:
                x1 = np.array(x)
                x1[cols] += s
                h = df(x1, rows, cols) # computation of `h` is part of step (e)
            else:
                x1 = x + s
                h = df(x1, rows)

            #
            if h is None:
                x = x2
                if verbose:
                    print(msg2(t, 'GRADIENT CALCULATION FAILED (t+1)'))
                continue
                resolution = 'GRADIENT CALCULATION FAILED (t+1)'
                break
            x2, x = x, x1

            if reference_parameters is not None:
                err = x - reference_parameters
                errs.append(np.sum(err * err))

            ## step (e) ##
            y = h - g
            y += l * s

            #
            proj = np.dot(s, y)
            projs.append(proj)
            if gtol is not None and abs(proj) < gtol:
                k1 += 1
                if gcount <= k1:
                    resolution = 'CONVERGENCE: |PROJ G| < GTOL'
                    break
            else:
                k1 = 0

            # step (f)
            if explicit_components:
                if i not in B:
                    B[i] = proj / np.dot(y, y)
            elif twoways and covariates and firsttime:
                B[cols, cols] = proj / np.dot(y, y)
            #elif not twoways or not covariates: raise NotImplementedError

            #
            if verbose:
                cumt += time.time() - t0
                try:
                    f1 = f(x, np.arange(ncomps))#rows)
                except ValueError as e:
                    print(msg2(t, 'ValueError', e.args[0]))
                else:
                    print(msg1(t, f1, proj))
                    fs.append(f1)
                t0 = time.time()

            ## step (g) ##
            rho = 1. / proj

            #
            if explicit_components:
                q = -np.dot(B[i], y)
            elif twoways:
                # no first time or small step computation for `q`
                if covariates:
                    q = np.ravel(-np.dot(B[np.ix_(cols, cols)].todense(), y))
                else:
                    q = -np.dot(B[np.ix_(cols, cols)], y)
            else:
                q = -np.dot(B, y)

            ## step (h) ##
            # B = (I - rho s y.T) B ( I - rho y s.T) + c rho s s.T
            phi = c - rho * np.dot(y, q) # rho s y.T B y s.T rho + c rho s s.T = rho phi s s.T
            deltaB = rho * (np.outer(q, s) + np.outer(s, q) + phi * np.outer(s, s))
            if explicit_components:
                B[i] = B[i] + deltaB
            elif twoways:
                B[np.ix_(cols, cols)] += deltaB
            else:
                B += deltaB

        except KeyboardInterrupt:
            x = x2
            # some variables may not be valid, e.g. t0
            break

    if verbose:
        cumt += time.time() - t0
        if not resolution:
            resolution = 'MAXIMUM ITERATION REACHED'
        print('       * * *\n\n{}\n'.format(resolution))
        minute = floor(cumt / 60.)
        second = cumt - minute * 60.
        if minute:
            print('Elapsed time = {:d}m{:.3f}s\n'.format(minute, second))
        else:
            print('Elapsed time = {:.3f}s\n'.format(second))
        return BFGSResult(x, B, errs, fs, projs, cumt)
    else:
        return BFGSResult(x, B, errs, None, projs, None)



def slnsrch(f, rows, cols, x, p, g, eta_max=1., iter_max=10, step_max=None):
    """
    Line search.

    Borrowed in large parts from *Numerical Recipes*.
    """
    if cols is None:
        eta, f_count, fnew =  ls.line_search_armijo(f, x, p, g, f(x, rows), (rows,))
        return eta
    #
    TOLX = 1e-8
    ALF = 1e-4
    #
    if step_max:
        norm = np.sqrt(np.sum(p * p))
        if step_max < norm * eta_max:
            p = p * (step_max / norm) # not in-place
    slope = np.sum(g * p)
    if 0 <= slope:
        #raise ValueError('positive slope: {}'.format(slope))
        return 0.
    fold = f2 = f(x, rows)
    xold = x[cols]
    xnew = np.array(x)
    eta_min = min(eta_max * 1e-3, TOLX / np.max(np.abs(p) / np.maximum(np.abs(xold), 1.)))
    eta = eta2 = eta_max
    last_valid_eta = None
    i = 0
    while True:
        xnew[cols] = xold + eta * p
        try:
            fnew = f(xnew, rows)
        except ValueError:
            eta /= 2.
        else:
            last_valid_eta = eta
            if eta < eta_min:
                #print('eta < eta_min={}'.format(eta_min))
                break
            df = fnew - fold
            if df <= ALF * eta * slope:
                #print('df <= ALF * eta * slope = {} * {} * {}'.format(ALF, eta, slope))
                break
            if eta == eta_max:
                eta1 = -slope / (2. * (df - slope))
            else:
                rhs1 = df - eta * slope
                rhs2 = f2 - fold - eta2 * slope
                rhs1 /= eta * eta
                rhs2 /= eta2 * eta2
                a = (rhs2 - rhs1) / (eta - eta2)
                b = (eta * rhs2 - eta2 * rhs1) / (eta - eta2)
                if a == 0:
                    eta1 = -slope / (2. * b)
                else:
                    discr = b * b - 3. * a * slope
                    if discr < 0:
                        eta1 = eta / 2.
                    elif b <= 0:
                        eta1 = (np.sqrt(discr) - b) / (3. * a)
                    else:
                        eta1 = -slope / (np.sqrt(discr) + b)
                eta1 = min(eta1, eta / 2.)
            eta2, f2 = eta, fnew
            eta = max(eta1, .1 * eta)
        i += 1
        if iter_max and i == iter_max:
            #print('iter_max reached')
            break
    return last_valid_eta


__all__ = [ 'BFGSResult', 'minimize_sgbfgs' ]

