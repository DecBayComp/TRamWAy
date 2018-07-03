
from math import *
import numpy as np
import scipy.optimize.linesearch as ls
import traceback
import time
import itertools
import scipy.sparse as sparse
import warnings
from collections import namedtuple


def dfpmin(func, p, args=(), maxcor=10, ftol=1.2e-10, gtol=1e-16, eps=3e-11, maxiter=10000, verbose=False):
        # precompute arguments for dfunc
        CON = 1.4
        dfunc_h = 1e-8 * ((1./CON) ** np.arange(maxcor))
        dfunc_fac = (CON*CON)# ** np.arange(1, dfunc_h.size)
        def _dfunc(p):
                return dfunc(func, p, dfunc_h, dfunc_fac, args)
        #
        ndim = p.size
        fp = func(p, *args)
        g = _dfunc(p)
        hessian = np.eye(ndim, dtype=float)
        xi = -g
        for its in range(maxiter):
                alpha, f_count, fp = lnsrch(func, p, xi, g, fp, args)
                xi *= alpha
                p += xi
                ferr = np.max(np.abs(xi) / np.maximum(np.abs(p), 1.))
                if ferr <= ftol:
                        if verbose:
                                print('           * * *\n\nCONVERGENCE: FERR <= FTOL\n')
                        return p, fp
                dg = g
                g = _dfunc(p)
                gerr = np.max(np.abs(g) * np.maximum(np.abs(p), 1.)) / max(fp, 1.)
                if verbose:
                        print('At iterate  {}\tf= {:E}\tferr= {:E}\t gerr= {:E}\n'.format(its, fp, ferr, gerr))
                if gerr <= gtol:
                        if verbose:
                                print('           * * *\n\nCONVERGENCE: GERR <= GTOL\n')
                        return p, fp
                dg = g - dg
                hdg = np.dot(hessian, dg)
                fac = np.dot(dg, xi)
                fae = np.dot(dg, hdg)
                sumdg = np.dot(dg, dg)
                sumxi = np.dot(xi, xi)
                if sqrt(eps * sumdg * sumxi) < fac:
                        fac = 1. / fac
                        fad = 1. / fae
                        dg = fac * xi - fad * hdg
                        hessian += fac * np.outer(xi, xi) - fad * np.outer(hdg, hdg) + fae * np.outer(dg, dg)
                xi = -np.dot(hessian, g)
        # dfpmin failed
        if verbose:
                print('           * * *\n\nMAXIMUM NUMBER OF ITERATIONS EXCEEDED\n')
        return p, fp


def dfunc(func, yy, h, CON2, args=()):
        ans = []
        for d in range(yy.size):
                ans_d = np.inf
                hh = np.zeros((h.size, yy.size), dtype=yy.dtype)
                hh[:,d] = h
                a_prev = np.array([ func(yy+hh[i], *args) - func(yy-hh[i], *args)
                                for i in range(h.size) ]) / (2. * h)
                fac = CON2
                for j in range(1, h.size):
                        a_cur = (a_prev[1:] * fac - a_prev[:-1]) / (fac - 1.)
                        fac *= CON2
                        errt = np.maximum(np.abs(a_cur - a_prev[1:]), np.abs(a_cur - a_prev[:-1]))
                        ans_d = min(ans_d, a_cur[errt.argmin()])
                        a_prev[0], a_prev[1:] = 0., a_cur
                ans.append(ans_d)
        return np.array(ans)


def lnsrch(func, p, xi, g, fp, args, c1=1e-4, alpha=1.):
        #return ls.line_search_armijo(func, p, xi, g, fp, args, c1)
        f_count = 0
        slope = np.dot(g, xi)
        assert slope < 0
        alphamin = c1 * c1 / np.max(np.abs(xi) / np.maximum(np.abs(p), 1.))
        f2 = np.nan
        while np.isnan(f2):
                if alpha < alphamin: # lnsrch failed
                        return 0., f_count, fp
                pnew = p + alpha * xi
                f2 = func(pnew, *args)
                f_count += 1
                alpha2, alpha = alpha, .1 * alpha
        while True:
                pnew = p + alpha * xi
                fpnew = func(pnew, *args)
                f_count += 1
                if alpha < alphamin: # lnsrch failed
                        return 0., f_count, fp
                elif fpnew <= fp + c1 * alpha * slope:
                        return alpha, f_count, fpnew
                else:
                        rhs1 = (fpnew - fp - alpha * slope) / (alpha * alpha)
                        rhs2 = (f2 - fp - alpha2 * slope) / (alpha2 * alpha2)
                        a = (rhs2 - rhs1) / (alpha2 - alpha)
                        b = (alpha2 * rhs1 - alpha * rhs2) / (alpha2 - alpha)
                        if a == 0.:
                                tmplam = -slope / (2. * b)
                        else:
                                discr = b * b - 3. * a * slope
                                if discr < 0.:
                                        tmplam = .5 * alpha
                                elif b <= 0.:
                                        tmplam = (-b + sqrt(discr)) / (3.*a)
                                else:
                                        tmplam = -slope / (b + sqrt(discr))
                        tmplam = min(tmplam, .5 * alpha)
                # prepare next iteration
                alpha2, alpha = alpha, max(tmplam, .1 * alpha)
                f2 = fpnew


def sdfpmin(func, p, args=(), minibatch=None, component_count=None,
        maxcor=10, ftol=1.2e-10, gtol=1e-16, eps=3e-11, maxiter=10000, epoch=None,
        verbose=False):
        """
        Stochastic minimization of sum_i func(i, p, *args).

        Arguments:

                func (callable):
                        scalar function which sum is to be minimized;
                        takes a (single) component index and a parameter vector.

                p (numpy.ndarray):
                        initial parameter vector.

                args (tuple):
                        extra positional input arguments for `func`.

                minibatch (callable):
                        function of the iteration number, the parameter vector and the multicomponent
                        objective function;
                        returns two lists of component indices and the list of the corresponding
                        indices in the parameter vector.

                component_count (int):
                        number of components.

        """
        if epoch is None:
                epoch = 2 * int(component_count)
        if verbose:
                msg = 'At iterate  {}\tf= {:E}\tferr= {:E}\t gerr= {:E}\n'
        def scalar_func(components):
                def _f(p):
                        return sum( func(j, p, *args) for j in components )
                return _f
        # precompute arguments for dfunc
        CON = 1.4
        dfunc_h = 1e-8 * ((1./CON) ** np.arange(maxcor))
        def _dfunc(p, ids, components):
                return sdfunc(func, p, ids, components, dfunc_h, args)
        #
        ndim = p.size
        fjp = np.array([ func(j, p, *args) for j in range(component_count) ])
        fp = np.sum(fjp)
        g = dfunc(scalar_func(range(component_count)), p, dfunc_h)
        hessian = np.eye(ndim, dtype=float)
        dp = -g
        lnsrch_failure_count, max_failures = 0, component_count
        i_prev = set()
        for its in range(maxiter):
                while True:
                        i, components, ids = minibatch(its, p, fjp)
                        if i in i_prev:
                                lnsrch_failure_count += 1
                                if max_failures < lnsrch_failure_count:
                                        if verbose:
                                                print('           * * *\n\nLINE SEARCH FAILED\n')
                                        return p, fjp, fp
                                continue
                        break
                #g = _dfunc(p, ids, components)
                #dp = -g
                alpha, f_count, fjp, fp = slnsrch(func, p, ids, components, dp, g, fjp, fp, args=args)
                if alpha == 0:
                        #if verbose:
                        #        print('           * * *\n\nLINE SEARCH FAILED\n')
                        #return p, fjp, fp
                        lnsrch_failure_count += 1
                        if max_failures < lnsrch_failure_count:
                                if verbose:
                                        print('           * * *\n\nLINE SEARCH FAILED\n')
                                return p, fjp, fp
                        #elif verbose and lnsrch_failure_count == 1:
                        #        print('At iterate  {}\tLINE SEARCH FAILED\n'.format(its))
                        i_prev.add(i)
                        continue
                else:
                        lnsrch_failure_count = 0
                        i_prev = set()
                dp[ids] = dp1 = dp[ids] * alpha
                #dp = dp1 = dp * alpha
                p[ids] = p1 = p[ids] + dp1
                check = True#(its+1) % epoch == 0
                if check:
                        _p = np.maximum(np.abs(p), 1.)
                        #_p = np.maximum(np.abs(p1), 1.)
                        ferr = np.max(np.abs(dp) / _p)
                        if ferr <= ftol:
                                if verbose:
                                        g1 = _dfunc(p, ids, components)
                                        gerr = np.max(np.abs(g1) * _p[ids]) / max(fp, 1.)
                                        #gerr = np.max(np.abs(g1) * _p) / max(fp, 1.)
                                        print(msg.format(its, fp, ferr, gerr))
                                        print('           * * *\n\nCONVERGENCE: FERR <= FTOL\n')
                                return p, fjp, fp
                if verbose:
                        ferr = np.max(np.abs(dp1) / _p[ids])
                        #ferr = np.max(np.abs(dp1) / _p)
                g0 = g[ids] # copy
                g[ids] = g1 = _dfunc(p, ids, components)
                #g0, g = g, _dfunc(p, ids, components)
                if check:
                        gerr = np.max(np.abs(g) * _p) / max(fp, 1.)
                        if gerr <= gtol:
                                if verbose:
                                        print(msg.format(its, fp, ferr, gerr))
                                        print('           * * *\n\nCONVERGENCE: GERR <= GTOL\n')
                                return p, fjp, fp
                if verbose:
                        gerr = np.max(np.abs(g1) * _p[ids]) / max(fp, 1.)
                        print(msg.format(its, fp, ferr, gerr))
                dg = g1 - g0
                #dg = g - g0
                hdg = np.dot(hessian[np.ix_(ids, ids)], dg)
                #hdg = np.dot(hessian, dg)
                #dp1 = dp
                fac = np.dot(dg, dp1)
                fae = np.dot(dg, hdg)
                sumdg = np.dot(dg, dg)
                sumdp = np.dot(dp1, dp1)
                if sqrt(eps * sumdg * sumdp) < fac:
                        fac = 1. / fac
                        fad = 1. / fae
                        dg = fac * dp1 - fad * hdg
                        #hessian += fac * np.outer(dp1, dp1) \
                        hessian[np.ix_(ids, ids)] += fac * np.outer(dp1, dp1) \
                                - fad * np.outer(hdg, hdg) \
                                + fae * np.outer(dg, dg)
                dp = -np.dot(hessian, g)
                #dp[ids] = -np.dot(hessian[np.ix_(ids, ids)], g)
        # dfpmin failed
        if verbose:
                print('           * * *\n\nMAXIMUM NUMBER OF ITERATIONS EXCEEDED\n')
        return p, fjp, fp


def _sdfunc(func, yy, ids, components, h, CON2, args=()):
        ans = []
        for d in ids:
                ans_d = np.inf
                # compute a_prev (a[:][0] in InferenceMAP)
                ok = np.ones(h.size, dtype=bool)
                a_prev = []
                for i, hi in enumerate(h):
                        yy0, yy1 = np.array(yy), np.array(yy) # copies
                        yy0[d] -= h[i]
                        yy1[d] += h[i]
                        try:
                                _a_i = 0.
                                for j in components:
                                        _a_i += func(j, yy1, *args) - func(j, yy0, *args)
                                a_prev.append(_a_i)
                        except (KeyboardInterrupt, SystemExit):
                                raise
                        except:
                                ok[i] = False
                a_prev = np.array(a_prev) / (2. * h[ok])
                # compute a_cur (a[:][j] in InferenceMAP) for j in 1,..
                fac = CON2
                for j in range(1, np.sum(ok)):
                        a_cur = (a_prev[1:] * fac - a_prev[:-1]) / (fac - 1.)
                        fac *= CON2
                        errt = np.maximum(np.abs(a_cur - a_prev[1:]), np.abs(a_cur - a_prev[:-1]))
                        ans_d = min(ans_d, a_cur[errt.argmin()])
                        a_prev[0], a_prev[1:] = 0., a_cur
                if np.isnan(ans_d) or np.isinf(ans_d):
                        return None
                        print('isnan(ans_d)')
                        ans_d = 0.
                ans.append(ans_d)
        return np.array(ans)


def slnsrch(func, p, ids, components, xi, g, fjp, fp, args=(), c1=1e-4, alpha=1.):
        def _func(*_args):
                try:
                        return func(*_args)
                except (KeyboardInterrupt, SystemExit):
                        raise
                except:
                        traceback.print_exc()
                        return np.inf
        f_count = 0
        slope = np.dot(g, xi)
        if 0 <= slope:
                print('positive slope')
                return 0., f_count, fjp, fp
        elif slope < -1e10:
                print('slope is too large')
                return 0., f_count, fjp, fp
        alphamin = c1 * c1 / np.max(np.abs(xi) / np.maximum(np.abs(p), 1.))
        #alphamin = c1 * c1 / np.max(np.abs(xi) / np.maximum(np.abs(p[ids]), 1.))
        xi = xi[ids]
        #g = g[ids] # not used
        fpi = np.sum(fjp[components])
        fp0, fp = fp - fpi, fpi
        pnew = np.array(p)
        f2 = np.nan
        while np.isnan(f2):
                if alpha < alphamin: # lnsrch failed
                        print('alpha < alphamin (1)')
                        return 0., f_count, fjp, fp0 + fp
                pnew[ids] = p[ids] + alpha * xi
                #pnew = p + alpha * xi
                try:
                        f2 = np.sum( func(j, pnew, *args) for j in components )
                except (KeyboardInterrupt, SystemExit):
                        raise
                except:
                        f2 = np.nan
                f_count += 1
                alpha2, alpha = alpha, .1 * alpha
        while True:
                if alpha < alphamin: # lnsrch failed
                        print('alpha < alphamin (2)')
                        if f2 < fp:
                                #fjp = np.array(fjp) # copy
                                fjp[components] = fjpnew
                                return alpha2, f_count, fjp, fp0+fpnew
                        else:
                                return 0., f_count, fjp, fp0 + fp
                pnew[ids] = p[ids] + alpha * xi
                #pnew = p + alpha * xi
                fjpnew = np.array([ _func(j, pnew, *args) for j in components ])
                fpnew = np.sum(fjpnew)
                f_count += 1
                if fpnew <= fp + c1 * alpha * slope:
                        #fjp = np.array(fjp) # copy
                        fjp[components] = fjpnew
                        return alpha, f_count, fjp, fp0 + fpnew
                else:
                        rhs1 = (fpnew - fp - alpha * slope) / (alpha * alpha)
                        rhs2 = (f2 - fp - alpha2 * slope) / (alpha2 * alpha2)
                        a = (rhs2 - rhs1) / (alpha2 - alpha)
                        b = (alpha2 * rhs1 - alpha * rhs2) / (alpha2 - alpha)
                        if a == 0.:
                                tmplam = -slope / (2. * b)
                        else:
                                discr = b * b - 3. * a * slope
                                if discr < 0.:
                                        tmplam = .5 * alpha
                                elif b <= 0.:
                                        tmplam = (-b + sqrt(discr)) / (3. * a)
                                else:
                                        tmplam = -slope / (b + sqrt(discr))
                        tmplam = min(tmplam, .5 * alpha)
                        if np.isnan(tmplam):
                                # candidate cause: `slope` is too large;
                                # let's consider lnsrch failed
                                print('isnan(tmplam)')
                                return 0., f_count, fjp, fp0 + fp
                                #return alpha, f_count, fjp, fp0 + fpnew
                # prepare next iteration
                alpha2, alpha = alpha, max(tmplam, .1 * alpha)
                f2 = fpnew

#warnings.filterwarnings('error', category=sparse.SparseWarning)
BFGSResult = namedtuple('BFGSResult', ('x', 'B', 'gerrs', 'f', 'projg', 'cumtime'))

def minimize_sgbfgs(minibatch, fun, x0, args=(), eta0=1., tau=1e2, c=1., l=1e-6, eps=1e-9,
        gtol=1e-4, gcount=10, ptol=1e-16, pcount=10, maxiter=1000000, maxcor=100, verbose=False):
        if verbose:
                def msg1(_i, _f, _dg):
                        _i = str(_i)
                        return 'At iterate {}{}\tf= {}{:E} \t|proj g| = {:E}\n'.format(
                                ' ' * max(0, 3 - len(_i)), _i, ' ' if 0 <= _f else '', _f, _dg)
                def msg2(_i, *_args):
                        msg = ''.join(('At iterate {}{}\t', ': '.join(['{}'] * len(_args)), '\n'))
                        _i = str(_i)
                        return msg.format(' ' * max(0, 3 - len(_i)), _i, *_args)
                cumt = 0.
                fs, projs = [], []
                t0 = time.time()
        gerrs = []
        # precompute arguments for dfunc
        h = max(1e-8, np.median(x0))
        CON = 1.4
        dfunc_h = h * ((1./CON) ** np.arange(maxcor))
        #
        def f(_x, _rows):
                return np.sum( fun(_j, _x, *args) for _j in _rows )
        #
        rows = minibatch(0, x0)
        try:
                rows, cols = rows
                twoways = True
                def df(_x, _rows, _cols):
                        return sdfunc(fun, _x, _cols, _rows, dfunc_h, args)
                g = df(x0, rows, cols)
        except (TypeError, ValueError):
                cols = None
                twoways = False
                def df(_x, _rows):
                        return sdfunc(fun, _x, range(_x.size), _rows, dfunc_h, args)
                g = df(x0, rows)
        if g is None:
                raise ValueError('gradient calculation failed')
        p = -eps * g
        s = eta0 / c * p
        if twoways:
                x = np.array(x0) # copy
                x[cols] += s
                h = df(x, rows, cols)
        else:
                x = x0 + s
                h = df(x, rows)
        if h is None:
                raise ValueError('gradient calculation failed')
        y = h - g + l * s
        proj = np.dot(s, y)
        if verbose:
                cumt += time.time() - t0
                try:
                        f1 = f(x, rows)
                except ValueError as e:
                        print(msg2(0, 'ValueError', e.args[0]))
                else:
                        print(msg1(0, f1, proj))
                        fs.append(f1)
                        projs.append(proj)
                t0 = time.time()
        # make B sparse and preallocate the nonzeros
        B = np.diag(np.full(x.size, proj / np.dot(y, y)))
        x2 = x0
        k1 = k2 = 0
        resolution = None
        for t in range(1, maxiter):
                if twoways:
                        rows, cols = minibatch(t, x)
                        g = df(x, rows, cols)
                else:
                        rows = minibatch(t, x)
                        g = df(x, rows)
                if g is None:
                        print(msg2(t, 'GRADIENT CALCULATION FAILED (t)'))
                        x = x2
                        #continue
                        resolution = 'GRADIENT CALCULATION FAILED (t)'
                        break
                if twoways:
                        p = -np.dot(B[np.ix_(cols, cols)], g)
                else:
                        p = -np.dot(B, g)
                eta = tau / (tau + float(t)) * eta0
                s = eta / c * p
                if twoways:
                        x1 = np.array(x)
                        x1[cols] += s
                        h = df(x1, rows, cols)
                else:
                        x1 = x + s
                        h = df(x1, rows)
                if h is None:
                        print(msg2(t, 'GRADIENT CALCULATION FAILED (t+1)'))
                        x = x2
                        #continue
                        resolution = 'GRADIENT CALCULATION FAILED (t+1)'
                        break
                x2, x = x, x1
                y = h - g
                gerr = np.dot(y, y)
                gerrs.append(gerr)
                if gerr < gtol:
                        k1 += 1
                        if gcount <= k1:
                                resolution = 'CONVERGENCE: |DELTA GRAD| < GTOL'
                                break
                else:
                        k1 = 0
                y += l * s
                proj = np.dot(s, y)
                if verbose:
                        cumt += time.time() - t0
                        try:
                                f1 = f(x, rows)
                        except ValueError as e:
                                print(msg2(t, 'ValueError', e.args[0]))
                        else:
                                print(msg1(t, f1, proj))
                                fs.append(f1)
                                projs.append(proj)
                        t0 = time.time()
                if abs(proj) < ptol:
                        k2 += 1
                        if pcount <= k2:
                                resolution = 'CONVERGENCE: |PROJ G| < PTOL'
                        break
                else:
                        k2 = 0
                # B = (I - rho s y.T) B ( I - rho y s.T) + c rho s s.T
                rho = 1. / proj
                if twoways:
                        q = -np.dot(B[np.ix_(cols, cols)], y)
                else:
                        q = -np.dot(B, y)
                phi = c - rho * np.dot(y, q) # rho s y.T B y s.T rho + c rho s s.T = rho phi s s.T
                if twoways:
                        B[np.ix_(cols, cols)] += \
                                rho * (np.outer(q, s) + np.outer(s, q) + phi * np.outer(s, s))
                else:
                        B += rho * (np.outer(q, s) + np.outer(s, q) + phi * np.outer(s, s))
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
                return BFGSResult(x, B, gerrs, fs, projs, cumt)
        else:
                return BFGSResult(x, B, gerrs, None, None, None)


def sdfunc(func, yy, ids, components, h, args=()):
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
                                a[0,i] = a_ip \
                                        = sum( func(j, yy1, *args) - func(j, yy0, *args) \
                                                for j in components ) \
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


def minimize_shbfgs(minibatch, fun, x0, args=(), eta0=1., tau=1e2, c=1., l=1e-6, eps=1e-9,
        gtol=1e-4, gcount=10, ptol=1e-16, pcount=10, maxiter=1e6, maxcor=100, covariates=None,
        verbose=False):
        if verbose:
                def msg1(_i, _f, _dg):
                        _i = str(_i)
                        return 'At iterate {}{}\tf= {}{:E} \t|proj g| = {:E}\n'.format(
                                ' ' * max(0, 3 - len(_i)), _i, ' ' if 0 <= _f else '', _f, _dg)
                def msg2(_i, *_args):
                        msg = ''.join(('At iterate {}{}\t', ': '.join(['{}'] * len(_args)), '\n'))
                        _i = str(_i)
                        return msg.format(' ' * max(0, 3 - len(_i)), _i, *_args)
                cumt = 0.
                fs, projs = [], []
                t0 = time.time()
        gerrs = []
        # precompute arguments for dfunc
        h = 1e-8#np.maximum(1e-8, 1e-4 * np.abs(x0))
        CON = 1.4
        dfunc_h = h * ((1./CON) ** np.arange(maxcor))
        #dfunc_h = np.outer(h, (1./CON) ** np.arange(maxcor))
        #
        def f(_x, _rows):
                return np.sum( fun(_j, _x, *args) for _j in _rows )
        #
        rows = minibatch(0, x0)
        try:
                rows, cols = rows
                twoways = True
                def df(_x, _rows, _cols):
                        return sdfunc(fun, _x, _cols, _rows, dfunc_h, args)
                g = df(x0, rows, cols)
        except (TypeError, ValueError):
                cols = None
                twoways = False
                def df(_x, _rows):
                        return sdfunc(fun, _x, range(_x.size), _rows, dfunc_h, args)
                g = df(x0, rows)
        if g is None:
                raise ValueError('gradient calculation failed')
        p = -eps * g
        s = eta0 / c * p
        if twoways:
                x = np.array(x0) # copy
                x[cols] += s
                h = df(x, rows, cols)
        else:
                x = x0 + s
                h = df(x, rows)
        if h is None:
                raise ValueError('gradient calculation failed')
        y = h - g + l * s
        proj = np.dot(s, y)
        if verbose:
                cumt += time.time() - t0
                try:
                        f1 = f(x, rows)
                except ValueError as e:
                        print(msg2(0, 'ValueError', e.args[0]))
                else:
                        print(msg1(0, f1, proj))
                        fs.append(f1)
                        projs.append(proj)
                t0 = time.time()
        # make B sparse and preallocate the nonzeros
        if twoways and covariates:
                indices, indptr = covariates
                B = sparse.csr_matrix((np.zeros(indices.size, dtype=x0.dtype), indices, indptr),
                        shape=(x0.size, x0.size))
                B[np.arange(B.shape[0]), np.arange(B.shape[1])] = proj / np.dot(y, y)
        else:
                B = np.diag(np.full(x.size, proj / np.dot(y, y)))
        x2 = x0
        k1 = k2 = 0
        resolution = None
        for t in range(1, maxiter):
                if twoways:
                        rows, cols = minibatch(t, x)
                        g = df(x, rows, cols)
                else:
                        rows = minibatch(t, x)
                        g = df(x, rows)
                if g is None:
                        x = x2
                        print(msg2(t, 'GRADIENT CALCULATION FAILED (t)'))
                        continue
                        resolution = 'GRADIENT CALCULATION FAILED (t)'
                        break
                if twoways:
                        if covariates:
                                p = np.ravel(-np.dot(B[np.ix_(cols, cols)].todense(), g))
                        else:
                                p = -np.dot(B[np.ix_(cols, cols)], g)
                else:
                        p = -np.dot(B, g)
                eta = tau / (tau + float(t)) * eta0
                s = eta / c * p
                if twoways:
                        x1 = np.array(x)
                        x1[cols] += s
                        h = df(x1, rows, cols)
                else:
                        x1 = x + s
                        h = df(x1, rows)
                if h is None:
                        x = x2
                        print(msg2(t, 'GRADIENT CALCULATION FAILED (t+1)'))
                        continue
                        resolution = 'GRADIENT CALCULATION FAILED (t+1)'
                        break
                x2, x = x, x1
                y = h - g
                gerr = np.dot(y, y)
                gerrs.append(gerr)
                if gerr < gtol:
                        k1 += 1
                        if gcount <= k1:
                                resolution = 'CONVERGENCE: |DELTA GRAD| < GTOL'
                                break
                else:
                        k1 = 0
                y += l * s
                proj = np.dot(s, y)
                if verbose:
                        cumt += time.time() - t0
                        try:
                                f1 = f(x, rows)
                        except ValueError as e:
                                print(msg2(t, 'ValueError', e.args[0]))
                        else:
                                print(msg1(t, f1, proj))
                                fs.append(f1)
                                projs.append(proj)
                        t0 = time.time()
                if abs(proj) < ptol:
                        k2 += 1
                        if pcount <= k2:
                                resolution = 'CONVERGENCE: |PROJ G| < PTOL'
                        break
                else:
                        k2 = 0
                # B = (I - rho s y.T) B ( I - rho y s.T) + c rho s s.T
                rho = 1. / proj
                if twoways:
                        if covariates:
                                q = np.ravel(-np.dot(B[np.ix_(cols, cols)].todense(), y))
                        else:
                                q = -np.dot(B[np.ix_(cols, cols)], y)
                else:
                        q = -np.dot(B, y)
                phi = c - rho * np.dot(y, q) # rho s y.T B y s.T rho + c rho s s.T = rho phi s s.T
                if twoways:
                        B[np.ix_(cols, cols)] += \
                                rho * (np.outer(q, s) + np.outer(s, q) + phi * np.outer(s, s))
                else:
                        B += rho * (np.outer(q, s) + np.outer(s, q) + phi * np.outer(s, s))
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
                return BFGSResult(x, B, gerrs, fs, projs, cumt)
        else:
                return BFGSResult(x, B, gerrs, None, None, None)

def minimize_sgbfgsb(minibatch, fun, x0, args=(), eta0=.1, tau=1e2, c=1., l=1e-6, eps=1e-9,
        gtol=1e-8, gcount=10, ptol=1e2, pcount=1e4, maxiter=100000, maxcor=50, covariates=None,
        lower_bounds=None, verbose=False):
        if verbose:
                def msg1(_i, _f, _dg):
                        _i = str(_i)
                        return 'At iterate {}{}\tf= {}{:E} \t|proj g| = {:E}\n'.format(
                                ' ' * max(0, 3 - len(_i)), _i, ' ' if 0 <= _f else '', _f, _dg)
                def msg2(_i, *_args):
                        msg = ''.join(('At iterate {}{}\t', ': '.join(['{}'] * len(_args)), '\n'))
                        _i = str(_i)
                        return msg.format(' ' * max(0, 3 - len(_i)), _i, *_args)
                cumt = 0.
                fs, projs = [], []
                t0 = time.time()
        gerrs = []
        # precompute arguments for dfunc
        h = 1e-8#np.maximum(1e-8, 1e-4 * np.abs(x0))
        CON = 1.4
        dfunc_h = h * ((1./CON) ** np.arange(maxcor))
        #dfunc_h = np.outer(h, (1./CON) ** np.arange(maxcor))
        #
        def f(_x, _rows):
                return np.sum( fun(_j, _x, *args) for _j in _rows )
        #
        rows = minibatch(0, x0)
        try:
                rows, cols = rows
                twoways = True
                def df(_x, _rows, _cols):
                        return sdfunc(fun, _x, _cols, _rows, dfunc_h, args)
                g = df(x0, rows, cols)
        except (TypeError, ValueError):
                cols = None
                twoways = False
                def df(_x, _rows):
                        return sdfunc(fun, _x, range(_x.size), _rows, dfunc_h, args)
                g = df(x0, rows)
        if g is None:
                raise ValueError('gradient calculation failed')
        p = -eps * g
        s = eta0 / c * p
        if twoways:
                x = np.array(x0) # copy
                x[cols] += s
                if lower_bounds is not None:
                        lb = lower_bounds[cols]
                        ok = ~lb.mask
                        if np.any(ok):
                                x_raw = x[cols[ok]]
                                x[cols[ok]] = x_corr = np.maximum(lb[ok], x_raw)
                                s[ok] += x_corr - x_raw
                h = df(x, rows, cols)
        else:
                x = x0 + s
                h = df(x, rows)
        if h is None:
                raise ValueError('gradient calculation failed')
        y = h - g + l * s
        proj = np.dot(s, y)
        if verbose:
                cumt += time.time() - t0
                try:
                        f1 = f(x, rows)
                except ValueError as e:
                        print(msg2(0, 'ValueError', e.args[0]))
                else:
                        print(msg1(0, f1, proj))
                        fs.append(f1)
                        projs.append(proj)
                t0 = time.time()
        # make B sparse and preallocate the nonzeros
        if twoways and (sparse.issparse(covariates) or covariates):
                if isinstance(covariates, tuple):
                        indices, indptr = covariates
                else:
                        indices, indptr = covariates.indices, covariates.indptr
                B = sparse.csr_matrix((np.zeros(indices.size, dtype=x0.dtype), indices, indptr),
                        shape=(x0.size, x0.size))
                B[cols, cols] = proj / np.dot(y, y)
                #B[np.arange(B.shape[0]), np.arange(B.shape[1])] = proj / np.dot(y, y)
                covariates = True
        else:
                raise NotImplementedError('fixme')
                B = np.diag(np.full(x.size, proj / np.dot(y, y)))
        x2 = x0
        k1 = k2 = 0
        resolution = None
        for t in range(1, int(maxiter)):
                if twoways:
                        rows, cols = minibatch(t, x)
                        g = df(x, rows, cols)
                else:
                        rows = minibatch(t, x)
                        g = df(x, rows)
                if g is None:
                        x = x2
                        print(msg2(t, 'GRADIENT CALCULATION FAILED (t)'))
                        continue
                        resolution = 'GRADIENT CALCULATION FAILED (t)'
                        break
                if twoways:
                        if covariates:
                                firsttime = np.all(B[cols,cols] == 0)
                                if firsttime:
                                        p = -eps * g
                                else:
                                        p = np.ravel(-np.dot(B[np.ix_(cols, cols)].todense(), g))
                        else:
                                p = -np.dot(B[np.ix_(cols, cols)], g)
                else:
                        p = -np.dot(B, g)
                eta = tau / (tau + float(t)) * eta0
                s = eta / c * p
                if twoways:
                        x1 = np.array(x)
                        x1[cols] += s
                        if lower_bounds is not None:
                                lb = lower_bounds[cols]
                                ok = ~lb.mask
                                if np.any(ok):
                                        x_raw = x1[cols[ok]]
                                        x1[cols[ok]] = x_corr = np.maximum(lb[ok], x_raw)
                                        s[ok] += x_corr - x_raw
                        h = df(x1, rows, cols)
                else:
                        x1 = x + s
                        h = df(x1, rows)
                if h is None:
                        x = x2
                        print(msg2(t, 'GRADIENT CALCULATION FAILED (t+1)'))
                        continue
                        resolution = 'GRADIENT CALCULATION FAILED (t+1)'
                        break
                x2, x = x, x1
                y = h - g
                gerr = np.dot(y, y)
                gerrs.append(gerr)
                if gerr < gtol:
                        k1 += 1
                        if gcount <= k1:
                                resolution = 'CONVERGENCE: |DELTA GRAD| < GTOL'
                                break
                else:
                        k1 = 0
                y += l * s
                proj = np.dot(s, y)
                if ptol < abs(proj):
                        x = x2
                        k2 += 1
                        msg = 'SOLUTION REJECTED: PTOL < |PROJ G|'
                        if k2 < pcount:
                                if verbose:
                                        print(msg2(t, msg))
                                continue
                        else:
                                resolution = msg
                                break
                if verbose:
                        cumt += time.time() - t0
                        try:
                                f1 = f(x, rows)
                        except ValueError as e:
                                print(msg2(t, 'ValueError', e.args[0]))
                        else:
                                print(msg1(t, f1, proj))
                                fs.append(f1)
                                projs.append(proj)
                        t0 = time.time()
                # B = (I - rho s y.T) B ( I - rho y s.T) + c rho s s.T
                rho = 1. / proj
                if twoways:
                        if covariates:
                                if firsttime:
                                        B[cols, cols] = proj / np.dot(y, y)
                                        q = -eps * y
                                else:
                                        q = np.ravel(-np.dot(B[np.ix_(cols, cols)].todense(), y))
                        else:
                                q = -np.dot(B[np.ix_(cols, cols)], y)
                else:
                        q = -np.dot(B, y)
                phi = c - rho * np.dot(y, q) # rho s y.T B y s.T rho + c rho s s.T = rho phi s s.T
                if twoways:
                        B[np.ix_(cols, cols)] += \
                                rho * (np.outer(q, s) + np.outer(s, q) + phi * np.outer(s, s))
                else:
                        B += rho * (np.outer(q, s) + np.outer(s, q) + phi * np.outer(s, s))
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
                return BFGSResult(x, B, gerrs, fs, projs, cumt)
        else:
                return BFGSResult(x, B, gerrs, None, None, None)



