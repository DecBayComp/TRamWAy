# -*- coding: utf-8 -*-

# Copyright © 2018, Alexander Serov


import logging
import math
import unittest
from multiprocessing import freeze_support

import numpy as np
from scipy.integrate import dblquad, quad

from .calculate_bayes_factors import calculate_bayes_factors
from .calculate_marginalized_integral import calculate_marginalized_integral
from .calculate_posteriors import (calculate_one_1D_posterior_in_2D,
                                   calculate_one_1D_prior_in_2D,
                                   calculate_one_2D_posterior)
from .convenience_functions import n_pi_func, p


class bayes_test(unittest.TestCase):
    """Tests for Bayes factor calculations"""

    def setUp(self):
        """Initialize data for different tests"""
        self.full_tests = False
        self.tol = 1e-32
        self.rel_tol = 1e-2
        self.B_threshold = 10.0
        self.t1_zeta_sps = np.asarray([[1.0, 1.0], [2.0, 3.0]])
        self.t1_zeta_ts = np.asarray([[2.0, 1.54], [-4.0, 0.0]])
        self.t1_ns = np.asarray([10, 5])
        self.t1_Vs = np.asarray([1, 0.2])
        self.t1_Vs_pi = np.asarray([0.3, 0.5])
        self.t1_us = np.divide(self.t1_Vs_pi, self.t1_Vs)

    def test_marginalized_integral(self):
        # Load data
        dim = 2
        n_pi = n_pi_func(dim)

        # >> Test case E = 0 <<
        zeta_t = [0.7, 0.4]
        zeta_sp = [-0.3, -0.2]
        n = 20
        u = 0.95
        rel_loc_error = 0.77
        eta = np.sqrt(n_pi / (n + n_pi))
        pow = p(n, dim)
        v0 = 1 + n_pi / n * u
        res = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                              p=pow, v=v0, E=0.0, rel_loc_error=rel_loc_error)
        true_res = 3.496167136E-23
        self.assertTrue(np.isclose(res, true_res, rtol=self.rel_tol, atol=self.tol),
                        "Marginalized integral calculation test failed for E = 0")

        # >> Check result with 0 break points <<
        zeta_t = [0.7, 0.4]
        zeta_sp = [-0.3, -0.2]
        n = 20
        u = 0.95
        rel_loc_error = 0.77
        eta = np.sqrt(n_pi / (n + n_pi))
        # pow = dim * (n + n_pi + 1.0) / 2.0 - 2.0
        pow = p(n, dim)
        v0 = 1 + n_pi / n * u
        # print([n_pi, n, u, v0])
        res = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                              p=pow, v=v0, E=eta ** 2.0, rel_loc_error=rel_loc_error)
        true_res = 3.182820939E-23
        self.assertTrue(np.isclose(res, true_res, rtol=self.rel_tol, atol=self.tol),
                        "Marginalized integral calculation test failed for 0 break points. The obtained value %.2g does not match the expected %.2g" % (res, true_res))

        # >> Check result with 1 break points <<
        zeta_t = [0.7, 0.4]
        zeta_sp = [-0.3, 0.6]
        n = 20
        u = 0.95
        rel_loc_error = 0.42
        eta = np.sqrt(n_pi / (n + n_pi))
        pow = p(n, dim)
        v0 = 1 + n_pi / n * u
        res = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                              p=pow, v=v0, E=eta ** 2.0, rel_loc_error=rel_loc_error)
        true_res = 1.457477573E-28
        self.assertTrue(np.isclose(res, true_res, rtol=self.rel_tol, atol=self.tol),
                        "Marginalized integral calculation test failed for 1 break points. The obtained value %.2g does not match the expected %.2g" % (res, true_res))

        # >> Check result with 2 break points <<
        zeta_t = [0.7, 0.4]
        zeta_sp = [0.8, 0.6]
        n = 20
        u = 0.95
        rel_loc_error = 2.30
        eta = np.sqrt(n_pi / (n + n_pi))
        pow = p(n, dim)
        v0 = 1 + n_pi / n * u
        res = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                              p=pow, v=v0, E=eta ** 2.0, rel_loc_error=rel_loc_error)
        true_res = 6.019713659E-14
        self.assertTrue(np.isclose(res, true_res, rtol=self.rel_tol, atol=self.tol),
                        "Marginalized integral calculation test failed for 2 break points. The obtained value %.8g does not match the expected %.8g" % (res, true_res))

        # >> Check the result with zeta_sp = 0 <<
        zeta_t = [0.7, 0.4]
        zeta_sp = [0.0, 0.0]
        n = 20
        u = 0.95
        rel_loc_error = 1.47
        eta = np.sqrt(n_pi / (n + n_pi))
        pow = p(n, dim)
        v0 = 1 + n_pi / n * u
        res = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                              p=pow, v=v0, E=eta ** 2.0, rel_loc_error=rel_loc_error)
        true_res = 1.145033778E-17
        self.assertTrue(np.isclose(res, true_res, rtol=self.rel_tol, atol=self.tol),
                        "Marginalized integral calculation test failed for zeta_sp = 0. The obtained value %.8g does not match the expected %.8g" % (res, true_res))

        # >> Check the result with zeta_a <<
        zeta_t = [0.7, 0.4]
        zeta_sp = [0.8, 0.6]
        zeta_a = [-0.1, -0.1]
        n = 20
        u = 0.95
        rel_loc_error = 1.47
        eta = np.sqrt(n_pi / (n + n_pi))
        pow = p(n, dim)
        v0 = 1 + n_pi / n * u
        res = calculate_marginalized_integral(zeta_t=zeta_t, zeta_sp=zeta_sp,
                                              p=pow, v=v0, E=eta ** 2.0, rel_loc_error=rel_loc_error, zeta_a=zeta_a)
        true_res = 1.225618999E-17
        self.assertTrue(np.isclose(res, true_res, rtol=self.rel_tol, atol=self.tol),
                        "Marginalized integral calculation test failed with zeta_a. The obtained value %.8g does not match the expected %.8g" % (res, true_res))

    def test_bayes_factors(self):

        # >> Test one input bin <<
        zeta_ts = np.asarray([[0.7, 0.4]])
        zeta_sps = np.asarray([[0.8, 0.6]])
        ns = np.asarray([[20]])
        Vs = np.asarray([[0.8 ** 2.0]])
        us = np.asarray([[0.95]])
        loc_error = 0.1**2.0
        Vs_pi = us * Vs
        lg_Bs, forces, _ = calculate_bayes_factors(
            zeta_ts=zeta_ts, zeta_sps=zeta_sps, ns=ns, Vs=Vs, Vs_pi=Vs_pi, loc_error=loc_error)
        true_B = 0.2974282533

        # Check value
        self.assertTrue(np.isclose(10**lg_Bs[0], true_B, rtol=self.rel_tol, atol=self.tol),
                        "Bayes factor calculation failed for one bin. The obtained B = %.8g does not match the expected B = %.8g" % (10**lg_Bs[0, 0], true_B))
        # Check force presence
        self.assertTrue((true_B >= self.B_threshold) ==
                        forces[0], "Boolean conservative force return incorrect for the case of one bin")

        # >> Test zero localization error <<
        zeta_ts = np.asarray([[0.7, 0.4]])
        zeta_sps = np.asarray([[0.8, 0.6]])
        ns = np.asarray([[20]])
        Vs = np.asarray([[0.8 ** 2.0]])
        us = np.asarray([[0.95]])
        loc_error = 0
        Vs_pi = us * Vs
        lg_Bs, forces, _ = calculate_bayes_factors(
            zeta_ts=zeta_ts, zeta_sps=zeta_sps, ns=ns, Vs=Vs, Vs_pi=Vs_pi, loc_error=loc_error)
        true_B = 0.2974282533

        # Check value
        self.assertTrue(np.isclose(10**lg_Bs[0], true_B, rtol=self.rel_tol, atol=self.tol),
                        "Bayes factor calculation failed for one bin. The obtained B = %.8g does not match the expected B = %.8g" % (10**lg_Bs[0, 0], true_B))
        # Check force presence
        self.assertTrue((true_B >= self.B_threshold) ==
                        forces[0], "Boolean conservative force return incorrect for the case of one bin")

        # >> Test three input bins <<
        zeta_ts = np.asarray([[0.4, 0.3], [-1.0, 2.3], [-1.1, -0.33]])
        zeta_sps = np.asarray([[0.8, 0.6], [-0.45, 0.67], [6.32, 0.115]])
        ns = np.asarray([[500, 100, 3]])
        Vs = np.asarray([[0.4 ** 2.0, 1.25 ** 2.0, 0.3 ** 2.0]])
        us = np.asarray([[0.95, 8.7, 2.45]])
        loc_error = 0.1**2.0
        ns = ns.T
        Vs = Vs.T
        us = us.T
        Vs_pi = us * Vs
        N = len(ns)
        lg_Bs, forces, _ = calculate_bayes_factors(
            zeta_ts=zeta_ts, zeta_sps=zeta_sps, ns=ns, Vs=Vs, Vs_pi=Vs_pi, loc_error=loc_error)
        true_Bs = [0.05997370802, 3.666049402e49, 1.378104868]
        # print(Bs)
        # print (forces)

        for i in range(N):
            # Check value
            self.assertTrue(np.isclose(10**lg_Bs[i], true_Bs[i], rtol=self.rel_tol, atol=self.tol),
                            "Bayes factor calculation failed for one bin. For bin no. %i, the obtained B = %.8g does not match the expected B = %.8g" % (i + 1, 10**lg_Bs[i], true_Bs[i]))
            # Check force presence
            true_forces = (1 * (np.log10(true_Bs[i]) >= np.log10(self.B_threshold)) -
                           1 * (np.log10(true_Bs[i]) <= -np.log10(self.B_threshold)))
            self.assertTrue(true_forces == forces[i],
                            "For bin %i, the boolean force prediction (%r) did not correspond to expected value (%r)" % (i, true_Bs[i] >= self.B_threshold, forces[i]))

    def test_minimal_n(self):
        """
        Test calculations of the minimal number of jumps per bin to produce strong evidence.
        """
        # Load data
        B_threshold = 10
        # >> Test three input bins <<
        zeta_ts = np.asarray([[0.7, 0.5], [-1.0, 2.3], [-1.1, -0.33]])
        zeta_sps = np.asarray([[0.8, 0.6], [-0.45, 0.67], [6.32, 0.115]])
        ns = np.asarray([[20, 100, 3]])
        Vs = np.asarray([[0.8 ** 2.0, 1.25 ** 2.0, 0.3 ** 2.0]])
        us = np.asarray([[0.95, 8.7, 2.45]])
        loc_error = 0.2 ** 2.0
        ns = ns.T
        Vs = Vs.T
        us = us.T
        Vs_pi = us * Vs
        # N = len(ns)

        # Calculate min_ns
        lg_Bs, _, min_ns = calculate_bayes_factors(
            zeta_ts=zeta_ts, zeta_sps=zeta_sps, ns=ns, Vs=Vs, Vs_pi=Vs_pi, loc_error=loc_error)
        # Calculate Bs for min_ns to check
        lg_Bs, _, _ = calculate_bayes_factors(
            zeta_ts=zeta_ts, zeta_sps=zeta_sps, ns=min_ns, Vs=Vs, Vs_pi=Vs_pi, loc_error=loc_error)

        # Perform check
        self.assertTrue(np.all(np.abs(lg_Bs[0]) >= np.log10(B_threshold)),
                        "Not all of the Bayes factors for the minimal ns returned strong evidence. lg_Bs: %s" % (lg_Bs))
        # print(Bs)

    def test_posteriors_and_priors(self):

        # >> Test that the 2D zeta_a posterior is normalized <<
        zeta_t = np.asarray([0.7, 0.4])
        zeta_sp = np.asarray([0.8, 0.6])
        n = np.asarray(20)
        V = np.asarray(0.8 ** 2.0)
        u = np.asarray(0.95)
        loc_error = 0.1**2.0
        V_pi = u * V
        lamb = 'marg'
        tol = 1e-4

        # Wrapper to get just a function of zeta_ax, zeta_ay
        def integrate_me(zay, zax):
            return calculate_one_2D_posterior(zeta_t=zeta_t, zeta_sp=zeta_sp, zeta_a=[zax, zay], n=n, V=V, V_pi=V_pi, loc_error=loc_error, dim=2, lamb=lamb)

        lims = np.array([-1, 1]) * 10

        if self.full_tests:
            norm = dblquad(integrate_me, lims[0], lims[1],
                           lims[0], lims[1], epsabs=tol, epsrel=tol)[0]
            true_norm = 1

            # Check value
            self.assertTrue(np.isclose(norm, true_norm, rtol=self.rel_tol, atol=self.tol),
                            "Active force posterior normalization test failed. Obtained norm = %.8g did not match the expected norm = %.8g" % (norm, true_norm))

        # >> Test that 1D zeta_ax posterior is normalized in 2D with localization error <<
        zeta_t = np.asarray([0.7, 0.4])
        zeta_sp = np.asarray([0.8, 0.6])
        n = np.asarray(20)
        V = np.asarray(0.8 ** 2.0)
        u = np.asarray(0.95)
        loc_error = 0.1**2.0
        V_pi = u * V
        lamb = 0  # 'marg'
        tol = 1e-4

        # Wrapper to get just a function of zeta_ax, zeta_ay
        def integrate_me(zax):
            return calculate_one_1D_posterior_in_2D(zeta_t=zeta_t, zeta_sp=zeta_sp, zeta_a=zax, n=n, V=V, V_pi=V_pi, loc_error=loc_error, lamb=lamb)

        lims = np.array([-1, 1]) * 10

        if self.full_tests:
            norm = quad(integrate_me, lims[0], lims[1], epsabs=tol, epsrel=tol)[0]
            true_norm = 1

            # Check value
            self.assertTrue(np.isclose(norm, true_norm, rtol=self.rel_tol, atol=self.tol),
                            "Active force posterior normalization test failed for 1D posteriors in 2D. Obtained norm = %.8g did not match the expected norm = %.8g" % (norm, true_norm))

        # >> Test that 1D zeta_ax posterior is normalized in 2D without localization error <<
        zeta_t = np.asarray([0.7, 0.4])
        zeta_sp = np.asarray([0.8, 0.6])
        n = np.asarray(20)
        V = np.asarray(0.8 ** 2.0)
        u = np.asarray(0.95)
        loc_error = 0
        V_pi = u * V
        lamb = 'marg'
        tol = 1e-4

        # Wrapper to get just a function of zeta_ax, zeta_ay
        def integrate_me(zax):
            return calculate_one_1D_posterior_in_2D(zeta_t=zeta_t, zeta_sp=zeta_sp, zeta_a=zax, n=n, V=V, V_pi=V_pi, loc_error=loc_error, lamb=lamb)

        lims = np.array([-1, 1]) * 10

        norm = quad(integrate_me, lims[0], lims[1], epsabs=tol, epsrel=tol)[0]
        true_norm = 1

        # Check value
        self.assertTrue(np.isclose(norm, true_norm, rtol=self.rel_tol, atol=self.tol),
                        "Active force posterior normalization test failed for 1D posteriors in 2D. Obtained norm = %.8g did not match the expected norm = %.8g" % (norm, true_norm))

        # >> Test that 1D zeta_ax PRIOR is normalized in 2D with localization error <<
        V = np.asarray(0.8 ** 2.0)
        u = np.asarray(0.95)
        sigma2 = 0.1**2.0
        V_pi = u * V
        rtol = 1e-6
        atol = 1e-4

        def integrate_me(zax):
            return calculate_one_1D_prior_in_2D(
                zeta_a=zax, V_pi=V_pi, sigma2=sigma2)

        norm = quad(integrate_me, -np.inf, np.inf, epsrel=rtol)[0]
        true_norm = 1
        self.assertTrue(np.isclose(norm, true_norm, rtol=rtol, atol=atol),
                        "Active force prior normalization test failed for 1D posteriors in 2D. Obtained norm = %.8g did not match the expected norm = %.8g" % (norm, true_norm))

        # >> Test that 1D zeta_ax PRIOR is normalized in 2D without localization error <<
        V = np.asarray(0.8 ** 2.0)
        u = np.asarray(0.95)
        sigma2 = 0
        V_pi = u * V
        rtol = 1e-6
        atol = 1e-4

        def integrate_me(zax):
            return calculate_one_1D_prior_in_2D(
                zeta_a=zax, V_pi=V_pi, sigma2=sigma2)

        norm = quad(integrate_me, -np.inf, np.inf, epsrel=rtol)[0]
        true_norm = 1
        self.assertTrue(np.isclose(norm, true_norm, rtol=rtol, atol=atol),
                        "Active force prior normalization test failed for 1D posteriors in 2D. Obtained norm = %.8g did not match the expected norm = %.8g" % (norm, true_norm))

    def test_diffusivity_posterior(self):
        from .get_D_posterior import get_D_posterior, get_MAP_D
        # >> Test  diffusivity posterior norm with localization error <<
        zeta_t = np.asarray([0.7, 0.4])
        zeta_sp = np.asarray([0.8, 0.6])
        n = np.asarray(20)
        V = np.asarray(0.8 ** 2.0)
        u = np.asarray(0.95)
        loc_errors = [0, 0.1**2.0]
        V_pi = u * V
        tol = 1e-16
        dt = 0.04
        lims = np.array([0, 1e3])

        for dim in range(1, 3):
            for loc_error in loc_errors:

                # Use MAP D estimate as an integration breakpoint
                MAP_D = get_MAP_D(n=n, zeta_t=zeta_t, V=V, V_pi=V_pi,
                                  dt=dt, sigma2=loc_error, dim=2)
                posterior = get_D_posterior(n=n, zeta_t=zeta_t, V=V,
                                            V_pi=V_pi, dt=dt, sigma2=loc_error, dim=2)

                norm = quad(posterior, lims[0], lims[1], points=[MAP_D], epsabs=tol, epsrel=tol)[0]
                true_norm = 1
                self.assertTrue(np.isclose(norm, true_norm, rtol=self.rel_tol, atol=self.tol),
                                "{dim}D diffusivity posterior normalization test with localization error = {loc_error} failed in 2D. Obtained norm = {norm:.8g} did not match the expected norm = {true_norm:.8g}".format(dim=dim, loc_error=loc_error, norm=norm, true_norm=true_norm))


# # A dirty fix for a weird bug in unittest
# if __name__ == '__main__':
#     freeze_support()
# unittest.main()
