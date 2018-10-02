"""
Small functions used throughout the package
"""


def n_pi_func(dim):
    return 5 - dim


def p(n, dim):
    n_pi = n_pi_func(dim)
    p = dim * (n + n_pi - 1.0) / 2.0 - 1.0
    return p
