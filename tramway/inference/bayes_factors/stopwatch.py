# -*- coding: utf-8 -*-

# Copyright © 2018, Alexander Serov

import time


class stopwatch:
    """A class for measuring execution time."""

    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        delta = self.end - self.start
        if self.verbose:
            print('\n{name} completed in {t} s.\n'.format(name=self.name, t=round(delta, 1)))


# def stopwatch_dec(func):
#     """An alternative decorator for measuring the elapsed time."""
#
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         results = func(*args, **kwargs)
#         delta = time.time() - start
#         print(f'\n{self.name} completed in {round(delta, 1)} s.\n')
#         return results
#     return wrapper
