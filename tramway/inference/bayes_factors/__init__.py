import sys
from collections import OrderedDict

from .calculate_bayes_factors import (calculate_bayes_factors,
                                      calculate_bayes_factors_for_one_cell)

# The package can be imported by just `import bayes_factors`.
__all__ = ['calculate_bayes_factors', 'calculate_bayes_factors_for_one_cell', 'setup']


if sys.version_info <= (3, 5):
    raise RuntimeError("Python 3.5+ is required for calculating Bayes factors")


def _bayes_factor(cells, localization_error=None, B_threshold=None, **kwargs):
    # TODO: use the same localization error as for inference
    # input arguments
    # loc_error
    if localization_error is None:
        raise RuntimeError("Localization error must be specified for calculating Bayes factors")
    # B_threshold
    if B_threshold is not None:
        kwargs['B_threshold'] = B_threshold

    # iterate over the cells
    for i in cells:
        calculate_bayes_factors_for_one_cell(cells[i], localization_error, **kwargs)


setup = {
    'name': 'bayes_factor',
    'infer': '_bayes_factor',
    'arguments': OrderedDict((
        ('localization_error', ('-e', dict(type=float, help='localization error (same units as the variance)'))),
        ('B_threshold', ('-b', dict(type=float, help='values of Bayes factor for thresholding'))),
    )),
    'returns': ['lg_B', 'force', 'min_n'],
}
