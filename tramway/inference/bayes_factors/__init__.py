# import datetime
import sys
from collections import OrderedDict

from .calculate_bayes_factors import (calculate_bayes_factors,
                                      calculate_bayes_factors_for_one_cell)

# The package can be imported by just `import bayes_factors`.
__all__ = ['calculate_bayes_factors', 'calculate_bayes_factors_for_one_cell', 'setup']


if sys.version_info <= (3, 5):
    raise RuntimeError("Python 3.5+ is required for calculating Bayes factors")


def _bayes_factor(cells, B_threshold=None, verbose=False, **kwargs):
    if verbose:
        try:
            from tqdm import tqdm
        except:
            import logging
            logging.warning(
                "Consider installing `tqdm` package (`pip install tqdm`) to see Bayes factors calculation progress.")

            def tqdm(x): return x
    else:
        def tqdm(x): return x

    # TODO: use the same localization error as for inference
    # input arguments
    # loc_error
    localization_error = cells.get_localization_error(kwargs)
    if localization_error is None:
        raise RuntimeError("Localization error must be specified for calculating Bayes factors")
    # B_threshold
    if B_threshold is not None:
        kwargs['B_threshold'] = B_threshold
    # verbose
    if verbose is not None:
        kwargs['verbose'] = verbose

    # iterate over the cells
    for key in tqdm(cells):
        calculate_bayes_factors_for_one_cell(cells[key], localization_error, **kwargs)


setup = {
    'name': 'bayes_factor',
    'infer': '_bayes_factor',
    'arguments': OrderedDict((
        ('localization_error', ('-e', dict(type=float, help='localization error (same units as the variance)'))),
        ('B_threshold', ('-b', dict(type=float, help='values of Bayes factor for thresholding'))),
        ('verbose', ()),
    )),
    # List of variables that the module returns as cell properties, e.g. cell.lg_B
    'returns': ['lg_B', 'force', 'min_n'],
}
