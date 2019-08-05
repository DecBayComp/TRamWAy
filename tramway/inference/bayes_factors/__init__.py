# import datetime
import logging
import sys
from collections import OrderedDict

import numpy as np

from tramway.tessellation.base import point_adjacency_matrix

from .calculate_bayes_factors import (NaNInputError, calculate_bayes_factors,
                                      calculate_bayes_factors_for_one_cell)
from .group_by_sign import group_by_sign

# The package can be imported by just `import bayes_factors`.
__all__ = ['calculate_bayes_factors', 'calculate_bayes_factors_for_one_cell', 'setup']


if sys.version_info < (3, 5):
    raise RuntimeError("Python 3.5+ is required for calculating Bayes factors")


def _bayes_factor(cells, B_threshold=None, verbose=True, **kwargs):
    if verbose:
        try:
            from tqdm import tqdm
        except:
            logging.warning(
                "Consider installing `tqdm` package (`pip install tqdm`) to see Bayes factors calculation progress.")

            def tqdm(x, desc=None): return x
    else:
        def tqdm(x, desc=None): return x

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
    nan_cells_list = []
    for key in tqdm(cells):
        try:
            calculate_bayes_factors_for_one_cell(cells[key], localization_error, **kwargs)
        except NaNInputError:
            nan_cells_list.append(key)

    # Report error if any
    if len(nan_cells_list) > 0:
        logging.warn(
            "A NaN value was present in the input parameters for the following cells: {nan_cells_list}.\nBayes factor calculations were skipped for them".format(nan_cells_list=nan_cells_list))

    # Group cells by Bayes factor
    group_by_sign(cells=cells, tqdm=tqdm, **kwargs)


setup = {
    'name': 'bayes_factor',
    'infer': '_bayes_factor',
    'arguments': OrderedDict((
        ('localization_error', ('-e', dict(type=float, help='localization error (same units as the variance)'))),
        ('B_threshold', ('-b', dict(type=float, help='values of Bayes factor for thresholding'))),
        ('verbose', ()),
    )),
    # List of variables that the module returns as cell properties, e.g. cell.lg_B
    'returns': ['lg_B', 'force', 'min_n', 'groups', 'group_lg_B', 'group_forces'],
}
