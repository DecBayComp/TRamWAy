from .calculate_bayes_factors import calculate_bayes_factors, calculate_bayes_factors_for_one_cell

# Only this function is supposed to be used publicly.
# The package can be imported by just `import bayes_factors`.
__all__ = ['calculate_bayes_factors', 'setup']


from collections import OrderedDict

def _bayes_factor(cells, localization_error=None, B_threshold=None, **kwargs):
    ## input arguments
    # loc_error
    if localization_error is None:
        loc_error = 0.
    else:
        loc_error = localization_error
    # B_threshold
    if B_threshold is not None:
        kwargs['B_treshold'] = B_threshold
    ## iterate over the cells
    for i in cells:
        calculate_bayes_factors_for_one_cell(cells[i], loc_error, **kwargs)

setup = {
    'name': 'bayes_factor',
    'infer': '_bayes_factor',
    'arguments': OrderedDict((
        ('localization_error', ('-e', dict(type=float, help='localization error'))),
        ('B_threshold', ('-b', dict(type=float, help='values of Bayes factor for thresholding'))),
        )),
    'returns': ['lg_B', 'force', 'min_n'],
    }

