
"""

Runtime configuration.

"""

import warnings
__log__ = lambda msg: warnings.warn(msg, RuntimeWarning) # could call the logging module instead


__user_interaction__ = True


__available_packages__ = set()
__reported_missing_packages__ = set()
def __has_package__(pkg, report_if_missing=True):
    if pkg in __available_packages__:
        return True
    elif report_if_missing:
        if pkg not in __reported_missing_packages__:
            __reported_missing_packages__.add(pkg)
            __log__("package '{}' is missing".format(pkg))
    return False


# example usage:
#try:
#    import tqdm
#except ImportError:
#    pass
#else:
#    __available_packages__.add('tqdm')


__all__ = ['__user_interaction__', '__available_packages__', '__has_package__']

