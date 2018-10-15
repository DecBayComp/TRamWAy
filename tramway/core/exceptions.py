
try:
    from .lazy import PermissionError
except ImportError:
    pass

class NaNWarning(RuntimeWarning):
    pass

class EfficiencyWarning(RuntimeWarning):
    pass

class MissingSupportWarning(UserWarning):
    pass

class IOWarning(Warning):
    pass

class FileNotFoundWarning(IOWarning):
    pass

class MultipleArgumentError(ValueError):
    def __str__(self):
        assert self.args and self.args[1:] and all( isinstance(a, str) for a in self.args )
        args = [ "'{}'".format(a) for a in self.args ]
        s = ' or '.join((', '.join(args[:-1]), args[-1]))
        return 'please define either ' + s

