
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

class MisplacedAttributeWarning(UserWarning):
    pass

class SideEffectWarning(UserWarning):
    pass

class RWAFileException(IOError):
    def __init__(self, filepath=None, exc=None):
        self.filepath = filepath
        self.exc = exc
    def __str__(self):
        if self.filepath is None:
            return 'cannot find any analysis tree'
        else:
            return 'cannot find any analysis tree in file: '+self.filepath

