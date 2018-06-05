
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

