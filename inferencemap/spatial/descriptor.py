
import numpy as np
import pandas as pd

def isstructured(x):
	"""Returns `True` if `x` has named columns."""
	if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
		return True
	else:
		try:
			return False or x.dtype.names
		except AttributeError:
			return False

def columns(x):
	"""Returns an iterable `c` over the named columns if `x = x[c]` holds.
	Raises a value error otherwise, or if columns are not named.""" 
	if isinstance(x, pd.DataFrame):
		return x.columns
	elif isinstance(x, pd.Series):
		return x.index
	elif x.dtype.names:
		return x.dtype.names
	else:
		raise ValueError('not structured')

