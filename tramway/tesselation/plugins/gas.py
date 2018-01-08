# test plugin

from ..gas import GasMesh
from collections import OrderedDict

setup = {
	'name': ('gas', 'gng'),
	'make': GasMesh,
	'make_arguments': OrderedDict((
		('min_distance', ()),
		('avg_distance', ()),
		('max_distance', ()),
		('min_probability', ()),
		('avg_probability', ()),
		)),
	}
