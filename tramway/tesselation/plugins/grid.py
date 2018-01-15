# regular grid

from ..base import RegularMesh
from collections import OrderedDict

setup = {
	'make': RegularMesh,
	'make_arguments': OrderedDict((
		('min_probability', ()),
		('avg_probability', ()),
		('max_probability', ()),
		('avg_location_count', dict(args=('-c', '--location-count'), kwargs=dict(type=int, default=80, help='average number of locations per cell'), translate=True)),
		)),
	}

