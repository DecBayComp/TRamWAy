# k-d tree

from ..kdtree import KDTreeMesh
from collections import OrderedDict

setup = {
	'make': KDTreeMesh,
	'make_arguments': OrderedDict((
		('min_distance', ()),
		('avg_distance', ()),
		('min_probability', ()),
		('max_probability', ()),
		('max_location_count', ('-S', dict(type=int, help='maximum number of locations per cell'))),
		('max_level', ('-ll', '--lower-levels', dict(type=int, help='number of levels below the smallest one', metavar='LOWER_LEVELS'))),
		)),
	}

