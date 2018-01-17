# k means

from ..kmeans import KMeansMesh
from collections import OrderedDict

def _metric(knn=None, **kwargs):
	if isinstance(knn, (tuple, list)):
		knn = knn[0]
	if knn is None:
		return None
	else:
		return 'euclidian'

setup = {
	'make': KMeansMesh,
	'make_arguments': OrderedDict((
		('min_distance', ()),
		('avg_probability', ()),
		('avg_location_count', dict(args=('-c', '--location-count'), kwargs=dict(type=int, default=80, help='average number of locations per cell'), translate=True)),
		('metric', dict(parse=_metric)),
		)),
	}

