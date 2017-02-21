
import numpy as np
import pandas as pd
import numpy.ma as ma
from inferencemap.tesselation import Voronoi
from inferencemap.inference import Distributed
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import scipy.spatial
from warnings import warn

def scalar_map_2d(cells, values):
	if isinstance(values, pd.DataFrame):
		if values.shape[1] != 1:
			warn('multiple parameters available; mapping first one only', UserWarning)
		values = values.iloc[:,0] # to Series
	#values = pd.to_numeric(values, errors='coerce')

	if isinstance(cells, Distributed):
		ix, xy, ok = zip(*[ (i, c.center, 0 < c.tcount) for i, c in cells.cells.items() ])
		ix, xy, ok = np.array(ix), np.array(xy), np.array(ok)
		voronoi = scipy.spatial.Voronoi(xy)
		polygons = []
		for c, r in enumerate(voronoi.point_region):
			if ok[c]:
				region = [ v for v in voronoi.regions[r] if 0 <= v ]
				if region[1:]:
					vertices = voronoi.vertices[region]
					polygons.append(Polygon(vertices, True))
	elif isinstance(cells, Voronoi):
		raise NotImplementedError

	xy_min, xy_max = xy.min(axis=0), xy.max(axis=0)

	scalar_map = values.loc[ix[ok]].values

	#print(np.nonzero(~ok)[0])

	if np.any(np.isnan(scalar_map)):
		warn('NaNs ; changing them into 0s', RuntimeWarning)
		scalar_map[np.isnan(scalar_map)] = 0

	fig, ax = plt.subplots() # before PatchCollection
	patches = PatchCollection(polygons, alpha=0.9)
	patches.set_array(scalar_map)
	ax.add_collection(patches)

	ax.set_xlim(xy_min[0], xy_max[0])
	ax.set_ylim(xy_min[1], xy_max[1])

	try:
		fig.colorbar(patches, ax=ax)
	except AttributeError as e:
		warn(e.args[0], RuntimeWarning)

