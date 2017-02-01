
import numpy as np
import pandas as pd
#from ..inference import Cell, Distributed
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def scalar_map_2d(cell_map, map_name=None):
	ix, xy = zip(*[ (i, c.center) for i, c in cell_map.cells.items() ])
	xy = np.array(xy)
	xy_min = xy.min(axis=0)
	xy_max = xy.max(axis=0)

	voronoi = Voronoi(xy)
	if map_name is None:
		map_name = cell_map.infered.columns[0]
	scalar_map = cell_map.infered[map_name]

	polygons = []
	ok = np.ones(xy.shape[0], dtype=bool)
	for c, r in enumerate(voronoi.point_region):
		region = [ v for v in voronoi.regions[r] if 0 <= v ]
		if region[1:]:
			vertices = voronoi.vertices[region]
			polygons.append(Polygon(vertices, True))
		else:
			ok[c] = False
	scalar_map = np.asarray(scalar_map)[ok]

	fig, ax = plt.subplots() # before PatchCollection

	patches = PatchCollection(polygons, alpha=0.9)
	patches.set_array(np.asarray(scalar_map))
	ax.add_collection(patches)

	ax.set_xlim(xy_min[0], xy_max[0])
	ax.set_ylim(xy_min[1], xy_max[1])

	fig.colorbar(patches, ax=ax)

