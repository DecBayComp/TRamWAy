# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from math import tan, atan2, degrees, radians
import numpy as np
import pandas as pd
import numpy.ma as ma
from tramway.core.exceptions import NaNWarning
from tramway.tessellation import *
from tramway.inference import Distributed
from matplotlib.patches import Polygon, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.linalg as la
import scipy.sparse as sparse
from warnings import warn


def scalar_map_2d(cells, values, aspect=None, clim=None, figure=None, axes=None, linewidth=1,
		delaunay=False, colorbar=True, **kwargs):
	if isinstance(values, pd.DataFrame):
		if values.shape[1] != 1:
			warn('multiple parameters available; mapping first one only', UserWarning)
		values = values.iloc[:,0] # to Series
	#values = pd.to_numeric(values, errors='coerce')

	if delaunay:
		delaunay_linewidth = linewidth
		linewidth = 0

	polygons = []
	if isinstance(cells, Distributed):
		ix, xy, ok = zip(*[ (i, c.center, 0 < c.tcount) for i, c in cells.cells.items() ])
		ix, xy, ok = np.array(ix), np.array(xy), np.array(ok)
		voronoi = scipy.spatial.Voronoi(xy)
		for c, r in enumerate(voronoi.point_region):
			if ok[c]:
				region = [ v for v in voronoi.regions[r] if 0 <= v ]
				if region[1:]:
					vertices = voronoi.vertices[region]
					polygons.append(Polygon(vertices, True))
	elif isinstance(cells, CellStats) and isinstance(cells.tessellation, Voronoi):
		Av = cells.tessellation.vertex_adjacency.tocsr()
		xy = cells.tessellation.cell_centers
		ix = np.arange(xy.shape[0])
		ok = 0 < cells.location_count
		for i in ix[ok]:
			vs = cells.tessellation.cell_vertices[i]
			# order the vertices so that they draw a polygon
			v = vs[0]
			vs = set(list(vs))
			vertices = []
			while True:
				vertices.append(cells.tessellation.vertices[v])
				vs.remove(v)
				ws = set(Av.indices[Av.indptr[v]:Av.indptr[v+1]]) & vs
				if not ws:
					break
				v = ws.pop()
			#
			if vertices:
				vertices = np.vstack(vertices)
				polygons.append(Polygon(vertices, True))
	else:
		_type = repr(type(cells))
		if _type.endswith("'>"):
			_type = _type.split("'")[1]
		try:
			_nested_type = repr(type(cells.tessellation))
			if _nested_type.endswith("'>"):
				_nested_type = _nested_type.split("'")[1]
			raise TypeError('wrong type for `cells`: {}<{}>'.format(_type, _nested_type))
		except AttributeError:
			raise TypeError('wrong type for `cells`: {}'.format(_type))

	xy_min, xy_max = xy.min(axis=0), xy.max(axis=0)

	scalar_map = values.loc[ix[ok]].values

	#print(np.nonzero(~ok)[0])

	try:
		if np.any(np.isnan(scalar_map)):
			#print(np.nonzero(np.isnan(scalar_map)))
			msg = 'NaNs ; changing them into 0s'
			try:
				warn(msg, NaNWarning)
			except:
				print('warning: {}'.format(msg))
			scalar_map[np.isnan(scalar_map)] = 0
	except TypeError as e: # help debug
		print(scalar_map)
		print(scalar_map.dtype)
		raise e

	if figure is None:
		figure = plt.gcf() # before PatchCollection
	if axes is None:
		axes = figure.gca()
	patches = PatchCollection(polygons, alpha=0.9, linewidth=linewidth)
	patches.set_array(scalar_map)
	if clim is not None:
		patches.set_clim(clim)
	axes.add_collection(patches)

	if delaunay:
		try:
			import tramway.plot.mesh as mesh
			obj = mesh.plot_delaunay(cells, centroid_style=None,
				linewidth=delaunay_linewidth,
				axes=axes, **kwargs)
		except:
			import traceback
			print(traceback.format_exc())
			pass

	axes.set_xlim(xy_min[0], xy_max[0])
	axes.set_ylim(xy_min[1], xy_max[1])
	if aspect is not None:
		axes.set_aspect(aspect)

	if colorbar:
		try:
			figure.colorbar(patches)
		except AttributeError as e:
			warn(e.args[0], RuntimeWarning)

	if delaunay:
		return obj



def field_map_2d(cells, values, angular_width=30.0, overlay=False, aspect=None, figure=None, axes=None,
		**kwargs):
	force_amplitude = values.pow(2).sum(1).apply(np.sqrt)
	if figure is None:
		figure = plt.gcf()
	if axes is None:
		axes = figure.gca()
	if not overlay:
		obj = scalar_map_2d(cells, force_amplitude, figure=figure, axes=axes, **kwargs)
	if aspect is not None:
		axes.set_aspect(aspect)
	if axes.get_aspect() == 'equal':
		aspect_ratio = 1
	else:
		xmin, xmax = axes.get_xlim()
		ymin, ymax = axes.get_ylim()
		aspect_ratio = (xmax - xmin) / (ymax - ymin)
	# compute the distance between adjacent cell centers
	if isinstance(cells, Distributed):
		A = cells.adjacency
	elif isinstance(cells, CellStats) and isinstance(cells.tessellation, Delaunay):
		A = cells.tessellation.cell_adjacency
	A = sparse.triu(A, format='coo')
	I, J = A.row, A.col
	if isinstance(cells, Distributed):
		pts_i = np.stack([ cells.cells[i].center for i in I ])
		pts_j = np.stack([ cells.cells[j].center for j in J ])
	elif isinstance(cells, CellStats):
		assert isinstance(cells.tessellation, Tessellation)
		pts_i = cells.tessellation.cell_centers[I]
		pts_j = cells.tessellation.cell_centers[J]
	dist = la.norm(pts_i - pts_j, axis=1)
	# scale force amplitude
	scale = np.nanmedian(force_amplitude)
	if np.isclose(scale, 0):
		scale = np.median(force_amplitude[0 < force_amplitude])
	scale = np.nanmedian(dist) / 2.0 / scale
	# 
	dw = float(angular_width) / 2.0
	t = tan(radians(dw))
	t = np.array([[0.0, -t], [t, 0.0]])
	markers = []
	for i in values.index:
		try:
			center = cells.tessellation.cell_centers[i]
		except AttributeError:
			center = cells.cells[i].center
		radius = force_amplitude[i]
		f = np.asarray(values.loc[i]) * scale
		#fx, fy = f
		#angle = degrees(atan2(fy, fx))
		#markers.append(Wedge(center, radius, angle - dw, angle + dw))
		base, vertex = center + np.outer([-0.5, 0.5], f)
		ortho = np.dot(t, f)
		vertices = np.stack((vertex, base + ortho, base - ortho), axis=0)
		#vertices[:,0] = center[0] + aspect_ratio * (vertices[:,0] - center[0])
		markers.append(Polygon(vertices, True))
		#if 12<center[0] and center[0]<13 and 13<center[1] and center[1]<14:
		#	print((i, center, radius, f, ortho, vertices))
		#	plt.plot(center[0], center[1], 'r+')

	patches = PatchCollection(markers, facecolor='y', edgecolor='k', alpha=0.9)
	axes.add_collection(patches)

	#axes.set_xlim(xmin, xmax)
	#axes.set_ylim(ymin, ymax)

	if not overlay and obj:
		return obj

