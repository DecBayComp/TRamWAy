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
		delaunay=False, colorbar=True, alpha=None, colormap=None, **kwargs):
	#	colormap (str): colormap name; see also https://matplotlib.org/users/colormaps.html
	if isinstance(values, pd.DataFrame):
		if values.shape[1] != 1:
			warn('multiple parameters available; mapping first one only', UserWarning)
		values = values.iloc[:,0] # to Series
	#values = pd.to_numeric(values, errors='coerce')

	if delaunay:
		delaunay_linewidth = linewidth
		linewidth = 0
		if isinstance(delaunay, dict):
			delaunay_linewidth = delaunay.pop('linewidth', delaunay_linewidth)
			kwargs.update(delaunay)

	polygons = []
	if isinstance(cells, Distributed):
		ix, xy, ok = zip(*[ (i, c.center, bool(c)) for i, c in cells.items() ])
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
		if cells.tessellation.cell_label is not None:
			ok = np.logical_and(ok, 0 < cells.tessellation.cell_label)
		map_defined = np.zeros_like(ok)
		map_defined[values.index] = True
		ok[np.logical_not(map_defined)] = False
		ok[ok] = np.logical_not(np.isnan(values.loc[ix[ok]].values))
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

	try:
		bounding_box = cells.bounding_box[['x', 'y']]
		xy_min, xy_max = bounding_box.values
	except (KeyboardInterrupt, SystemExit):
		raise
	except:
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
	if alpha is None:
		alpha = .9
	patches = PatchCollection(polygons, alpha=alpha, linewidth=linewidth, cmap=colormap)
	patches.set_array(scalar_map)
	if clim is not None:
		patches.set_clim(clim)
	axes.add_collection(patches)

	obj = None
	if delaunay:
		try:
			import tramway.plot.mesh as mesh
			obj = mesh.plot_delaunay(cells, centroid_style=None,
				linewidth=delaunay_linewidth,
				axes=axes, **kwargs)
		except:
			import traceback
			print(traceback.format_exc())

	axes.set_xlim(xy_min[0], xy_max[0])
	axes.set_ylim(xy_min[1], xy_max[1])
	if aspect is not None:
		axes.set_aspect(aspect)

	if colorbar:
		try:
			figure.colorbar(patches)
		except AttributeError as e:
			warn(e.args[0], RuntimeWarning)

	return obj



def field_map_2d(cells, values, angular_width=30.0, overlay=False, aspect=None, figure=None, axes=None,
		cell_arrow_ratio=0.4, markeralpha=0.8, **kwargs):
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
	inter_cell_distance = la.norm(pts_i - pts_j, axis=1)
	# scale force amplitude
	#scale = np.nanmedian(force_amplitude)
	#if np.isclose(scale, 0):
	#	scale = np.median(force_amplitude[0 < force_amplitude])
	#scale = np.nanmedian(inter_cell_distance) / 2.0 / scale
	large_arrow_length = np.max(force_amplitude) # consider clipping
	scale = np.nanmedian(inter_cell_distance) / (large_arrow_length * cell_arrow_ratio)
	# 
	dw = float(angular_width) / 2.0
	t = tan(radians(dw))
	t = np.array([[0.0, -t], [t, 0.0]])
	markers = []
	for i in values.index:
		try:
			center = cells.tessellation.cell_centers[i]
		except AttributeError:
			center = cells[i].center
		radius = force_amplitude[i]
		f = np.asarray(values.loc[i]) * scale
		#fx, fy = f
		#angle = degrees(atan2(fy, fx))
		#markers.append(Wedge(center, radius, angle - dw, angle + dw))
		base, vertex = center + np.outer([-1./3, 2./3], f)
		ortho = np.dot(t, f)
		vertices = np.stack((vertex, base + ortho, base - ortho), axis=0)
		#vertices[:,0] = center[0] + aspect_ratio * (vertices[:,0] - center[0])
		markers.append(Polygon(vertices, True))

	patches = PatchCollection(markers, facecolor='y', edgecolor='k', alpha=markeralpha)
	axes.add_collection(patches)

	#axes.set_xlim(xmin, xmax)
	#axes.set_ylim(ymin, ymax)

	if not overlay and obj:
		return obj

