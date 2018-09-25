# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import numpy as np
import pandas as pd
from tramway.feature.adjacency import *
from tramway.plot.mesh import *
from tramway.plot.map import *
from tramway.helper.inference import _clip
import traceback
import matplotlib.pyplot as plt


class ContourEditor(object):
    def __init__(self, figure, renderer=None):
        self.figure = figure
        if renderer is None:
            renderer = plt
        self.renderer = renderer
        self._variables = None
        self._cells = None
        self._map = None
        self._cell_ptr = []
        self._area_ptr = []
        self._contour_ptr = []
        self._delaunay_ptr = []
        self._cell_centers = None
        self._cell_adjacency = None
        self._dilation_adjacency = None
        self._delaunay = False
        self._cell = None
        self._step = 1
        self._variable = None
        self._area = True
        self.clip = 4.
        self.debug = False
        self._areas = None
        self.callback = None

        self.cell_marker = 's'
        self.cell_marker_color = 'r'
        self.cell_marker_size = 10
        self.delaunay_color = (.1, .1, .1)

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, m):
        self._map = m
        if self._map is not None:
            self._variables = None
        self._clear()

    @property
    def variables(self):
        if self._variables is None:
            if self.map is None:
                return []
            self._variables = []
            for v in self.map.columns:
                self._variables.append(v[::-1].split(None, 1)[-1][::-1])
            self._variables = list(set(self._variables))
            self._variables = { v: [ c for c in self.map.columns if c.startswith(v) ] \
                    for v in self._variables }
        return list(self._variables.keys())

    @property
    def cell_centers(self):
        if self._cell_centers is None:
            self._cell_centers = self.cells.tessellation.cell_centers
        return self._cell_centers

    @property
    def cell_adjacency(self):
        if self._cell_adjacency is None:
            if self.cells.tessellation.cell_label is None:
                label = self.cells.location_count
            else:
                label = np.logical_and(0 < self.cells.tessellation.cell_label,
                    0 < self.cells.location_count)
            self._cell_adjacency = self.cells.tessellation.simplified_adjacency(
                label=label, format='csr')
        return self._cell_adjacency

    @property
    def cell_count(self):
        return self.cells.tessellation.cell_adjacency.shape[0]

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cs):
        self._cells = cs
        self._cell_centers = None
        self._cell_adjacency = None
        self._dilation_adjacency = None
        self.map = None
        self._areas = None

    @property
    def dilation_adjacency(self):
        if self._dilation_adjacency is None:
            try:
                self._dilation_adjacency = self.cells.tessellation.diagonal_adjacency
            except AttributeError:
                self._dilation_adjacency = False
            else:
                if self.cells.tessellation.cell_label is None:
                    label = self.cells.location_count
                else:
                    label = np.logical_and(0 < self.cells.tessellation.cell_label,
                        0 < self.cells.location_count)
                self.cells.tessellation.simplified_adjacency(
                    adjacency=self._dilation_adjacency, label=label, format='csr')
        if self._dilation_adjacency is False:
            return self.cell_adjacency
        else:
            return self._dilation_adjacency

    def _clear(self):
        self._cell_ptr = []
        self._area_ptr = []
        self._contour_ptr = []
        self._delaunay_ptr = []
        #self.axes.clear()
        self.figure.clear()

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, c):
        if c is None or (-1 <= c and (self.cells is None or c < self.cell_count)):
            self._cell = c
            self.refresh()
        else:
            raise ValueError('cell index not in range (-1,%s)', self.cell_count-1)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, s):
        if s < 1:
            raise ValueError
        else:
            self._step = s
            self.refresh()

    @property
    def variable(self):
        return self._variable

    @variable.setter
    def variable(self, v):
        if self.cells is None or self.map is None:
            return
        self._clear()
        self._variable = v
        if self.debug or self._delaunay:
            kwargs = {'delaunay': True,
                'color': [self.delaunay_color] * 10,
                'linewidth': .3}
        else:
            kwargs = {'linewidth': 0}
        if self._variables[v][1:]:
            obj = field_map_2d(self.cells, _clip(self.map[self._variables[v]], self.clip),
                figure=self.figure, **kwargs)
        else:
            obj = scalar_map_2d(self.cells, _clip(self.map[v], self.clip),
                figure=self.figure, **kwargs)
        if obj:
            self._delaunay_ptr = obj
            self._delaunay = True
        #self.delaunay = self._delaunay
        self.refresh()

    @property
    def delaunay(self):
        return self._delaunay

    @delaunay.setter
    def delaunay(self, state):
        #if self.debug:
        #       return
        toggle = self._delaunay != bool(state)
        if toggle:
            self._delaunay = not self._delaunay
        if self.cells is None:
            return
        if self._delaunay_ptr:
            if toggle:
                for p in self._delaunay_ptr:
                    p.set_visible(self._delaunay)
                self.draw()
        elif self._delaunay:
            self._delaunay_ptr = plot_delaunay(self.cells, centroid_style=None,
                axes=self.axes)
            self.draw()

    def refresh(self):
        c = self.cell
        if c is None or c < 0 or self.cells is None:
            for p in self._cell_ptr:
                self.axes.lines.remove(p)
            self._cell_ptr = []
            s = 0
        else:
            x, y = self.cell_centers[c]
            if self._cell_ptr:
                self._cell_ptr[0].set_xdata(x)
                self._cell_ptr[0].set_ydata(y)
            else:
                self._cell_ptr = self.axes.plot(x, y, self.cell_marker,
                    markerfacecolor=self.cell_marker_color,
                    markersize=self.cell_marker_size)
            s = self.step
        if 0 < s and self.cells is not None:
            try:
                cs, ok = self.cells.tessellation.contour(c, s, fallback=True,
                        adjacency=self.cell_adjacency,
                        debug=self.debug, cells=self.cells)
            except:
                if self.debug:
                    print(traceback.format_exc())
                cs = []
        else:
            cs = []
        w = None
        if (isinstance(cs, np.ndarray) and cs.size) or cs:
            ws = set()
            for _ws in dilation(c, self.dilation_adjacency, (1, s), cs).values():
                ws |= _ws
            if ws:
                if self.area:
                    w = self.cell_centers[list(ws)]
                    if self._area_ptr:
                        self._area_ptr[0].set_xdata(w[:,0])
                        self._area_ptr[0].set_ydata(w[:,1])
                    else:
                        self._area_ptr = self.axes.plot(w[:,0], w[:,1], 'g.',
                            markersize=6)
            color = 'k' if ok else 'r'
            markeredgecolor = 'y' if ok else 'k'
            z = self.cell_centers[cs]
            z = np.vstack((z, z[0]))
            if self._contour_ptr:
                self._contour_ptr[0].set_xdata(z[:,0])
                self._contour_ptr[0].set_ydata(z[:,1])
                self._contour_ptr[0].set_color(color)
                self._contour_ptr[0].set_markerfacecolor(color)
                self._contour_ptr[0].set_markeredgecolor(markeredgecolor)
            else:
                self._contour_ptr = self.axes.plot(z[:,0], z[:,1], color+'s-',
                    markeredgecolor=markeredgecolor, markerfacecolor=color,
                    markersize=6, linewidth=3)
            if self.callback:
                ws.add(self.cell)
                try:
                    self.callback(self.integral(cs, ws))
                except:
                    print(traceback.format_exc())
        elif self._contour_ptr:
            for p in self._contour_ptr:
                self.axes.lines.remove(p)
            self._contour_ptr = []
        if w is None and self.area:
            for p in self._area_ptr:
                self.axes.lines.remove(p)
            self._area_ptr = []
        self.draw()

    @property
    def axes(self):
        return self.figure.gca()

    @property
    def area(self):
        return self._area

    def draw(self):
        self.renderer.draw()

    @area.setter
    def area(self, a):
        self._area = a
        self.refresh()

    def find_cell(self, pt):
        pt = np.asarray(pt)[np.newaxis,:]
        if isinstance(self.cells.points, pd.DataFrame):
            coord_names = self.cells.points.columns[1:-1]
            pt = pd.DataFrame(data=pt, columns=coord_names)
        elif isstructured(self.cells.points):
            raise NotImplementedError
        ix = self.cells.tessellation.cell_index(pt)
        return ix[0]

    def field(self, cs):
        return self.map.loc[cs, self._variables[self._variable]].values

    def integral(self, contour, inner):
        field = self.field(contour)
        tangent = self.tangent(contour)
        area = self.surface_area(contour, inner)
        if 1 < field.shape[1]:
            return self.curl(field, tangent, area)
        else:
            return self.sum(field, tangent, area)

    def tangent(self, cs):
        X = self.cells.tessellation.cell_centers
        uvw = zip([cs[-1]]+cs[:-1], cs, cs[1:]+[cs[0]])
        return np.vstack([ (X[w]-X[u])*.5 for u, v, w in uvw ])

    def surface_area(self, contour, inner):
        if self._areas is None:
            ncells = self.cells.tessellation.cell_adjacency.shape[0]
            centers = self.cells.tessellation.cell_centers
            vertices = self.cells.tessellation.cell_vertices
            adjacency = self.cells.tessellation.vertex_adjacency.tocsr()
            vert_coords = self.cells.tessellation.vertices
            self._areas = np.zeros(ncells)
            for c in range(ncells):
                u = centers[c]
                verts = set(vertices[c].tolist())
                ordered_verts = []
                next_verts = set(verts) # copy
                try:
                    while verts:
                        vert = next_verts.pop()
                        ordered_verts.append(vert)
                        verts.remove(vert)
                        next_verts = set(adjacency.indices[adjacency.indptr[vert]:adjacency.indptr[vert+1]]) & verts
                except KeyError:
                    if self.debug:
                        print(traceback.format_exc())
                    continue
                for a, b in zip(ordered_verts, ordered_verts[1:]+[ordered_verts[0]]):
                    v, w = vert_coords[a], vert_coords[b]
                    self._areas[c] += abs((v[0]-u[0])*(w[1]-u[1])-(w[0]-u[0])*(v[1]-u[1]))
            self._areas *= .5
        return np.sum(self._areas[list(inner)])

    def sum(self, x, *args):
        return np.mean(x)

    def curl(self, F, dr, A):
        return np.sum(F * dr) / A

