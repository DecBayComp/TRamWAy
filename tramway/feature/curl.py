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
from tramway.core.namedcolumns import splitcoord
from .adjacency import *
from tramway.inference import Maps
import traceback


class Curl(object):
    def __init__(self, cells=None, map=None):
        self._variables = None
        self._cells = None
        self._map = None
        self._map_index = None
        self._cell_centers = None
        self._cell_adjacency = None
        self._dilation_adjacency = None
        self._area = True
        self.debug = False
        self._areas = None
        if cells is not None:
            self.cells = cells
        if map is not None:
            self.map = map

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, m):
        if isinstance(m, Maps):
            m = m.maps
        self._map = m
        if self._map is not None:
            self._variables = None
            self._map_index = None

    @property
    def variables(self):
        if self._variables is None:
            if self.map is None:
                return []
            self._variables = splitcoord(self.map.columns)
        return self._variables

    @variables.setter
    def variables(self, vs):
        raise AttributeError("'variables' is read-only")

    @property
    def cell_centers(self):
        if self._cell_centers is None:
            self._cell_centers = self.cells.tessellation.cell_centers
        return self._cell_centers

    @cell_centers.setter
    def cell_centers(self, vs):
        raise AttributeError("'cell_centers' is read-only")

    @property
    def cell_adjacency(self):
        if self._cell_adjacency is None:
            self._cell_adjacency = self.cells.tessellation.simplified_adjacency().tocsr()
        return self._cell_adjacency

    @cell_adjacency.setter
    def cell_adjacency(self, vs):
        raise AttributeError("'cell_adjacency' is read-only")

    @property
    def cell_count(self):
        return self.cells.tessellation.cell_adjacency.shape[0]

    @cell_count.setter
    def cell_count(self, cc):
        raise AttributeError("'cell_count' is read-only")

    @property
    def dilation_adjacency(self):
        if self._dilation_adjacency is None:
            try:
                self._dilation_adjacency = self.cells.tessellation.diagonal_adjacency
            except AttributeError:
                self._dilation_adjacency = False
        if self._dilation_adjacency is False:
            return self.cell_adjacency
        else:
            return self._dilation_adjacency

    @dilation_adjacency.setter
    def dilation_adjacency(self, vs):
        raise AttributeError("'dilation_adjacency' is read-only")

    def local_curl(self, variable, cell, distance):
        v, c, s = variable, cell, distance
        try:
            cs = self.cells.tessellation.contour(c, s,
                adjacency=self.cell_adjacency,
                debug=self.debug, cells=self.cells)
        except:
            return None
        ws = set((cell,))
        for _ws in dilation(c, self.dilation_adjacency, (1, s), cs).values():
            ws |= _ws
        return self.curl_integral(v, cs, ws)

    def curl_integral(self, variable, contour, inner):
        area = self.surface_area(contour, inner)
        if np.isclose(area, 0.):
            return 0.
        ok = np.array([ c in self.map_index for c in contour ])
        field = self.field(variable, np.asarray(contour)[ok])
        tangent = np.asarray(self.tangent(contour))[ok]
        return np.sum(field * tangent) / area

    @property
    def map_index(self):
        if self._map_index is None:
            self._map_index = set(self.map.index)
        return self._map_index

    def field(self, v, cs):
        return self.map.loc[cs, self.variables[v]].values

    def tangent(self, cs):
        X = self.cells.tessellation.cell_centers
        uvw = zip([cs[-1]]+cs[:-1], cs, cs[1:]+[cs[0]])
        return np.vstack([ (X[w]-X[u])*.5 for u, v, w in uvw ])

    def surface_area(self, contour, inner):
        if False:#self._areas is None:
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
        return np.sum(self.cells.tessellation.cell_volume[list(inner)])

    def extract(self, label, variable=None, distance=1):
        if variable is None:
            vector_field = { f: vs for f, vs in self.variables if len(vs) == 2 }
            if vector_field[1:]:
                raise ValueError('multiple vector fields; `variable` argument is required')
            else:
                variable = list(vector_field.keys())[0]
        cells, curls = [], []
        for cell in self.map.index:
            curl = self.local_curl(variable, cell, distance)
            if curl is not None:
                cells.append(cell)
                curls.append(curl)
        curls = pd.Series(index=np.array(cells), data=np.array(curls))
        #self.map[label] = curls
        return pd.DataFrame({label: curls})

