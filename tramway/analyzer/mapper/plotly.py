# -*- coding: utf-8 -*-

# Copyright © 2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..attribute import AnalyzerNode, single
from ..artefact import Analysis
import numpy as np
import pandas as pd
import plotly  # missing dependencies: plotly nbformat
import plotly.graph_objects as go
import tramway.plot.map as tplt

"""
Plotly interfaces currently are highly experimental
and may suffer from numerous bugs.

The `plotly` dependencies are not enforced and an
:class:`ImportError` is raised if the library cannot
be found.
"""


class Plotly(AnalyzerNode):
    """
    Plotly interface for maps.

    It does not access other attributes of the :class:`RWAnalyzer`,
    and thus can be safely used from any :class:`RWAnalyzer` object:

        from tramway.analyzer import RWAnalyzer

        RWAnalyzer().mapper.plotly.plot_surface(...)

    or:

        from tramway.analyzer.mapper.plotly import Plotly

        Plotly.plot_surface(...)

    """

    def plot_surface(
        self,
        maps,
        feature,
        sampling=None,
        fig=None,
        row=None,
        col=None,
        colormap="viridis",
        title=None,
        colorbar=None,
        resolution=2000,
        interpolation=None,
        **kwargs,
    ):
        """
        Plot a 2D map as a colored 3D surface.

        Argument `interpolation` currently admits any of the following values:

        * :const:`'flat grid'`: the surface is evaluated on a regular grid with
          no interpolation
        * :const:`'flat'`: the surface is made of constant-z polygons (no
          interpolation, default)

        Argument `resolution` only applies to `'grid'` interpolations.

        """
        surface_kwargs = kwargs
        surface_kwargs["colorscale"] = kwargs.pop("colorscale", colormap)

        figure_kwargs = {}
        for kw in ("scene",):
            try:
                arg = kwargs.pop(kw)
            except KeyError:
                pass
            else:
                figure_kwargs[kw] = arg

        if fig is not None:
            if not (row is None and col is None):
                # subplots
                figure_kwargs.update(
                    dict(
                        row=1 if row is None else row + 1,
                        col=1 if col is None else col + 1,
                    )
                )

        if colorbar is not None:
            surface_kwargs["colorbar"] = colorbar

        if isinstance(maps, Analysis):
            if sampling is None:
                sampling = maps.get_parent()
            maps = maps.data
        if isinstance(sampling, Analysis):
            sampling = sampling.data

        tessellation = sampling.tessellation
        try:
            tessellation = tessellation.spatial_mesh
        except AttributeError:
            pass

        map_ = maps[feature]
        if 1 < map_.shape[1]:
            map_ = map_.pow(2).sum(1).apply(np.sqrt)
        else:
            map_ = map_[feature]

        xlim, ylim = sampling.bounding_box[["x", "y"]].values.T

        if interpolation == "flat grid":
            step = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / resolution
            x = np.arange(xlim[0], xlim[1], step)
            y = np.arange(ylim[0], ylim[1], step)
            x, y = np.meshgrid(x, y)
            pts = np.stack((x.ravel(), y.ravel()), axis=1)
            partition_kwargs = dict(sampling.param.get("partition", {}))
            for kw in list(partition_kwargs.keys()):
                if kw.startswith("time_"):
                    del partition_kwargs[kw]
                elif kw in ("min_location_count", "knn"):
                    del partition_kwargs[kw]
            cell_ix = tessellation.cell_index(
                pd.DataFrame(pts, columns=["x", "y"]), **partition_kwargs
            )

            z = np.zeros(len(pts), dtype=map_.dtype)

            if isinstance(cell_ix, tuple):
                pts_, cells = cell_ix
                assert np.all(np.unique(pts_) == np.arange(len(pts)))
                # brute force accum array
                n = np.zeros(z.shape, dtype=int)
                for i in map_.index:
                    k = pts_[cells == i]
                    if k.size:
                        z[k] += map_[i]
                        n[k] += 1
                ok = 0 < n
                z[1 < n] /= n[1 < n]
            else:
                ok = 0 <= cell_ix
                map__ = np.full(tessellation.number_of_cells, np.nan, dtype=map_.dtype)
                map__[map_.index] = map_.values
                z[ok] = map__[cell_ix[ok]]
            z[~ok] = np.nan

            z = z.reshape(x.shape)
            surface = go.Surface(x=x, y=y, z=z, **surface_kwargs)

        elif interpolation in (None, "flat"):
            if type(tessellation).__name__ in (
                "RegularMesh",
                "HexagonalMesh",
                "KDTreeMesh",
            ):
                vertices, cell_vertices, Av = (
                    tessellation.vertices,
                    tessellation.cell_vertices,
                    tessellation.vertex_adjacency.tocsr(),
                )
            else:
                try:
                    vertices, cell_vertices, Av = tplt.box_voronoi_2d(
                        tessellation, xlim, ylim
                    )
                except AssertionError:
                    raise RuntimeError(
                        "could not fix the borders; try again with interpolation='flat grid'"
                    ) from None
            polygons = []
            for c in map_.index:
                vs = tplt.cell_to_polygon_(c, vertices, cell_vertices, Av, xlim, ylim)
                for i, v in vs:
                    polygons.append((v, map_[i]))
            triangles = levelled_2d_polygons_to_triangulated_piecewise_surface(
                polygons, 0
            )
            surface = go.Mesh3d(
                x=triangles[:, 0],
                y=triangles[:, 1],
                z=triangles[:, 2],
                i=np.arange(0, len(triangles), 3),
                j=np.arange(1, len(triangles), 3),
                k=np.arange(2, len(triangles), 3),
                intensitymode="vertex",
                intensity=triangles[:, 2],
                cauto=True,
                **surface_kwargs,
            )

        if not isinstance(surface, list):
            surface = [surface]
        if fig is None:
            fig = go.Figure(data=surface, **figure_kwargs)
        else:
            for trace in surface:
                fig.add_trace(trace, **figure_kwargs)


class EdgeSet:
    """
    Compilation of edges, for independently represented faces.

    Edges are not directed; the order of the vertices does not matter.
    Edge indices and face indices are stored and returned as sets.

    Attributes:

        vertices (list): unique vertices with their respective edge indices.

        edges (list): unique edges (as pairs of vertex indices) with their
            respective face indices.

    Methods:

        add(u, v[, face]): register edge [`u`, `v`], with vertices `u` and `v`,
            as being part of face index `face`.
            The order of `u` and `v` does not matter.

        lookup(v[, face]): find neighbor faces (as indices) for face index
            `face` at vertex `v`;
            neighbor faces are defined as having a common edge; faces that
            share `v` but no edges are excluded.
            If `face` is not defined, return all faces (as indices) that have
            vertex `v`.

        lookup(u, v[, face]): find neighbor faces (as indices) for face index
            `face` at edge [`u`, `v`].
            If `face` is not defined, return all faces (as indices) that have
            vertices `u` and `v`.
            This method is useful mostly to check for neighbors.

        compile(faces): incrementally add each edge of each face.
            `faces` is a list of vertices (NxD matrix, with D dimensions);
            the vertices must be ordered to form a proper path delimiting the
            corresponding face.

    """

    @classmethod
    def _same(cls, u, v):
        # return np.all( u == v )
        return all([i == j for i, j in zip(u, v)])

    def __init__(self, faces=None):
        self.vertices = []
        self.edges = []
        self.same = self._same
        if faces is not None:
            self.compile(faces)

    def compile(self, faces):
        for face_ix, ordered_vertices in enumerate(faces):
            if len(ordered_vertices.shape) != 2:
                raise ValueError("vertices should be NxD matrices")
            if ordered_vertices.shape[1] < 2:
                raise ValueError("vertices should be at least 2D")
            n = len(ordered_vertices)
            if n < 3:
                raise ValueError("faces should have at least 3 vertices")
            for i in range(n):
                j = i + 1
                if j == n:
                    j = 0
                p, q = ordered_vertices[i], ordered_vertices[j]
                self.add(p, q, face_ix)

    def add(self, u, v, face=None):
        i, u_edges = self._add_vertex(u)
        j, v_edges = self._add_vertex(v)
        edges = u_edges & v_edges
        if edges:
            assert len(edges) == 1
            edge = next(iter(edges))
            if face is not None:
                _, faces = self.edges[edge]
                faces.add(face)
        else:
            faces = set()
            if face is not None:
                faces.add(face)
            edge = self._add_edge(i, j, faces)
        return edge, faces

    def _add_edge(self, i, j, faces):
        edge_ix = len(self.edges)
        self.edges.append(((i, j), faces))
        for k in (i, j):
            _, edges = self.vertices[k]
            edges.add(edge_ix)
        return edge_ix

    def _add_vertex(self, v):
        found = False
        for i, rec in enumerate(self.vertices):
            u, edges = rec
            if self.same(u, v):
                found = True
                break
        if not found:
            i = len(self.vertices)
            edges = set()
            self.vertices.append((v, edges))
        return i, edges

    def __iter__(self):
        return self.edges.__iter__()

    def copy(self):
        dup = EdgeSet()
        dup.vertices = [(v, set(edges)) for v, edges in self.vertices]
        dup.edges = list(self.edges)
        return dup

    def lookup(self, u, v_or_face=None, face=None):
        # implement some sort of dispatch
        if v_or_face is None:
            return self.lookup_vertex(u, face)
        elif isinstance(v_or_face, np.ndarray) and 1 < v_or_face.size:
            v = v_or_face
            return self.lookup_edge(u, v, face)
        elif face is None:
            return self.lookup_vertex(u, v_or_face)
        else:
            raise ValueError(
                f"cannot dispatch to lookup_vertex or lookup_edge with args: {u} {v_or_face}, {face}"
            )

    def lookup_vertex(self, v, face=None):
        faces = set()
        # Faces that have vertex `v`, or
        # Neighbor faces of face `face` at vertex `v`, with a common edge
        for u, edges in self.vertices:
            if self.same(u, v):
                for edge in edges:
                    _, fs = self.edges[edge]
                    if face is None or face in fs:
                        faces |= fs
                break
        faces.discard(face)
        return faces

    def lookup_edge(self, u, v, face=None):
        faces = self.lookup_vertex(u, face)
        if faces:
            faces &= self.lookup_vertex(v, face)
        return faces


def levelled_2d_polygons_to_triangulated_piecewise_surface(polygons, dr=0.1):
    flat_mesh = EdgeSet([xy for xy, z in polygons])
    # optionally insert a margin between neighbor polygons
    # and insert a polygon between each pair of matching edges
    extra_polygons = dict()
    updated_polygons = []
    if dr:
        dr_dist = []
        polygons_with_pending_updates = []
        for xy, z in polygons:
            center = np.mean(xy, axis=0, keepdims=True)
            polygons_with_pending_updates.append((center, [], xy, z))
            dxy = xy - center
            dr_dist.append(np.sum(dxy * dxy, axis=1))
        dr_dist = np.concatenate(dr_dist)
        margin = dr * max(dr_dist.min(), 1e-2 * dr_dist.max())
        for polygon_ix, p in enumerate(polygons_with_pending_updates):
            xy0, updates, xy, _ = p
            center = xy0[0]  # to vector
            n = len(xy)
            for i in range(n):
                j = i + 1
                if j == n:
                    j = 0
                p, q = xy[i], xy[j]
                neighbor_polygon = flat_mesh.lookup(p, q, polygon_ix)
                if neighbor_polygon:
                    pq = q - p
                    norm = np.sqrt(np.sum(pq * pq))
                    normal = np.r_[pq[1], -pq[0]] / norm  # 2d only
                    # align the normal towards the center
                    if np.dot(normal, center - p) < 0:
                        normal = -normal
                    # update p and q
                    update = np.zeros_like(xy)
                    update[i, :] = margin * normal
                    update[j, :] = margin * normal
                    updates.append(update)
                    # insert extra polygon
                    assert len(neighbor_polygon) == 1
                    neighbor_polygon = next(iter(neighbor_polygon))
                    try:
                        edge = (neighbor_polygon, polygon_ix)
                        extra_polygon = extra_polygons[edge]
                    except KeyError:
                        edge = (polygon_ix, neighbor_polygon)
                        extra_polygons[edge] = extra_polygon = []
                    extra_polygon.append([polygon_ix, i, j])
        # commit the updates
        for polygon_ix, p in enumerate(polygons_with_pending_updates):
            _, updates, xy, z = p
            updates = iter(updates)
            update = next(updates)
            while True:
                try:
                    # TODO: summing is not the right operation...
                    update += next(updates)
                except StopIteration:
                    break
            xy = xy + update
            updated_polygons.append(np.c_[xy, np.full((xy.shape[0], 1), z)])
    else:
        for polygon_ix, p in enumerate(polygons):
            xy, z = p
            updated_polygons.append(np.c_[xy, np.full((xy.shape[0], 1), z)])
            n = len(xy)
            for i in range(n):
                j = i + 1
                if j == n:
                    j = 0
                p, q = xy[i], xy[j]
                neighbor_polygon = flat_mesh.lookup(p, q, polygon_ix)
                if neighbor_polygon:
                    # insert extra polygon
                    assert len(neighbor_polygon) == 1
                    neighbor_polygon = next(iter(neighbor_polygon))
                    try:
                        edge = (neighbor_polygon, polygon_ix)
                        extra_polygon = extra_polygons[edge]
                    except KeyError:
                        edge = (polygon_ix, neighbor_polygon)
                        extra_polygons[edge] = extra_polygon = []
                    extra_polygon.append([polygon_ix, i, j])
    # append the extra polygons
    for p, q in extra_polygons.values():
        p, pi, pj = p
        q, qi, qj = q
        p0, q0 = polygons[p][0], polygons[q][0]
        p, q = updated_polygons[p], updated_polygons[q]
        pu0, qu0 = p0[pi], q0[qi]
        puv, quv = p[[pi, pj]], q[[qi, qj]]
        if flat_mesh.same(pu0, qu0):
            qvu = quv[::-1]
            polygon = np.r_[puv, qvu]
        else:
            polygon = np.r_[puv, quv]
        updated_polygons.append(polygon)
    # triangulate
    triangles = []
    for vs in updated_polygons:
        r = np.mean(vs, axis=0, keepdims=True)
        n = len(vs)
        for i in range(n):
            j = i + 1
            if j == n:
                j = 0
            p, q = vs[[i]], vs[[j]]
            triangle = np.r_[p, q, r]
            triangles.append(triangle)
    # format
    # levelled_mesh = EdgeSet(triangles)
    return np.concatenate(triangles, axis=0)


__all__ = ["Plotly"]
