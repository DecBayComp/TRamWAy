# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
import polytope as pt
import copy
from tramway.core.hdf5.store import save_rwa
from tramway.core.xyt import crop
from tramway.helper import *

import warnings
__log__ = lambda msg: warnings.warn(msg, RuntimeWarning) # could call the logging module instead

__available_packages__ = set()
__reported_missing_packages__ = set()
def __has_package__(pkg, report_if_missing=True):
    if pkg in __available_packages__:
        return True
    elif report_if_missing:
        if pkg not in __reported_missing_packages__:
            __reported_missing_packages__.add(pkg)
            __log__("package '{}' is missing".format(pkg))
    return False

try:
    from tqdm import tqdm
except ImportError:
    pass
else:
    __available_packages__.add('tqdm')


class AutosaveCapable(object):
    def __init__(self, rwa_file=None, autosave=True):
        self.autosave = autosave
        self.rwa_file = rwa_file
        self.save_options = dict(force=True)
        self._analysis_tree = None
        self._modified = None
    def save(self):
        if self.rwa_file:
            save_rwa(self.rwa_file, self._analysis_tree, **self.save_options)
            return True
    def autosaving(self, analysis_tree):
        if self.autosave:
            self._analysis_tree = analysis_tree
        return self
    def __enter__(self):
        self._modified = False
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is None and self._modified:
            if self.autosave:
                self.save()
                # unload
                self._analysis_tree = None
        # reset
        self._modified = None
    @property
    def modified(self):
        if self._modified is None:
            raise RuntimeError("property 'modified' called from outside the context")
        return self._modified
    @modified.setter
    def modified(self, b):
        if self._modified is None:
            raise RuntimeError("property 'modified' called from outside the context")
        if b is not True:
            raise ValueError("property 'modified' can only be set to True")
        self._modified = b


class RoiToolbox(AutosaveCapable):
    def __init__(self, roi, rwa_file=None, autosave=True, verbose=True):
        AutosaveCapable.__init__(self, rwa_file, autosave)
        # group overlapping rois
        current_index = 0
        not_an_index = -1
        assignment = np.full(len(roi), not_an_index, dtype=int)
        groups = dict()
        for i in range(len(roi)):
            if isinstance(roi[i], pt.Region):
                region_i = roi[i]
                xi0 = yi0 = xi1 = yi1 = None
            else:
                region_i = None
                xi0, yi0, xi1, yi1 = roi[i]
            if assignment[i] == not_an_index:
                group_index = current_index
                group = set([i])
            else:
                group_index = assignment[i]
                group = groups[group_index]
            group_with = set()
            for j in range(i+1,len(roi)):
                if j in group:
                    continue
                if isinstance(roi[j], pt.Region):
                    region_j = roi[j]
                    if region_i is None:
                        region_i = pt.box2poly([[xi0,xi1],[yi0,yi1]])
                    i_and_j_intersect = pt.is_adjacent(region_i, region_j)
                else:
                    xj0, yj0, xj1, yj1 = roi[j]
                    if xi0 is None:
                        region_j = pt.box2poly([[xj0,xj1],[yj0,yj1]])
                        i_and_j_intersect = pt.is_adjacent(region_i, region_j)
                    else:
                        i_and_j_intersect = xi0<xj1 and xj0<xi1 and yi0<yj1 and yj0<yi1
                if i_and_j_intersect:
                    if assignment[j]==not_an_index:
                        group.add(j)
                    else:
                        other_group_index = assignment[j]
                        group_with.add(other_group_index)
                        group += groups.pop(other_group_index)
            if group_with:
                group_index = min(group_with)
            else:
                current_index += 1
            groups[group_index] = group
            assignment[list(group)] = group_index
        self.bounding_box = roi
        self.roi_group = groups
        #
        self.verbose = verbose
    @property
    def label_prefix(self):
        return 'roi'
    @property
    def label_sep(self):
        return '-'
    @property
    def label_format(self):
        return '{:0>3d}'
    @property
    def label_suffix(self):
        return ''
    def get_group_index(self, i):
        broke = False
        for j in self.roi_group:
            if i in self.roi_group[j]:
                broke = True
                break
        if not broke:
            raise RuntimeError('cannot find roi index {}'.format(i))
        return j
    def get_group(self, i):
        return self.roi_group[self.get_group_index(i)]
    def gen_group_label(self, i):
        label = ''.join((
                self.label_prefix,
                self.label_sep.join([ self.label_format.format(j) for j in i ]),
                self.label_suffix))
        return label
    def gen_label(self, i):
        return self.gen_group_label(self.get_group(i))
    @property
    def roi_labels(self):
        return [ self.gen_group_label(self.roi_group[i]) for i in self.roi_group ]
    def get_subtree(self, i, analysis_tree):
        label = self.gen_label(i)
        if label in analysis_tree:
            return analysis_tree[label]
    def crop(self, i, df):
        loc_indices = set()
        df_i = None
        for r in self.get_group(i):
            if isinstance(self.bounding_box[r], pt.Region):
                raise NotImplementedError
            else:
                x0,y0,x1,y1 = self.bounding_box[r]
                df_r = crop(df, [x0,y0,x1-x0,y1-y0])
            if df_i is None:
                df_i = df_r
            else:
                df_i = pd.merge(df_i, df_r, how='outer')
        return df_i
    def tessellate(self, i, analysis_tree, *args, **kwargs):
        label = self.gen_label(i)
        if label in analysis_tree:
            return False
        else:
            trajectories = self.crop(i, analysis_tree.data)
            partition = tessellate(trajectories, *args, **kwargs)
            analysis_tree[label] = partition
            return True
    def infer(self, i, analysis_tree, *args, **kwargs):
        label = self.gen_label(i)
        if label in analysis_tree:
            try:
                if kwargs['output_label'] in analysis_tree[label]:
                    return False
            except KeyError:
                pass
            kwargs['input_label'] = label
            infer(analysis_tree, *args, **kwargs)
            return True
        return False
    def __range__(self, n, desc=None):
        if self.verbose:
            if __has_package__('tqdm'):
                return tqdm(range(n), desc=desc)
        return range(n)
    def iter_roi(self, desc=None):
        return self.__range__(len(self.bounding_box), desc)
    @property
    def tessellate_desc(self):
        return 'Tessellating the regions of interest'
    def tessellate_all(self, analysis_tree, *args, **kwargs):
        with self.autosaving(analysis_tree) as a:
            for i in self.iter_roi(self.tessellate_desc):
                if self.tessellate(i, analysis_tree, *args, **kwargs):
                    a.modified = True
            return a.modified
    @property
    def infer_desc(self):
        return 'Inferring dynamics parameters'
    def infer_all(self, analysis_tree, *args, **kwargs):
        with self.autosaving(analysis_tree) as a:
            for i in self.iter_roi(self.infer_desc):
                if self.infer(i, analysis_tree, *args, **kwargs):
                    a.modified = True
            return a.modified
    def get_map(self, i, analysis_tree, map_label, full=False):
        label = self.gen_label(i)
        try:
            subtree = analysis_tree[label]
            maps = subtree[map_label].data
        except KeyError:
            return None
        if not full and 1 < len(self.get_group(i)):
            cx, cy = subtree.data.tessellation.cell_centers.T
            rx0, ry0, rx1, ry1 = self.bounding_box[i]
            inside, = np.nonzero((cx<rx1) & (cy<ry1) & (rx0<cx) & (ry0<cy))
            maps = maps.sub(inside)
        return maps
    def get_cells(self, i, analysis_tree):
        label = self.gen_label(i)
        cells = analysis_tree[label].data
        tessellation = cells.tessellation
        if len(self.get_group(i)) == 1:
            inner_cell = np.ones(tessellation.number_of_cells, dtype=bool)
        else:
            cx, cy = tessellation.cell_centers.T
            rx0, ry0, rx1, ry1 = self.bounding_box[i]
            inner_cell = (cx<rx1) & (cy<ry1) & (rx0<cx) & (ry0<cy)
        return cells, inner_cell
    def get_tessellation(self, i, analysis_tree):
        cells, inner_cell = self.get_cells(i, analysis_tree)
        return cells.tessellation, inner_cell
    def cell_plot(self, i, analysis_tree, **kwargs):
        label = self.gen_label(i)
        title = kwargs.pop('title', self.gen_group_label([i]))
        if 'delaunay' not in kwargs:
            # anticipate future changes in `helper.tessellation.cell_plot`
            voronoi = kwargs.pop('voronoi', dict(centroid_style=None))
            kwargs['voronoi'] = voronoi
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'equal'
        plot_bb = 1<len(self.get_group(i))
        if plot_bb:
            kwargs['show'] = False
        #
        cell_plot(analysis_tree, label=label, title=title, **kwargs)
        #
        import matplotlib.pyplot as plt
        ax = plt.gca()
        x0, y0, x1, y1 = self.bounding_box[i]
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xc, yc = .5 * (x0 + x1), .5 * (y0 + y1)
        if not plot_bb:
            x0, y0, x1, y1 = xl[0], yl[0], xl[1], yl[1]
        ax.plot(
                [x0, x1, np.nan, xc, xc],
                [yc, yc, np.nan, y0, y1],
                color='k', linestyle='--', alpha=1, linewidth=1)
        if plot_bb:
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Rectangle
            masks = []
            if xl[0]<x0:
                masks.append(Rectangle((xl[0], yl[0]), x0-xl[0], yl[1]-yl[0]))
            if x1<xl[1]:
                masks.append(Rectangle((x1, yl[0]), xl[1]-x1, yl[1]-yl[0]))
            if yl[0]<y0:
                masks.append(Rectangle((x0, yl[0]), x1-x0, y0-yl[0]))
            if y1<yl[1]:
                masks.append(Rectangle((x0, y1), x1-x0, yl[1]-y1))
            pc = PatchCollection(masks, facecolor='k', alpha=.1, edgecolor=None)
            ax.add_collection(pc)
            #
            ax.plot(
                    [x0,x0,x1,x1,x0],
                    [y0,y1,y1,y0,y0],
                    color='k', linestyle='-', alpha=1, linewidth=1)
            ax.set_xlim(xl)
            ax.set_ylim(yl)
    def map_plot(self, i, analysis_tree, map_label, **kwargs):
        xmin, ymin, xmax, ymax = self.bounding_box[i]
        label = self.gen_label(i)
        title = kwargs.pop('title', self.gen_group_label([i]))
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'equal'
        kwargs['show'] = False
        plot_bb = 1<len(self.get_group(i))
        #
        map_plot(analysis_tree, label=(label,map_label), title=title, **kwargs)
        #
        x0, y0, x1, y1 = self.bounding_box[i]
        xc, yc = .5 * (x0 + x1), .5 * (y0 + y1)
        import matplotlib.pyplot as plt
        ax = plt.gca()
        ax.plot(xc, yc, 'r+')
        if plot_bb:
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Rectangle
            xl, yl = ax.get_xlim(), ax.get_ylim()
            #
            masks = []
            if xl[0]<x0:
                masks.append(Rectangle((xl[0], yl[0]), x0-xl[0], yl[1]-yl[0]))
            if x1<xl[1]:
                masks.append(Rectangle((x1, yl[0]), xl[1]-x1, yl[1]-yl[0]))
            if yl[0]<y0:
                masks.append(Rectangle((x0, yl[0]), x1-x0, y0-yl[0]))
            if y1<yl[1]:
                masks.append(Rectangle((x0, y1), x1-x0, yl[1]-y1))
            pc = PatchCollection(masks, facecolor='k', alpha=.1, edgecolor=None)
            ax.add_collection(pc)
            #
            ax.plot(
                    [x0,x0,x1,x1,x0],
                    [y0,y1,y1,y0,y0],
                    'k-', linewidth=1)
            ax.set_xlim(xl)
            ax.set_ylim(yl)


#from .base import Helper, Analyses
#from tramway.helper.tessellation import Tessellate, Partition, Voronoi
import scipy.sparse as sparse
import itertools

class RoiHelper(Helper):
    def __init__(self, input_data, roi=None, meta_label='all roi',
            rwa_file=None, autosave=True, verbose=True):
        Helper.__init__(self)
        input_data = Tessellate.prepare_data(self, input_data)

        if rwa_file is None and isinstance(input_data, Analyses):
            rwa_file = self.input_file

        self.meta_label = meta_label
        if roi is None:
            if meta_label in self.analyses:
                roi = self.get_bounding_boxes()
            else:
                self._toolbox_args = (rwa_file, autosave, verbose)
        else:
            self.set_bounding_boxes(roi)

        if roi is None:
            self.toolbox = None
        else:
            self.toolbox = RoiToolbox(roi, rwa_file, autosave, verbose)

    def set_bounding_boxes(self, roi, roi_size=None):

        trajectories = self.analyses.data

        roi_centers = np.array([ [ (bb[0]+bb[2])/2, (bb[1]+bb[3])/2 ] for bb in roi ])
        roi_adjacency = sparse.coo_matrix(([],([],[])), shape=(len(roi),len(roi)), dtype=bool)
        roi_vertices = np.array(list(itertools.chain(*[
            [ [bb[0], bb[1]], [bb[0], bb[3]], [bb[2], bb[3]],  [bb[2], bb[1]] ] for bb in roi ])))
        roi_cell_vertices = list(np.reshape(np.arange(len(roi_vertices)), (len(roi),-1)))
        roi_vertex_adjacency_row = np.concatenate([ 4*i + np.array([0, 0, 1, 2]) for i in range(len(roi)) ])
        roi_vertex_adjacency_col = np.concatenate([ 4*i + np.array([1, 3, 2, 3]) for i in range(len(roi)) ])
        _n_edges = roi_vertex_adjacency_row.size
        roi_vertex_adjacency = sparse.csr_matrix((np.ones((2*_n_edges,), dtype=bool),
               (np.r_[roi_vertex_adjacency_row, roi_vertex_adjacency_col],
                np.r_[roi_vertex_adjacency_col, roi_vertex_adjacency_row])), shape=(_n_edges, _n_edges))

        if roi_size is None:
            # assume same shape
            roi_size = roi_vertices[0][2,0] - roi_vertices[0][1,0]

        roi_polytopes = [ pt.box2poly([[bb[0],bb[2]],[bb[1],bb[3]]]) for bb in roi ]
        roi_index = []
        for i, x in enumerate(trajectories[['x','y']].values):
            for j, p in enumerate(roi_polytopes):
                if x in p:
                    roi_index.append((i,j))
        roi_index = tuple([ np.array(a) for a in zip(*roi_index) ])

        roi_mesh = Voronoi() # not really a Voronoi diagram...
        roi_mesh._preprocess(trajectories[['x','y']])
        roi_mesh.cell_centers = roi_centers
        roi_mesh.cell_adjacency = roi_adjacency
        roi_mesh.vertices = roi_vertices
        roi_mesh.cell_vertices = roi_cell_vertices
        roi_mesh.vertex_adjacency = roi_vertex_adjacency
        roi_mesh.cell_volume = np.full(len(roi), roi_size * roi_size)

        roi_partition = Partition(trajectories, roi_mesh)
        roi_partition.cell_index = roi_index

        #from tramway.tessellation.nesting import NestedTessellations
        #from tramway.tessellation.hexagon import HexagonalMesh
        #roi_partitions = Partition(trajectories,
        #            NestedTessellations(parent=roi_partition, factory=HexagonalMesh, ref_distance=.01))

        self.analyses[self.meta_label] = roi_partition

        if hasattr(self, '_toolbox_args'):
            self.toolbox = RoiToolbox(roi, *self._toolbox_args)
            del self._toolbox_args

    def get_bounding_boxes(self):
        roi_mesh = self.global_partition.tessellation
        roi = []
        for vertex_indices in roi_mesh.cell_vertices:
            bottom_left = roi_mesh.vertices[vertex_indices[0]]
            top_right = roi_mesh.vertices[vertex_indices[2]]
            roi.append(np.r_[bottom_left, top_right])
        return roi

    @property
    def global_partition(self):
        return self.analyses[self.meta_label].data

    @property
    def roi_labels(self):
        return self.toolbox.roi_labels

    def tessellate(self, *args, **kwargs):
        return self.toolbox.tessellate_all(self.analyses, *args, **kwargs)

    def infer(self, *args, **kwargs):
        return self.toolbox.infer_all(self.analyses, *args, **kwargs)

    def cell_plot(self, i, **kwargs):
        return self.toolbox.cell_plot(i, self.analyses, **kwargs)

    def map_plot(self, i, map_label, **kwargs):
        return self.toolbox.map_plot(i, self.analyses, map_label, **kwargs)

    def get_tessellation(self, i):
        return self.toolbox.get_tessellation(i, self.analyses)

    def get_map(self, i, map_label, full=False):
        return self.toolbox.get_map(i, self.analyses, map_label, full)


__all__ = [ 'AutosaveCapable', 'RoiToolbox', 'RoiHelper' ]

