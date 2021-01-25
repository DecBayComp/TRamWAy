# -*- coding: utf-8 -*-

# Copyright © 2020-2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import pandas as pd
try:
    import polytope as pt
except ImportError:
    pt = None
import copy
from tramway.core.xyt import crop, reindex_trajectories
import tramway.core.analyses.auto as autosaving
from tramway.helper import *
import re
import itertools
from collections import defaultdict, OrderedDict


try:
    from tqdm import tqdm
except ImportError:
    pass
else:
    rc.__available_packages__.add('tqdm')


class SupportRegions(object):
    """
    A support region can be either a region of interest or the union of regions of interest.
    Major processing steps such as tessellation and inference may operate
    on support regions so that a region of interest simply acts as a window.

    This class offers a base implementation for :class:`UnitRegions` and :class:`GroupedRegions`.
    """
    __slots__ = ('gen_label','update_metadata','verbose')
    def __init__(self, region_label=None, update_metadata=None, verbose=True):
        self.__reset__()
        self.gen_label = region_label
        self.update_metadata = update_metadata
        self.verbose = verbose
    def __reset__(self):
        raise NotImplementedError('abstract method')
    def tessellate(self, r, analysis_tree, *args, **kwargs):
        if isinstance(r, str):
            if r == 'all':
                description = 'Tessellating the regions of interest'
            else:
                description = r
            any_new = False
            for r in self.iter_regions(description):
                if self.tessellate(r, analysis_tree, *args, **kwargs):
                    any_new = True
            return any_new
        #
        label = self.region_label(r)
        if label in analysis_tree:
            return False
        else:
            if self.verbose:
                print(label)
            trajectories = self.crop(r, analysis_tree.data)
            partition = tessellate(trajectories, *args, **kwargs)
            analysis_tree[label] = partition
            if self.update_metadata is not None:
                self.update_metadata(analysis_tree[label])
            return True
    def infer(self, r, analysis_tree, *args, **kwargs):
        if isinstance(r, str) and (r == 'all' or r not in analysis_tree):
            if r == 'all':
                description = 'Inferring dynamics parameters'
            else:
                description = r
            any_new = False
            for r in self.iter_regions(description):
                if self.infer(r, analysis_tree, *args, **kwargs):
                    any_new = True
            return any_new
        #
        if isinstance(r, str):
            label = r
        else:
            label = self.region_label(r)
        if label in analysis_tree:
            skip_interrupted = kwargs.pop('preserve_interrupted_inferences', False)
            try:
                maps = analysis_tree[label][kwargs['output_label']].data
            except KeyError: # either 'output_label' in kwargs or kwargs['output_label'] in analysis_tree
                pass
            else:
                try:
                    if skip_interrupted or maps.resolution.upper() != 'INTERRUPTED':
                        return False
                except AttributeError: # either resolution in maps or upper in maps.resolution
                    return False
            kwargs['input_label'] = label
            output_label = analysis_tree[label].autoindex(kwargs.get('output_label', None))
            if self.verbose:
                print('{} -- {}'.format(label, output_label))
            infer(analysis_tree, *args, **kwargs)
            if self.update_metadata is not None:
                self.update_metadata(analysis_tree[label][output_label])
            return True
        else:
            import warnings
            warnings.warn("no partition available for region '{}'".format(label))
        return False
    def reset_roi(self, r, analysis_tree, *args, **kwargs):
        del analysis_tree[self.region_label(r)]
    def __range__(self, n, desc=None):
        iterable = not isinstance(n, int)
        if iterable:
            iterator = n
            n = len(iterator)
        if self.verbose and rc.__user_interaction__ and rc.__has_package__('tqdm'):
            iternum = tqdm(range(n), desc=desc)
        else:
            iternum = range(n)
        if iterable:
            return zip(iternum, iterator)
        else:
            return iternum

class UnitRegions(SupportRegions):
    """
    Regions of interest are considered separately, independently of whether they overlap or not.
    """
    __slots__ = ('unit_region','_bw_comp',)
    def __init__(self, *args, **kwargs):
        SupportRegions.__init__(self, *args, **kwargs)
        self._bw_comp = False
    def __reset__(self):
        self.unit_region = OrderedDict()
    def add_collection(self, unit_regions, label=None):
        if label is None:
            label = ''
        self.unit_region[label] = unit_regions
    def __len__(self):
        return sum([0]+[ len(self.unit_region[r]) for r in self.unit_region ])
    def __contains__(self, r):
        return 0<=r and r<len(self)
    def __iter__(self):
        yield from range(len(self))
    def __getitem__(self, r):
        _r = r
        for coll in self.unit_region:
            rs = self.unit_region[coll]
            if _r<len(rs):
                return rs[_r]
            else:
                _r -= len(rs)
        raise IndexError('out of bounds: {}'.format(r))
    def __setitem__(self, r, bounds):
        _r = r
        for coll in self.unit_region:
            rs = self.unit_region[coll]
            if _r<len(rs):
                rs[_r] = bounds
                return
            else:
                _r -= len(rs)
        raise IndexError('out of bounds: {}'.format(r))
    def region_label(self, r):
        if not isinstance(r, int):
            assert not r[1:]
            r = r[0]
        for collection in self.unit_region:
            n = len(self.unit_region[collection])
            if n < r:
                r -= n
            else:
                break
        if self._bw_comp:
            return self.gen_label(r, collection if collection else None)
        else:
            return self.gen_label({ collection if collection else '': [r] })
    @property
    def region_labels(self):
        labels = []
        for collection in self.unit_region:
            n = len(self.unit_region[collection])
            if not collection:
                collection = None
            for r in range(n):
                if self._bw_comp:
                    labels.append(self.gen_label(r, collection))
                else:
                    labels.append(self.gen_label({ collection if collection else '': [r] }))
        return labels
    def unit_to_region(self, u, collection=None):
        if collection is None:
            collection = ''
        r = 0
        for label in self.unit_region:
            if label == collection:
                return r+u
            else:
                r += len(self.unit_region[label])
        raise KeyError("no such roi collection: '{}'".format(collection))
    def region_to_units(self, r):
        """
        Returns a `dict` with a single key and single-element list value.

        For compatibility with : class:`GroupedRegions`.
        """
        _r = r
        for coll in self.unit_region:
            rs = self.unit_region[coll]
            if _r<len(rs):
                return { coll: [_r] }
            else:
                _r -= len(rs)
        raise IndexError('out of bounds: {}'.format(r))
    #def collection_range(self, collection_label):
    #    m = 0
    #    for label in self.unit_region:
    #        k = len(self.unit_region[label])
    #        if collection_label == label:
    #            n = m + k
    #            break
    #        else:
    #            m += k
    #    return m, n
    def iter_regions(self, desc=None):
        return self.__range__(len(self), desc)
    def crop(self, r, df):
        n_space_cols = len([ col for col in 'xyz' if col in df.columns ])
        for regions in self.unit_region.values():
            if len(regions) < r:
                r -= len(regions)
            else:
                if isinstance(regions[r], (pt.Polytope, pt.Region)):
                    raise NotImplementedError
                else:
                    _min,_max = regions[r]
                    if n_space_cols < _min.size:
                        assert _min.size == n_space_cols + 1
                        df = df[(_min[-1] <= df['t']) & (df['t'] <= _max[-1])]
                        df = crop(df, np.r_[_min[:-1], _max[:-1]-_min[:-1]])
                    else:
                        df = crop(df, np.r_[_min,_max-_min])
                return df

class GroupedRegions(SupportRegions):
    """
    Overlapping regions of interest are pooled together and form a unique support region.

    Pooling disregards which collection a ROI pertains to.
    """
    def __reset__(self):
        self.unit_region = []
        self._unit_polytope = {}
        self.group = {}
        self.index = OrderedDict()
    def unit_bounding_box(self, i, exact_only=False, atleast_2d=False):
        r = self.unit_region[i]
        if isinstance(r, tuple):
            if atleast_2d:
                _min,_max = r
                if not _min.shape[1:]:
                    assert not _max.shape[1:]
                    r = _min[np.newaxis,:], _max[np.newaxis,:]
        elif isinstance(r, list):
            if atleast_2d:
                _min,_max = r
                if _min.shape[1:]:
                    assert _max.shape[1:]
                    r = tuple(r)
                else:
                    assert not _max.shape[1:]
                    r = _min[np.newaxis,:], _max[np.newaxis,:]
            else:
                r = tuple(r)
            self.unit_region[i] = r
        else:#isinstance(r, (pt.Polytope, pt.Region)):
            if exact_only:
                r = None, None
            elif atleast_2d:
                r = r.bounding_box
            else:
                _min,_max = r.bounding_box
                r = np.ravel(_min), np.ravel(_max)
        return r
    def unit_polytope(self, i):
        try:
            return self._unit_polytope[i]
        except KeyError:
            r = self.unit_region[i]
            if isinstance(r, pt.Region):
                pass
            else:
                if isinstance(r, (tuple, list)):
                    r = pt.box2poly(list(zip(*r)))
                    self._unit_polytope[i] = r
                r = pt.Region([r])
            return r
    @property
    def unit_bounding_boxes(self):
        return [ self.unit_bounding_box(i) for i in range(len(self.unit_region)) ]
    @property
    def unit_polytopes(self):
        return [ self.unit_polytope(i) for i in range(len(self.unit_region)) ]
    def adjacent(self, i, j):
        _min_i,_max_i = self.unit_bounding_box(i, True)
        _min_j,_max_j = self.unit_bounding_box(j, True)
        if _min_i is None or _min_j is None:
            _ri,_rj = self.unit_polytope(i), self.unit_polytope(j)
            return pt.is_adjacent(_ri,_rj)
        else:
            return np.all(_min_i<=_max_j) and np.all(_min_j<=_max_i)
    def add_collection(self, unit_regions, label=None):
        # check if already existing
        if label in self.index:
            if 1<len(self.index):
                raise RuntimeError('cannot overwrite a collection if other collections have already been defined')
            self.__reset__()
        #
        i0 = len(self.unit_region)
        self.unit_region += list(unit_regions)
        # first group overlapping unit regions in the collection
        current_index = max(self.group.keys())+1 if self.group else 0
        not_an_index = -1
        n = len(unit_regions)
        assignment = np.full(n, not_an_index, dtype=int)
        groups = dict()
        for i in range(n):
            region_i = unit_regions[i]
            if isinstance(region_i, pt.Polytope):
                region_i = pt.Region([region_i])
                _min_i = _max_i = None
            elif isinstance(region_i, pt.Region):
                _min_i = _max_i = None
            else:#if isinstance(region_i, (tuple, list)):
                _min_i, _max_i = region_i
                region_i = None
            group_with = set()
            if assignment[i] == not_an_index:
                group_index = current_index
                group = set([i0+i])
            else:
                group_index = assignment[i]
                group_with.add(group_index)
                group = groups[group_index]
                assert i0+i in group
            for j in range(i+1,n):
                if i0+j in group:
                    continue
                region_j = unit_regions[j]
                if isinstance(region_j, (pt.Polytope, pt.Region)):
                    if region_i is None:
                        region_i = pt.box2poly(list(zip(_min_i,_max_i)))
                        self._unit_polytope[i0+i] = region_i
                    i_and_j_are_adjacent = pt.is_adjacent(region_i, region_j)
                else:#if isinstance(region_j, (tuple, list)):
                    _min_j,_max_j = region_j
                    if _min_i is None:
                        region_j = pt.box2poly(list(zip(_min_j,_max_j)))
                        self._unit_polytope[i0+j] = region_j
                        i_and_j_are_adjacent = pt.is_adjacent(region_i, region_j)
                    else:
                        i_and_j_are_adjacent = np.all(_min_i<=_max_j) and np.all(_min_j<=_max_i)
                if i_and_j_are_adjacent:
                    if assignment[j]==not_an_index:
                        group.add(i0+j)
                    else:
                        other_group_index = assignment[j]
                        group_with.add(other_group_index)
                        group |= groups.pop(other_group_index)
            if group_with:
                group_index = min(group_with)
            else:
                current_index += 1
            groups[group_index] = group
            group = np.array(list(group)) - i0 # indices in `assignment`
            assignment[group] = group_index
        #
        # merge the new and existing groups together
        for g in list(groups.keys()):
            adjacent = set()
            for h in self.group:
                g_and_h_are_adjacent = False
                for i in groups[g]:
                    for j in self.group[h]:
                        if self.adjacent(i,j):
                            g_and_h_are_adjacent = True
                            break
                    if g_and_h_are_adjacent:
                        adjacent.add(h)
                        break
            if adjacent:
                h = min(adjacent)
                for i in adjacent - {h}:
                    self.group[h] |= self.group.pop(i)
                self.group[h] |= groups.pop(g)
                assignment[assignment==g] = h
        if groups:
            self.group.update(groups)
        #
        if label is None:
            label = ''
        self.index[label] = assignment
        #
        self.reverse_index = np.c_[
                np.repeat(np.arange(len(self.index)), [ len(self.index[s]) for s in self.index ]),
                np.concatenate([ np.arange(len(self.index[s])) for s in self.index ]) ]
    def unit_to_region(self, u, collection=None):
        if collection is None:
            collection = ''
        return self.index[collection][u]
    def region_to_units(self, r):
        coll_num = self.reverse_index[list(self.group[r])]
        units = defaultdict(list)
        for coll_ix, ix_in_coll in coll_num:
            label = self.collection_labels[coll_ix]
            units[label].append(ix_in_coll)
        return units
    @property
    def collection_labels(self):
        return list(self.index.keys())
    def region_label(self, r):
        return self.gen_label(self.region_to_units(r))
    @property
    def region_labels(self):
        return [ self.region_label(r) for r in self.group ]
    def __len__(self):
        return len(self.group)
    def __contains__(self, r):
        return r in self.group
    def __iter__(self):
        return iter(self.group)
    def __getitem__(self, r):
        return self.group[r]
    #def __setitem__(self, r, roi_set):
    #    self.group[r] = roi_set
    #def __delitem__(self, r):
    #    del self.group[r]
    def collection_range(self, collection_label):
        m = 0
        for label in self.index:
            k = len(self.index[label])
            if collection_label == label:
                n = m + k
                break
            else:
                m += k
        return m, n
    def iter_regions(self, desc=None):
        rs = list(self.group.keys())[::-1]
        done = 0
        for _ in self.__range__(len(self.unit_region), desc):
            if done == 0:
                r = rs.pop()
                unit_roi = self.region_to_units(r)
                done = sum([ len(roi) for roi in unit_roi.values() ])
                yield r
            done -= 1
    def crop(self, r, df):
        n_space_cols = len([ col for col in 'xyz' if col in df.columns ])
        loc_indices = set()
        df_r = None
        for u in self.group[r]:
            if isinstance(self.unit_region[u], (pt.Polytope, pt.Region)):
                raise NotImplementedError
            else:
                _min,_max = self.unit_region[u]
                if n_space_cols < _min.size:
                    assert _min.size == n_space_cols + 1
                    df_u = df[(_min[-1] <= df['t']) & (df['t'] <= _max[-1])]
                    df_u = crop(df_u, np.r_[_min[:-1], _max[:-1]-_min[:-1]], preserve_index=True)
                else:
                    df_u = crop(df, np.r_[_min,_max-_min], preserve_index=True)
            if df_r is None:
                df_r = df_u
            else:
                df_r = pd.merge(df_r, df_u, how='outer')
        return reindex_trajectories(df_r.sort_values(by=['n','t']))


class RoiCollection(object):
    """
    """
    def __init__(self, roi, label=None, regions=None):
        self.label = label
        self.regions = regions
        self.regions.add_collection(roi, label)
    @property
    def bounding_box(self):
        try:
            return self.regions.unit_region[self.label]
        except TypeError:
            m,n = self.regions.collection_range(self.label)
            return self.regions.unit_bounding_boxes[m:n]
    def __len__(self):
        raise NotImplementedError
        # TODO: this collection only?
        return len(self.regions.unit_region)
    def subset_index(self, r):
        return self.regions.unit_to_region(r, self.label)
    def subset_label(self, r=None, s=None):
        if s is None:
            s = self.subset_index(r)
        return self.regions.region_label(s)
    def get_subset(self, r=None, s=None):
        if s is None:
            s = self.subset_index(r)
        try:
            return self.regions.group[s]
        except AttributeError:
            return [s]
    def roi_label(self, r):
        try:
            return self.regions.unit_region_label(r, self.label)
        except AttributeError:
            return self.regions.gen_label({self.label: [r]})
    def get_subtree(self, i, analysis_tree):
        label = self.subset_label(i)
        if label in analysis_tree:
            return analysis_tree[label]
    def overlaps(self, i):
        return 1 < len(self.get_subset(i))
    def get_map(self, i, analysis_tree, map_label, full=False):
        """
        returns the `Map` object for unit roi *i*.
        """
        label = self.subset_label(i)
        try:
            subtree = analysis_tree[label]
            maps = subtree[map_label].data
        except KeyError:
            return None
        if not full and self.overlaps(i):
            tessellation = subtree.data.tessellation
            try:
                segments = len(tessellation.time_lattice)
            except AttributeError:
                segments = None
            else:
                tessellation = tessellation.spatial_mesh
            coords = tessellation.cell_centers
            dim = coords.shape[1]
            _min,_max = self.bounding_box[i]
            inside = np.all((_min[np.newaxis,:dim]<coords) & (coords<_max[np.newaxis,:dim]),axis=1)
            if segments:
                inside = np.tile(inside, segments)
            maps.maps = maps.maps[inside[maps.maps.index]]
        return maps
    def get_cells(self, i, analysis_tree):
        """
        returns the `Partition` object for unit roi *i*.
        """
        label = self.subset_label(i)
        cells = analysis_tree[label].data
        tessellation = cells.tessellation
        try:
            tessellation = tessellation.spatial_mesh
        except AttributeError:
            pass
        if self.overlaps(i):
            coords = tessellation.cell_centers
            dim = coords.shape[1]
            _min,_max = self.bounding_box[i]
            inner_cell = np.all((_min[np.newaxis,:dim]<coords) & (coords<_max[np.newaxis,:dim]),axis=1)
        else:
            inner_cell = np.ones(tessellation.number_of_cells, dtype=bool)
        return cells, inner_cell
    def get_tessellation(self, i, analysis_tree):
        """
        returns the `Tessellation` object for unit roi *i*,
        together with a boolean array so that the *k*-th element is ``True``
        if microdomain *k* lies within roi *i*.
        """
        cells, inner_cell = self.get_cells(i, analysis_tree)
        return cells.tessellation, inner_cell
    def cell_plot(self, i, analysis_tree, decorate=True, **kwargs):
        """
        plots the `Partition` for unit roi *i*.
        """
        if isinstance(i, str):
            label = i
            title = kwargs.pop('title', label)
        else:
            label = self.subset_label(i)
            title = kwargs.pop('title', self.roi_label(i))
        if 'delaunay' not in kwargs:
            # anticipate future changes in `helper.tessellation.cell_plot`
            voronoi = kwargs.pop('voronoi', dict(centroid_style=None))
            kwargs['voronoi'] = voronoi
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'equal'
        plot_bb = decorate and self.overlaps(i)
        if True:#plot_bb:
            kwargs['show'] = False
        #
        cell_plot(analysis_tree, label=label, title=title, **kwargs)
        #
        if decorate:
            if 'axes' in kwargs:
                ax = kwargs['axes']
            else:
                import matplotlib.pyplot as plt
                ax = plt.gca()
            _min,_max = self.bounding_box[i]
            x0, y0, x1, y1 = _min[0], _min[1], _max[0], _max[1]
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
    def map_plot(self, i, analysis_tree, map_label, decorate=True, **kwargs):
        """
        plots the `Maps` for unit roi *i*.
        """
        if isinstance(i, str):
            label = i
            title = kwargs.pop('title', label)
        else:
            label = self.subset_label(i)
            title = kwargs.pop('title', self.roi_label(i))
        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'equal'
        kwargs['show'] = False
        plot_bb = decorate and self.overlaps(i)
        #
        map_plot(analysis_tree, label=(label,map_label), title=title, **kwargs)
        #
        if decorate:
            _min,_max = self.bounding_box[i]
            x0, y0, x1, y1 = _min[0], _min[1], _max[0], _max[1]
            xc, yc = .5 * (x0 + x1), .5 * (y0 + y1)
            if 'axes' in kwargs:
                ax = kwargs['axes']
            else:
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
    def reset_roi(self, i, analysis_tree):
        self.regions.reset_roi(self.get_subset_index(i), analysis_tree)

class RoiCollections(AutosaveCapable):
    """
    collection of all collections of roi.

    manages the analysis tree for the spt data.
    Modifications can be automatically saved into an *.rwa* file.
    """
    def __init__(self, group_overlapping_roi=False, rwa_file=None, autosave=True, metadata=None, verbose=True, _bw_comp=False):
        AutosaveCapable.__init__(self, rwa_file, autosave)
        label = self.roi_label
        kwargs = dict()
        if group_overlapping_roi:
            Regions = GroupedRegions
        else:
            Regions = UnitRegions
            if _bw_comp:
                label = self.single_roi_label
        self.regions = Regions(
                region_label=label,
                update_metadata=metadata,
                verbose=verbose)
        if group_overlapping_roi:
            self.regions.unit_region_label = self.single_roi_label
        elif _bw_comp:
            self.regions._bw_comp = True
        self.collections = {}
    def __len__(self):
        return len(self.collections)
    def __contains__(self, label):
        return label in self.collections
    def __iter__(self):
        return iter(self.collections)
    def __setitem__(self, label, roi):
        self.collections[label] = roi if isinstance(roi, RoiCollection) else RoiCollection(roi, label, self.regions)
    def __getitem__(self, label):
        return self.collections[label]
    def __delitem__(self, label):
        del self.collections[label]
    @property
    def verbose(self):
        return self.regions.verbose
    @verbose.setter
    def verbose(self, v):
        self.regions.verbose = v
    @property
    def numeric_format(self):
        return '{:0>3d}'
    def roi_label(self, coll_num):
        label = []
        for coll in coll_num:
            num = sorted(coll_num[coll])
            num_label = '-'.join([ self.numeric_format.format(i) for i in num ])
            if coll:
                label.append( ' '.join((coll, num_label)) )
            else:
                label.append( 'roi'+num_label )
        return ' - '.join(label)
    def single_roi_label(self, i, collection_label=None):
        """ for figure titles """
        num_label = self.numeric_format.format(i) if isinstance(i, int) else i
        if collection_label:
            return '{} roi {}'.format(collection_label, num_label)
        else:
            return 'roi'+num_label
    @property
    def roi_labels(self):
        return self.regions.region_labels
    @property
    def tessellate_desc(self):
        return 'Tessellating the regions of interest'
    def tessellate(self, analysis_tree, *args, **kwargs):
        self.regions.tessellate(self.tessellate_desc, analysis_tree, *args, **kwargs)
    @property
    def infer_desc(self):
        return 'Inferring dynamics parameters'
    def infer(self, analysis_tree, *args, **kwargs):
        self.regions.infer(self.infer_desc, analysis_tree, *args, **kwargs)

    def reset(self):
        self.regions.__reset__()
        self.collections = {}


#from .base import Helper, Analyses
#from tramway.helper.tessellation import Tessellate, Partition, Voronoi
import scipy.sparse as sparse
import itertools

class RoiHelper(Helper):
    """
    maintains an extra data artefact in the *.rwa* file to store/load
    roi definitions.
    """
    def __init__(self, input_data, roi=None,
            meta_label='all %Sroi', meta_label_sep=' ',
            rwa_file=None, autosave=True, verbose=True,
            group_overlapping_roi=True, _bw_comp=False):
        Helper.__init__(self)
        input_data = Tessellate.prepare_data(self, input_data)

        if rwa_file is None and isinstance(input_data, Analyses):
            rwa_file = self.input_file
            if isinstance(rwa_file, list):
                assert not rwa_file[1:]
                rwa_file = rwa_file[0]

        analyses = self.analyses
        self.analyses = autosaving.Analyses(rwa_file, autosave)
        self.analyses.analyses = analyses

        self.meta_label_pattern = meta_label
        self.meta_label_sep = meta_label_sep

        #self.collections = RoiCollections(group_overlapping_roi,
        #        rwa_file, autosave, self.add_metadata, verbose)
        self.collections = RoiCollections(autosave=False, metadata=self.add_metadata,
                group_overlapping_roi=group_overlapping_roi, _bw_comp=_bw_comp)

        if roi is None:
            for coll_label, meta_label in self.get_meta_labels().items():
                self.collections[coll_label] = self.get_bounding_boxes(meta_label=meta_label)
        elif isinstance(roi, dict):
            for label in roi:
                self.set_bounding_boxes(roi[label], collection_label=label)
        else:
            self.set_bounding_boxes(roi)
            self.collections[''] = roi

        self._extra_artefacts = {}

    def get_meta_labels(self):
        pattern = self.meta_label_pattern.replace('%s', '(?P<collection>.*)')
        pattern = pattern.replace('%S', '((?P<collection>.+){})?'.format(self.meta_label_sep))
        labels = {}
        for label in self.analyses.labels:
            match = re.fullmatch(pattern, label)
            if match is not None:
                collection = match.group('collection')
                if collection is None:
                    collection = ''
                labels[collection] = label
        return labels

    def to_meta_label(self, collection_label):
        if collection_label:
            meta_label = self.meta_label_pattern.replace('%S', collection_label+self.meta_label_sep)
        else:
            collection_label = ''
            meta_label = self.meta_label_pattern.replace('%S', '')
        meta_label = meta_label.replace('%s', collection_label)
        return meta_label

    def set_bounding_boxes(self, roi, roi_size=None, collection_label=None):

        trajectories = self.analyses.data

        space_cols = [ col for col in 'xyz' if col in trajectories.columns ]
        def _space(_xyt):
            _xy = _xyt[:-1] if len(space_cols) < _xyt.size else _xyt
            return _xy

        roi_centers = np.stack([ _space((_min+_max)/2) for _min,_max in roi ], axis=0)
        roi_adjacency = sparse.coo_matrix(([],([],[])), shape=(len(roi),len(roi)), dtype=bool)
        # 2D only
        roi_vertices = np.vstack(list(itertools.chain(*[
            [ [_min[0],_min[1]], [_min[0],_max[1]], [_max[0],_max[1]],  [_max[0],_min[1]] ] for _min,_max in roi ])))
        roi_cell_vertices = list(np.reshape(np.arange(len(roi_vertices)), (len(roi),-1)))
        roi_vertex_adjacency_row = np.concatenate([ 4*i + np.array([0, 0, 1, 2]) for i in range(len(roi)) ])
        roi_vertex_adjacency_col = np.concatenate([ 4*i + np.array([1, 3, 2, 3]) for i in range(len(roi)) ])
        _n_edges = roi_vertex_adjacency_row.size
        roi_vertex_adjacency = sparse.csr_matrix((np.ones((2*_n_edges,), dtype=bool),
               (np.r_[roi_vertex_adjacency_row, roi_vertex_adjacency_col],
                np.r_[roi_vertex_adjacency_col, roi_vertex_adjacency_row])), shape=(_n_edges, _n_edges))

        if roi_size is None:
            # assume same shape
            roi_size = roi_vertices[2,0] - roi_vertices[1,0]

        #roi_polytopes = [ pt.box2poly([[_min[0],_max[0]],[_min[1],_max[1]]]) for _min,_max in roi ]
        roi_index = []
        #for i, x in enumerate(trajectories[['x','y']].values):
        #    for j, p in enumerate(roi_polytopes):
        #        if x in p:
        #            roi_index.append((i,j))
        #roi_index = tuple([ np.array(a) for a in zip(*roi_index) ])
        xy = xyt = None
        I = []
        for j, (_min,_max) in enumerate(roi):
            if len(space_cols) < _min.size:
                if xyt is None:
                    xyt = trajectories[space_cols+['t']].values
                pt = xyt
            else:
                if xy is None:
                    xy = trajectories[space_cols].values
                pt = xy
            i, = np.nonzero(np.all((_min[np.newaxis,:] <= pt) & (pt <= _max[np.newaxis,:]), axis=1))
            I.append(i)
        J = np.repeat(np.arange(len(I)), [ len(i) for i in I])
        I = np.concatenate(I)
        roi_index = I,J

        roi_mesh = Voronoi() # not really a Voronoi diagram...
        roi_mesh._preprocess(trajectories[space_cols])
        roi_mesh.cell_centers = roi_centers
        roi_mesh.cell_adjacency = roi_adjacency
        roi_mesh.vertices = roi_vertices
        roi_mesh.cell_vertices = roi_cell_vertices
        roi_mesh.vertex_adjacency = roi_vertex_adjacency
        roi_mesh.cell_volume = np.full(len(roi), roi_size * roi_size)

        # TODO: wrap the spatial mesh into a TimeLattice object to store the time segment

        roi_partition = Partition(trajectories, roi_mesh)
        roi_partition.cell_index = roi_index

        #from tramway.tessellation.nesting import NestedTessellations
        #from tramway.tessellation.hexagon import HexagonalMesh
        #roi_partitions = Partition(trajectories,
        #            NestedTessellations(parent=roi_partition, factory=HexagonalMesh, ref_distance=.01))

        meta_label = self.to_meta_label(collection_label)
        self.analyses[meta_label] = roi_partition
        self.add_metadata(self.analyses[meta_label])

        if collection_label is None:
            collection_label = ''
        self.collections[collection_label] = roi

    def get_bounding_boxes(self, collection_label=None, meta_label=None):
        if meta_label is None:
            if collection_label is None:
                collection_label = ''
            return self.collections[collection_label].bounding_box

        roi_mesh = self.get_global_partition(meta_label=meta_label).tessellation
        roi = []
        for vertex_indices in roi_mesh.cell_vertices:
            assert len(vertex_indices) == 4
            bottom_left = roi_mesh.vertices[vertex_indices[0]]
            top_right = roi_mesh.vertices[vertex_indices[2]]
            roi.append((bottom_left, top_right))
        return roi

    def get_global_partition(self, collection_label=None, meta_label=None):
        if meta_label is None:
            meta_label = self.to_meta_label(collection_label)
        return self.analyses[meta_label].data

    @property
    def roi_labels(self):
        return self.collections.roi_labels

    def tessellate(self, *args, **kwargs):
        with self.analyses.autosaving(kwargs.pop('autosave',None)):
            return self.collections.tessellate(self.analyses, *args, **kwargs)

    def infer(self, *args, **kwargs):
        with self.analyses.autosaving(kwargs.pop('autosave',None)):
            return self.collections.infer(self.analyses, *args, **kwargs)

    def cell_plot(self, i, collection_label='', **kwargs):
        return self.collections[collection_label].cell_plot(i, self.analyses, **kwargs)

    def map_plot(self, i, map_label, collection_label='', **kwargs):
        return self.collections[collection_label].map_plot(i, self.analyses, map_label, **kwargs)

    def get_tessellation(self, i, collection_label=''):
        return self.collections[collection_label].get_tessellation(i, self.analyses)

    def get_map(self, i, map_label, full=False, collection_label=''):
        return self.collections[collection_label].get_map(i, self.analyses, map_label, full)

    def reset_roi(self, i, collection_label=''):
        self.collections[collection_label].reset_roi(i)

    def save_analyses(self, output_file=None, verbose=None, force=None, **kwargs):
        if output_file is not None:
            raise NotImplementedError
        self.analyses.save(out_of_context=True)
        return
        # deprecated
        if output_file is None:
            output_file = self.collections.rwa_file
        if verbose is None:
            verbose = self.collections.verbose
        Helper.save_analyses(self, output_file, verbose, force, **kwargs)

    def set_base_grid(self, label, grid):
        label = 'tramway.roi.RoiHelper.basegrid:'+label
        self.analyses[label] = grid
        self.add_metadata(self.analyses[label])

    def get_base_grid(self, label, from_root=False):
        label = 'tramway.roi.RoiHelper.basegrid:'+label
        if from_root:
            try:
                artefact = self._extra_artefacts[label]
            except KeyError:
                artefact = None
                if self.collections.rwa_file:
                    from rwa import HDF5Store, lazyvalue
                    f = HDF5Store(self.collections.rwa_file, 'r')
                    try:
                        artefact = lazyvalue(f.peek(label))
                    except KeyError:
                        pass
                    else:
                        self._extra_artefacts[label] = artefact
                    finally:
                        f.close()
        else:
            artefact = self.analyses[label].data
        return artefact

    def reset(self):
        self.collections.reset()


__all__ = [ 'SupportRegions', 'UnitRegions', 'GroupedRegions', 'RoiCollection', 'RoiCollections', 'RoiHelper' ]

