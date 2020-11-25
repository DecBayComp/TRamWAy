# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from ..abc import AnalyzerNode, TessellationPostProcessing
from tramway.helper.tessellation import Partition, delete_low_count_cells, update_cell_centers

class CellMerger(AnalyzerNode):
    __slots__ = ()
    def post_process(self, tessellation, spt_dataframe):
        # does nothing
        return tessellation

class ByTranslocationCount(CellMerger):
    """
    Merges the cells/microdomains that exhibit a number of translocations
    below some threshold.

    .. note::

        For now, all locations are counted instead of translocations only.

    """
    __slots__ = ('_count_threshold', '_update_centroids')
    def __init__(self, **kwargs):
        CellMerger.__init__(self, **kwargs)
        self._count_threshold = None
        self._update_centroids = None
    def post_process(self, tessellation, spt_dataframe):
        dim = tessellation.cell_centers.shape[1]
        min_ncells = dim + 2
        ncells = tessellation.number_of_cells
        cells = Partition(spt_dataframe, tessellation) # no options; Voronoi partition
        label = True
        while True:
            # TODO: consider translocations instead of any location
            cells, deleted_cells, label = delete_low_count_cells(
                    cells, self.count_threshold, 'count', label)
            if deleted_cells.size == 0:
                break
            if cells.number_of_cells < min_ncells:
                raise RuntimeError('too few remaining cells ({}<{})'.format(cells.number_of_cells, min_ncells))
            if self.update_centroids:
                cells = update_cell_centers(cells, self.update_centroids)
        return cells.tessellation
    @property
    def count_threshold(self):
        return self._count_threshold
    @count_threshold.setter
    def count_threshold(self, thres):
        self._count_threshold = thres
    @property
    def update_centroids(self):
        return self._update_centroids
    @update_centroids.setter
    def update_centroids(self, update):
        self._update_centroids = update

TessellationPostProcessing.register(ByTranslocationCount)


__all__ = ['ByTranslocationCount']

