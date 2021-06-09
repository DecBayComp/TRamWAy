
import numpy
import pandas

seed = 123456789

from tramway.analyzer import RWAnalyzer, tessellers

from tramway.tessellation.base import *
class TestCellIndex(object):
    """
    TODO: generate a segmentation that makes `fast_min_nn` useful.
    """

    def example_spt_data(self, n=50):
        numpy.random.seed(seed)
        data = numpy.random.randn(n, 2)
        return pandas.DataFrame(data, columns=['x','y'])

    def preset_analyzer(self):
        a = RWAnalyzer()
        a.spt_data.from_dataframe(self.example_spt_data())
        a.tesseller = tessellers.KMeans
        a.tesseller.resolution = .3
        return a

    def test_min_nn(self):
        a = self.preset_analyzer()
        a.sampler.from_nearest_neighbors(10)
        assignment = a.sampler.sample(a.spt_data.dataframe)
        pts, cells = assignment.cell_index
        assigned_points = numpy.unique(pts)
        assert numpy.all(assigned_points == \
                numpy.arange(len(a.spt_data.dataframe)))
        cells = numpy.unique(cells)
        assert numpy.all(cells == numpy.arange(assignment.number_of_cells))

    def test_dyn_min_nn(self):
        a = self.preset_analyzer()
        a.sampler.from_nearest_neighbors(10)
        assignment_a = a.sampler.sample(a.spt_data.dataframe)
        pts_a, cells_a = assignment_a.cell_index
        b = self.preset_analyzer()
        b.sampler.from_nearest_neighbors(lambda _: (10, None))
        assignment_b = b.sampler.sample(b.spt_data.dataframe)
        pts_b, cells_b = assignment_b.cell_index
        sort = numpy.sort
        assert numpy.all(sort(pts_a) == sort(pts_b))
        assert numpy.all(sort(cells_a) == sort(cells_b))

