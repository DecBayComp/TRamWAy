
import numpy
import pandas

seed = 123456789


from tramway.core.scaler import *
class TestScaler(object):

    def example_list(self, n=5):
        return self.example_ndarray(n).tolist()

    def example_ndarray(self, n=5):
        numpy.random.seed(seed)
        return numpy.random.randn(n, 3)

    def example_structured_array(self, n=5):
        arr = self.example_ndarray(n)
        t = arr.dtype
        arr = numpy.array([ tuple(row) for row in arr ], dtype=numpy.dtype([('x', t), ('y', t), ('t', t)]))
        return arr

    def example_series(self):
        return pandas.Series(self.example_list()[0], index=list('xyt'))

    def example_dataframe(self, n=5):
        return pandas.DataFrame(self.example_ndarray(n), columns=list('xyt'))

    def _isclose(self, a, b):
        return numpy.all(numpy.isclose(numpy.asarray(a).tolist(), b, atol=1e-2, rtol=0))

    def _test_unstructured(self, scaler, unscaled_array, scaled_array):
        # test 'unscale_*' methods fail on uninitialized scalers
        try:
            scaler.unscale_point(unscaled_array, inplace=False)
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            assert True
        else:
            assert False
        # initialize
        scaler.euclidean = [0,1]
        _array = scaler.scale_point(unscaled_array, inplace=False)
        assert self._isclose(_array, scaled_array)
        # test unscale
        _array = scaler.unscale_point(scaled_array, inplace=False)
        assert self._isclose(_array, numpy.asarray(unscaled_array))
        # test inplace (last test)
        if not isinstance(unscaled_array, list):
            _array = scaler.scale_point(unscaled_array, inplace=True)
            assert _array is unscaled_array
            assert self._isclose(_array, scaled_array)

    def _test_structured(self, scaler, unscaled_array, scaled_array, *args):
        # test 'unscale_*' methods fail on uninitialized scalers
        try:
            scaler.unscale_point(unscaled_array, inplace=False)
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            assert True
        else:
            assert False
        # initialize
        scaler.euclidean = list('xy') # first 2 columns
        _array = scaler.scale_point(unscaled_array, inplace=False)
        assert self._isclose(_array, scaled_array)
        # test unscale
        _array = scaler.unscale_point(_array, inplace=False)
        assert self._isclose(_array, numpy.asarray(numpy.asarray(unscaled_array).tolist()))
        # test inplace (last test)
        _array = scaler.scale_point(unscaled_array, inplace=True)
        assert _array is unscaled_array
        assert self._isclose(_array, scaled_array)
        # test additional sample data
        for unscaled, scaled in zip(args[0::2], args[1::2]):
            _scaled = scaler.scale_point(unscaled, inplace=False)
            assert self._isclose(_scaled, scaled)

    def test_noscaling(self):
        expected_result = self.example_ndarray()
        self._test_unstructured(Scaler(), self.example_list(), expected_result)
        self._test_unstructured(Scaler(), self.example_ndarray(), expected_result)
        self._test_structured(Scaler(), self.example_structured_array(), expected_result)
        self._test_structured(Scaler(), self.example_series(), expected_result[0])
        df = self.example_dataframe()
        self._test_structured(Scaler(), df, expected_result, df.iloc[0], expected_result[0])

    def test_whitening(self):
        expected_result = self.example_ndarray()
        expected_result -= numpy.mean(expected_result, axis=0, keepdims=True)
        expected_result[:,:2] /= numpy.std(expected_result[:,:2])
        expected_result[:,-1] /= numpy.std(expected_result[:,-1])
        self._test_unstructured(whiten(), self.example_list(), expected_result)
        self._test_unstructured(whiten(), self.example_ndarray(), expected_result)
        #self._test_structured(whiten(), self.example_structured_array(), expected_result) # not supported
        #self._test_structured(whiten(), self.example_series(), expected_result[0]) # supporting Series here does not make sense
        df = self.example_dataframe()
        self._test_structured(whiten(), df, expected_result, df.iloc[0], expected_result[0])

    def test_unitrange(self):
        expected_result = self.example_ndarray()
        expected_result -= numpy.min(expected_result, axis=0, keepdims=True)
        expected_result[:,:2] /= numpy.max(expected_result[:,:2])
        expected_result[:,-1] /= numpy.max(expected_result[:,-1])
        self._test_unstructured(unitrange(), self.example_list(), expected_result)
        self._test_unstructured(unitrange(), self.example_ndarray(), expected_result)
        #self._test_structured(unitrange(), self.example_structured_array(), expected_result) # not supported
        #self._test_structured(unitrange(), self.example_series(), expected_result[0]) # supporting Series here does not make sense
        df = self.example_dataframe()
        self._test_structured(unitrange(), df, expected_result, df.iloc[0], expected_result[0])



from tramway.core.xyt import *
class TestXyt(object):

    def example_nxyt(self):
        return pandas.DataFrame([[1,.1,.1,.05],[1,.45,.45,.1],[1,.85,.65,.12],[1,1.1,.9,.15],[1,.9,1.1,.2],[1,.6,.9,.25],[1,.4,.5,.3]], columns=list('nxyt'))

    def example_bbox(self):
        return [0,0,1,1]

    def test_crop(self):
        expected_result = pandas.DataFrame([[1,.1,.1,.05,.35,.35,.05],[1,.45,.45,.1,.4,.2,.02],[1,.85,.65,.12,.25,.25,.03],[2,.6,.9,.25,-.2,-.4,.05]], columns=['n','x','y','t','dx','dy','dt'])
        tested_result = crop(self.example_nxyt(), self.example_bbox())
        assert isinstance(tested_result, pandas.DataFrame) and \
                numpy.all(tested_result.columns == expected_result.columns) and \
                numpy.all(tested_result.index == expected_result.index) and \
                numpy.allclose(tested_result, expected_result)
        return pandas.DataFrame([[1,.1,.1,.05],[1,.45,.45,.1],[1,.85,.65,.12],[2,.6,.9,.25],[2,.4,.5,.3]], columns=list('nxyt'))
        assert crop(self.example_nxyt(), self.example_bbox(), add_deltas=False).equals(expected_result[list('nxyt')])

    def test_reindex(self):
        from io import StringIO
        df=pandas.read_csv(StringIO("""\
             n          x          y           t
78894    19782  30.030399  30.079800   93.279999
78895    19782  30.054001  30.084801   93.320000
78897    19782  30.031000  30.090000   93.400000
659881  160801  30.197300  30.087900  452.640015
659882  160801  30.129200  30.116600  452.679993
659883  160801  30.191401  30.155399  452.720001
793381  192719  30.062099  30.159700  516.520020
796888  193576  30.150600  30.155199  518.280029
842499  204282  30.147100  30.192900  541.239990
907067  219796  30.189301  30.164801  570.400024\
"""), delim_whitespace=True)
        df = reindex_trajectories(df)
        assert numpy.all(df['n'].values == numpy.r_[1,1,2,3,3,3,4,5,6,7])
        reindex_trajectories(pandas.DataFrame([], columns=list('nxyt')))


from tramway.core.analyses import *
class TestAnalyses(object):

    def example_list(self):
        return ['a', 1, [], True]

    def example_dict(self, i=1):
        if i == 1:
            return dict(a=(), b=2, c='str')
        elif i == 2:
            return {'key1': 'val1', 'key2': 'val2'}

    def example_set(self):
        return set([-3, 'abc'])

    def example_tuple(self):
        return (False, {}, 1e-2)

    def example_tree(self):
        tree = Analyses('a string')
        subtree = Analyses(self.example_list())
        tree.add(subtree, label='a list', comment='heterogeneous list')
        subsubtree = Analyses(self.example_dict())
        subtree.add(subsubtree, label='a dict')
        return tree

    def example_comment(self):
        return 'example comment'

    def test_accessors(self):
        tree = self.example_tree()
        assert tree.data == 'a string'
        assert tree['a list'].data == self.example_list()
        assert tree['a list']['a dict'].data == self.example_dict()
        tree.add(self.example_set())
        tree['a list']['a dict']['a tuple'] = self.example_tuple()
        assert tree[0].data == self.example_set()
        tree['a list']['a dict'].comments['a tuple'] = self.example_comment()
        assert tree['a list']['a dict'].comments['a tuple'] == self.example_comment()
        tree['a list']['a dict']['another dict'] = {}
        tree['a list']['a dict']['another dict']['yet another dict'] = self.example_dict(2)
        art1, art2 = find_artefacts(tree, ((set, list), dict), ('a list', 'a dict'))
        assert art1 == self.example_list()
        assert art2 == self.example_dict()
        art1, art2 = find_artefacts(tree, ((set, list), dict), ('a list', 'a dict', 'another dict'))
        assert art2 == {}
        art1, art2 = find_artefacts(tree, ((set, list), dict), ('a list', 'a dict', 'another dict', 'yet another dict'))
        assert art2 == self.example_dict(2)

