
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

