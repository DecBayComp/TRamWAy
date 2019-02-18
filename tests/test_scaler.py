
import numpy
import pandas
from tramway.core.scaler import *

seed = 123456789

def example_list(n=5):
    return example_ndarray(n).tolist()

def example_ndarray(n=5):
    numpy.random.seed(seed)
    return numpy.random.randn(n, 3)

def example_structured_array(n=5):
    arr = example_ndarray(n)
    t = arr.dtype
    arr = numpy.array([ tuple(row) for row in arr ], dtype=numpy.dtype([('x', t), ('y', t), ('t', t)]))
    return arr

def example_series():
    return pandas.Series(example_list()[0], index=list('xyt'))

def example_dataframe(n=5):
    return pandas.DataFrame(example_ndarray(n), columns=list('xyt'))

def _isclose(a, b):
    return numpy.all(numpy.isclose(numpy.asarray(a).tolist(), b, atol=1e-2, rtol=0))


class TestScaler(object):

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
        assert _isclose(_array, scaled_array)
        # test unscale
        _array = scaler.unscale_point(scaled_array, inplace=False)
        assert _isclose(_array, numpy.asarray(unscaled_array))
        # test inplace (last test)
        if not isinstance(unscaled_array, list):
            _array = scaler.scale_point(unscaled_array, inplace=True)
            assert _array is unscaled_array
            assert _isclose(_array, scaled_array)

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
        assert _isclose(_array, scaled_array)
        # test unscale
        _array = scaler.unscale_point(_array, inplace=False)
        assert _isclose(_array, numpy.asarray(numpy.asarray(unscaled_array).tolist()))
        # test inplace (last test)
        _array = scaler.scale_point(unscaled_array, inplace=True)
        assert _array is unscaled_array
        assert _isclose(_array, scaled_array)
        # test additional sample data
        for unscaled, scaled in zip(args[0::2], args[1::2]):
            _scaled = scaler.scale_point(unscaled, inplace=False)
            assert _isclose(_scaled, scaled)

    def test_noscaling(self):
        expected_result = example_ndarray()
        self._test_unstructured(Scaler(), example_list(), expected_result)
        self._test_unstructured(Scaler(), example_ndarray(), expected_result)
        self._test_structured(Scaler(), example_structured_array(), expected_result)
        self._test_structured(Scaler(), example_series(), expected_result[0])
        df = example_dataframe()
        self._test_structured(Scaler(), df, expected_result, df.iloc[0], expected_result[0])

    def test_whitening(self):
        expected_result = example_ndarray()
        expected_result -= numpy.mean(expected_result, axis=0, keepdims=True)
        expected_result[:,:2] /= numpy.std(expected_result[:,:2])
        expected_result[:,-1] /= numpy.std(expected_result[:,-1])
        self._test_unstructured(whiten(), example_list(), expected_result)
        self._test_unstructured(whiten(), example_ndarray(), expected_result)
        #self._test_structured(whiten(), example_structured_array(), expected_result) # not supported
        #self._test_structured(whiten(), example_series(), expected_result[0]) # supporting Series here does not make sense
        df = example_dataframe()
        self._test_structured(whiten(), df, expected_result, df.iloc[0], expected_result[0])

    def test_unitrange(self):
        expected_result = example_ndarray()
        expected_result -= numpy.min(expected_result, axis=0, keepdims=True)
        expected_result[:,:2] /= numpy.max(expected_result[:,:2])
        expected_result[:,-1] /= numpy.max(expected_result[:,-1])
        self._test_unstructured(unitrange(), example_list(), expected_result)
        self._test_unstructured(unitrange(), example_ndarray(), expected_result)
        #self._test_structured(unitrange(), example_structured_array(), expected_result) # not supported
        #self._test_structured(unitrange(), example_series(), expected_result[0]) # supporting Series here does not make sense
        df = example_dataframe()
        self._test_structured(unitrange(), df, expected_result, df.iloc[0], expected_result[0])

