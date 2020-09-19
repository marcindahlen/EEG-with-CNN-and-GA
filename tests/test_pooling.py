import numpy

from layers.layer_avgPool import AvgPool
from layers.layer_maxPool import MaxPool


class TestPooling:
    def test_pooling(self):
        data_1d = numpy.random.rand(15)
        data_2d = numpy.random.rand(15, 15)
        data_3d = numpy.random.rand(15, 15, 15)
        print(data_1d)
        print(data_3d)

        pooling_1d_layer = AvgPool((15,), (3,))
        pooling_2d_layer = MaxPool((15, 15), (3, 15))
        pooling_3d_layer = AvgPool((15, 15, 15), (3, 3, 15))

        print("")
        print("")
        output = pooling_1d_layer.forward_pass(data_1d)
        print(output)
        assert numpy.shape(output) == (3,)

        print("")
        print("")
        output = pooling_2d_layer.forward_pass(data_2d)
        print(output)
        assert numpy.shape(output) == (3, 15)

        print("")
        print("")
        output = pooling_3d_layer.forward_pass(data_3d)
        print(output)
        assert numpy.shape(output) == (3, 3, 15)
