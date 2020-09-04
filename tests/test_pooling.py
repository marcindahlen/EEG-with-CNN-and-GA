import numpy

from layers.layer_avgPool import AvgPool


class TestPooling:
    def test_pooling(self):
        data_1d = numpy.random.rand(15)
        data_3d = numpy.random.rand(15, 15, 15)
        print(data_1d)
        print(data_3d)

        pooling_layer = AvgPool()

        print("")
        print("")
        output = pooling_layer.forward_pass(data_1d)
        print(output)
        assert numpy.shape(output) == (1, 3, 1)
        print("")
        print("")
        output = pooling_layer.forward_pass(data_3d)
        print(output)

        assert numpy.shape(output) == (1, 3, 1, 3, 15)
