import numpy
import tensorflow

from layers.available_layers import Layer
from layers.layer_convolution import Convolution


class TestConvLayer:
    def test_conv(self):
        data_1d = numpy.random.rand(15)
        data_1d = tensorflow.reshape(data_1d, (15, 1))
        data_2d = numpy.random.rand(15, 15, 6)
        data_2d = tensorflow.reshape(data_2d, (15, 15, 6))
        data_3d = numpy.random.rand(15, 15, 15)
        data_3d = tensorflow.reshape(data_3d, (15, 15, 15, 1))

        print(numpy.shape(data_1d))
        print(numpy.shape(data_2d))
        print(numpy.shape(data_3d))

        conv_layer_in1d = Convolution((15, 1), (3, 3), 5)
        conv_layer_in2d = Convolution((15, 15, 6), (15, 3, 12), 5)
        conv_layer_in3d = Convolution((15, 15, 15, 1), (15, 3, 3, 1), 5)

        print("")
        print("")
        output = conv_layer_in1d.forward_pass(data_1d)
        print(output)
        assert numpy.shape(output) == (3, 3)

        print("")
        print("")
        output = conv_layer_in2d.forward_pass(data_2d)
        print(output)
        assert numpy.shape(output) == (15, 3, 12)

        print("")
        print("")
        output = conv_layer_in3d.forward_pass(data_3d)
        print(output)
        assert numpy.shape(output) == (15, 3, 3, 1)
