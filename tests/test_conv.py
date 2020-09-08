import numpy
import tensorflow

from layers.available_layers import Layer
from layers.layer_convolution import Convolution


class TestConvLayer:
    def test_conv(self):
        data_1d = numpy.random.rand(15)
        data_1d = tensorflow.reshape(data_1d, (1, 1, 15, 1))        # batch_shape + [height, width, channels]
        data_3d = numpy.random.rand(15, 15, 15)
        data_3d = tensorflow.reshape(data_3d, (1, 15, 15, 15, 1))     # [batch, in_depth, in_height, in_width, in_channels]

        print(data_1d)
        print(data_3d)

        conv_layer_in1d = Convolution(6, 1, (1, 1, 15, 1), 5)
        conv_layer_in3d = Convolution(6, 1, (1, 5, 5, 5, 1), 5)

        print("")
        print("")
        output = conv_layer_in1d.forward_pass(data_1d)
        print(output)
        assert numpy.shape(output) == (1, 1, 3, 6)
        print("")
        print("")
        output = conv_layer_in3d.forward_pass(data_3d)
        print(output)

        assert numpy.shape(output) == (1, 3, 3, 3, 6)
