import numpy
from tensorflow import reshape

from layers.available_layers import Layer
from networks.network import Network


class TestNetwork:
    def test_net(self):
        data_1d = numpy.random.rand(150)
        data_2d = numpy.random.rand(150, 150)
        data_3d = numpy.random.rand(150, 150, 150)

        # convolution layer have to work without preceding pooling layer
        layers = [Layer.convolution, Layer.convolution, Layer.basic_neuron, Layer.basic_neuron]
        layers_1d_shape = [((150, 1), (30, 6)),
                           ((30, 6), (6, 12)),
                           ((6, 12), 12),
                           (12, 1)]
        layers_2d_shape = [((150, 150, 1), (150, 30, 6)),
                           ((150, 30, 6), (150, 6, 12)),
                           ((150, 6, 12), 6),
                           (6, 1)]
        layers_3d_shape = [((150, 150, 150, 1), (150, 30, 30, 4)),
                           ((150, 30, 30, 4), (150, 6, 6, 8)),
                           ((150, 6, 6, 8), 12),
                           (12, 1)]

        data_1d = reshape(data_1d, (150, 1))
        data_2d = reshape(data_2d, (150, 150, 1))
        data_3d = reshape(data_3d, (150, 150, 150, 1))

        print(numpy.shape(data_1d))
        print(numpy.shape(data_2d))
        print(numpy.shape(data_3d))

        print("")
        print("")
        network = Network(layers, layers_1d_shape)
        output = network.forward_pass(data_1d)
        print(output)
        assert numpy.shape(output) == (1,)

        print("")
        print("")
        network = Network(layers, layers_2d_shape)
        output = network.forward_pass(data_2d)
        print(output)
        assert numpy.shape(output) == (1,)

        print("")
        print("")
        network = Network(layers, layers_3d_shape)
        output = network.forward_pass(data_3d)
        print(output)
        assert numpy.shape(output) == (1,)

        # And following with pooling layer at the beginning
        data_1d = reshape(data_1d, (150,))
        data_2d = reshape(data_2d, (150, 150))
        data_3d = reshape(data_3d, (150, 150, 150))

        layers = [Layer.AvgPool, Layer.convolution, Layer.basic_neuron, Layer.basic_neuron]
        layers_1d_shape = [((150, 1), (30, 1)),
                           ((30, 1), (6, 6)),
                           ((6, 6), 6),
                           (6, 1)]
        layers_2d_shape = [((150, 150), (150, 30, 1)),
                           ((150, 30, 1), (150, 6, 8)),
                           ((150, 6, 8), 6),
                           (6, 1)]
        layers_3d_shape = [((150, 150, 150), (150, 30, 30, 1)),
                           ((150, 30, 30, 1), (150, 6, 6, 8)),
                           ((150, 6, 6, 8), 12),
                           (12, 1)]

        print(numpy.shape(data_1d))
        print(numpy.shape(data_2d))
        print(numpy.shape(data_3d))

        print("")
        print("")
        network = Network(layers, layers_1d_shape)
        output = network.forward_pass(data_1d)
        print(output)
        assert numpy.shape(output) == (1,)

        print("")
        print("")
        network = Network(layers, layers_2d_shape)
        output = network.forward_pass(data_2d)
        print(output)
        assert numpy.shape(output) == (1,)

        print("")
        print("")
        network = Network(layers, layers_3d_shape)
        output = network.forward_pass(data_3d)
        print(output)
        assert numpy.shape(output) == (1,)
