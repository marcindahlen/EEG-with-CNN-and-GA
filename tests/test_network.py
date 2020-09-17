import numpy

from layers.available_layers import Layer
from networks.network import Network


class TestNetwork:
    def test_net(self):
        data_1d = numpy.random.rand(150)
        data_2d = numpy.random.rand(150, 150)
        data_3d = numpy.random.rand(150, 150, 150)

        # Pooling layer at the beginning
        layers = [Layer.AvgPool, Layer.convolution, Layer.basic_neuron, Layer.basic_neuron]
        layers_1d_shape = [(150, 30),
                           (30, (6, 6)),
                           ((6, 6), 6),
                           (6, 1)]
        layers_2d_shape = [((150, 150), (30, 150)),  # -> (2x6, 30)
                           ((12, 30), (24, 6)),
                           ((24, 6), 6),
                           (6, 1)]
        layers_3d_shape = [((150, 150, 150), (30, 30, 150)),  # -> (3x6, 30)
                           ((30, 30, 150), (36, 30)),
                           ((36, 30), 6),
                           (6, 1)]

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
        output = network.forward_pass(data_2d)
        print(output)
        assert numpy.shape(output) == (1,)


        # Convolution layer have to work without preceding pooling layer
        layers = [Layer.convolution, Layer.convolution, Layer.basic_neuron, Layer.basic_neuron]
        layers_1d_shape = [(150, (6, 30)),
                           ((6, 30), (12, 6)),
                           ((12, 6), 6),
                           (6, 1)]
        layers_2d_shape = [((150, 150), (12, 30)),   # -> (2x6, 30)
                           ((12, 30), (24, 6)),
                           ((24, 6), 6),
                           (6, 1)]
        layers_3d_shape = [((150, 150, 150), (18, 30)),   # -> (3x6, 30)
                           ((18, 30), (36, 30)),
                           ((36, 30), 6),
                           (6, 1)]

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
        output = network.forward_pass(data_2d)
        print(output)
        assert numpy.shape(output) == (1,)
