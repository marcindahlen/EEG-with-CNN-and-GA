import numpy

from layers.available_layers import Layer


class TestNetwork:
    def test_net(self):
        data_1d = numpy.random.rand(150)
        data_2d = numpy.random.rand(150, 150)

        layers = [Layer.convolution, Layer.convolution, Layer.basic_neuron, Layer.basic_neuron]
        layers_1d_shape = [(150, (6, 30)),
                           ((6, 30), (12, 6)),
                           ((12, 6), 6),
                           (6, 1)]

        print(data_1d)
        print(data_2d)

        print("")
        print("")
        output = LSTM_1d_layer.forward_pass(data_1d)
        print(output)
        assert numpy.shape(output) == (4,)
        print("")
        print("")
        output = LSTM_2d_layer.forward_pass(data_2d)
        print(output)

        assert numpy.shape(output) == (4,)
