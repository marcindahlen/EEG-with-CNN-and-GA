import numpy

from layers.available_layers import Layer
from layers.layer_LSTMneurons import LSTMLayer


class TestLSTMLayer:
    def test_layer(self):
        data_1d = numpy.random.rand(15)
        data_2d = numpy.random.rand(15, 15)

        print(data_1d)
        print(data_2d)

        LSTM_1d_layer = LSTMLayer((15,), 4)
        LSTM_2d_layer = LSTMLayer((15, 15), 4)

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
