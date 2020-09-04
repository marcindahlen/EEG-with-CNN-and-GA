import numpy

from layers.layer_simpleNeurons import SimpleLayer


class TestSNLayer:
    def test_layer(self):
        data_1d = numpy.random.rand(15)
        data_2d = numpy.random.rand(15, 15)

        print(data_1d)
        print(data_2d)

        simple_1d_layer = SimpleLayer((15,), 4)
        simple_2d_layer = SimpleLayer((15, 15), 4)

        print("")
        print("")
        output = simple_1d_layer.forward_pass(data_1d)
        print(output)
        assert numpy.shape(output) == (4, )
        print("")
        print("")
        output = simple_2d_layer.forward_pass(data_2d)
        print(output)

        assert numpy.shape(output) == (4, )
