from networks.simple_neuron import NumpyNeuron
import numpy
import pytest

class TestData:

    pytest.data_0 = [numpy.random.random() for x in range(100)]
    pytest.data_1 = [numpy.random.random() for x in range(100)]

    def test_simple_neurons(self):
        neuron_A = NumpyNeuron(100)
        neuron_B = NumpyNeuron(100)

        output_A = neuron_A.calculate(pytest.data_0)
        output_B = neuron_B.calculate(pytest.data_1)
        
        assert output_A != output_B
