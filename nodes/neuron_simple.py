from utils import utility
import numpy
from networks import node

class NumpyNeuron(node.Node):
    """
    Class defines a single classic neuron.
    → https://en.wikipedia.org/wiki/Artificial_neuron#Basic_structure

    """

    def __init__(self, window, from_existing_data=False, weights_data=[]):
        """

        :param window: int
        :param from_existing_data: bool
        :param weights_data:
        """
        self.suma_in = 0
        self.bias = 1
        self.output = 0
        self.sqrt_std_dev = 0.5477225575
        if not from_existing_data:                                                      # @TODO normal distribution might be not the best choice
            self.weights = []
            self.weights = numpy.random.beta(0.5, 0.5, window + 1)

        if from_existing_data:
            if not weights_data:
                raise Exception('No weights data passed to the neuron constructor!')
            else:
                self.weights = weights_data

    def calculate(self, input=[]):
        """
        Executes a single forward pass on a neuron.
        """
        input = numpy.append(input, self.bias)

        self.suma_in = numpy.multiply(input, self.weights)
        self.suma_in = numpy.sum(self.suma_in, dtype=numpy.float32)
        self.output = utility.sigmoid(self.suma_in)

        return self.output
