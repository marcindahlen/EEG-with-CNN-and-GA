from utils import utility
import numpy


class NumpyNeuron(object):
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

    def get_size(self):
        """
        Returns number of all weights of this neuron.
        :return: int
        """
        return len(self.weights)

    def get_weights(self):
        """
        Method outputs neuron's weights in form of numpy array.
        :return: numpy 2-dimensional array
        """
        return self.weights

    def get_weights_vectorised(self):
        """

        :return: array (vector) of all weights
        """
        return numpy.array(self.weights, dtype=numpy.float32)


    def set_weights_from_vectorized(self,  weights_data=[]):
        """
        """
        pass

    def set_weights(self, weights_data=[]):
        """

        :param weights_data:
        :return: void
        """
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
