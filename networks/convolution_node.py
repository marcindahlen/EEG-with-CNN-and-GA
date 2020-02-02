import numpy
from networks import node


class ConvNode(node.Node):
    """
    This differs from a single neuron by not using the activation function
    nor bias.
    → http://brohrer.github.io/how_convolutional_neural_networks_work.html
    → https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
    """

    def __init__(self, window, from_existing_data=False, weights_data=[]):
        """

        :param window: int
        :param from_existing_data: bool
        :param weights_data:
        """
        self.suma_in = 0
        self.output = 0
        self.sqrt_std_dev = 0.5477225575
        if not from_existing_data:                                                      # @TODO normal distribution might be not the best choice
            self.weights = []
            self.weights = numpy.random.uniform(-2, 2, window)

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
        self.output = numpy.sum(self.suma_in, dtype=numpy.float32)

        return self.output