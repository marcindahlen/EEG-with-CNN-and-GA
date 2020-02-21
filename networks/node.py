import numpy


class Node(object):
    """
    Basic methods to be shared by neurons, pools and convolution nodes.
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
        self.weights = []

    def calculate(self):
        pass

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
