import utility
import random
import numpy


class NumpyNeuron(object):
    """
    Class defines a single neuron of long-short term memory type.
    → https://cdn-images-1.medium.com/max/1250/1*laH0_xXEkFE0lKJu54gkFQ.png

    Using numpy arrays, since basic LSTM_neuron caused over 73 days
    of calculations on 4 i7 cores.
    """

    def __init__(self, window, from_existing_data=False, weights_data=[]):
        self.y_prev = 0
        self.suma_in, self.suma_out, self.suma_mem, self.suma_forget = 0, 0, 0, 0
        self.y_in, self.y_forget, self.state, self.y_out = 0, 0, 0, 0
        self.bias = 1
        self.mem = 0
        self.output = 0
        if not from_existing_data:
            self.weights = 0.407 * numpy.random.randn(4, window + 2) + 0.5              # +2 from weights for bias and previous value; normal distribution mu=0.5 sigma=0.166
            numpy.append(self.weights[0], numpy.random.rand())                          # append weight for previous memory state
            numpy.append(self.weights[1], numpy.random.rand())                          # append weight for previous memory state
        if from_existing_data:
            if not weights_data:
                raise Exception('No weights data passed to the neuron constructor!')
            elif len(weights_data) != 4:
                raise Exception('List passed to the neuron constructor has wrong dimensions!')
            else:
                self.weights = weights_data

    def get_size(self):
        """
        Returns number of all weights of this neuron.
        :return: int
        """
        size = 0
        for gate in self.weights:
            size += len(gate)
        return size

    def get_weights(self):
        """
        Method outputs neuron's weights in form of numpy array.
        :return: numpy array
        """
        return self.weights

    def get_weights_structured(self):
        """
        Returns neuron's weights in the same order
        as method set_weights() would read them.
        :return: list of four lists
        """

    def set_weights(self, weights_data=[]):
        """
        Weights should be given as nested list:
        in a list there should be four lists, one for each gate,
        each list's last element should be bias weight,
        and the last list's last but one float should be previous output's weight.
        :param weights_data:
        :return: void
        """

    def calculate(self, input=[]):
        """
        Executes a single forward pass on a neuron.
        """
        input.append(self.y_prev)
        input.append(self.bias)
        input = numpy.array(input)

        self.suma_in = numpy.multiply(input, self.weights[0])
        self.suma_in = numpy.sum(self.suma_in, dtype=numpy.float32)
        self.y_in = utility.sigmoid(self.suma_in)

        self.suma_forget = numpy.multiply(input, self.weights[1])
        self.suma_forget = numpy.sum(self.suma_forget, dtype=numpy.float32)
        self.y_forget = utility.sigmoid(self.suma_forget)

        self.suma_forget = numpy.multiply(input, self.weights[2])
        self.suma_forget = numpy.sum(self.suma_forget, dtype=numpy.float32)
        self.y_forget = utility.sigmoid(self.suma_forget)

        self.suma_mem = 0
        for i in range(len(input)):
            self.suma_mem += input[i] * self.weights[2][i]
        self.suma_mem += self.bias_weights[2] * self.bias_mem
        self.suma_mem += self.y_prev * self.waga_prev
        self.mem = self.y_forget * self.state + self.y_in * utility.tanh(self.suma_mem)

        self.suma_out = 0
        for i in range(len(input)):
            self.suma_out += input[i] * self.weights[3][i]
        self.suma_out += self.bias_weights[3] * self.bias_out
        self.y_out = utility.sigmoid(self.suma_out)

        self.output = utility.tanh(self.mem) * self.y_out

        return self.output