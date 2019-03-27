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
        """

        :param window: int
        :param from_existing_data: bool
        :param weights_data: numpy two dimensional array:
                4 vectors of lengths: [window + 2], [window + 2], [window + 1], [window + 1]
        """
        self.y_prev = 0
        self.suma_in, self.suma_out, self.suma_mem, self.suma_forget = 0, 0, 0, 0
        self.y_in, self.y_forget, self.state, self.y_out = 0, 0, 0, 0
        self.bias = 1
        self.mem = 0
        self.output = 0
        if not from_existing_data:
            self.weights = 0.407 * numpy.random.randn(4, window + 1) + 0.5              # +1 from weights for bias; normal distribution mu=0.5 sigma=0.166
            numpy.append(self.weights[0], numpy.random.rand())                          # append weight for previous output value
            numpy.append(self.weights[1], numpy.random.rand())                          # append weight for previous output value
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
        :return: numpy 2-dimensional array
        """
        return self.weights

    def get_weights_vectorised(self):
        """

        :return: array (vector) of all weights
        """
        output = numpy.array(self.weights[0], dtype=numpy.float32)
        output = numpy.append(output, self.weights[1])
        output = numpy.append(output, self.weights[2])
        output = numpy.append(output, self.weights[3])

        return output


    def set_weights_from_vectorized(self,  weights_data=[]):
        """
        Given array (vector) it saved as numpy 2-dimensional array
        :return: void
        """
        for i in range(self.weights):
            for j in range(self.weights[i]):
                self.weights[i][j] = weights_data[0]
                weights_data = weights_data[1:]

    def set_weights(self, weights_data=[]):
        """
        Weights should be given as nested list:
        in a list there should be four lists, one for each gate,
        each list's last element should be bias weight,
        and the last list's last but one float should be previous output's weight.
        :param weights_data: numpy two dimensional array:
                4 vectors of lengths: [window + 2], [window + 2], [window + 1], [window + 1]
        :return: void
        """
        if not weights_data:
            raise Exception('No weights data passed to the neuron constructor!')
        elif len(weights_data) != 4:
            raise Exception('List passed to the neuron constructor has wrong dimensions!')
        else:
            self.weights = weights_data

    def calculate(self, input=[]):
        """
        Executes a single forward pass on a neuron.
        """
        input.append(self.bias)
        input.append(self.y_prev)
        input = numpy.array(input)

        self.suma_in = numpy.multiply(input, self.weights[0])
        self.suma_in = numpy.sum(self.suma_in, dtype=numpy.float32)
        self.y_in = utility.sigmoid(self.suma_in).astype(dtype=numpy.float32)

        self.suma_forget = numpy.multiply(input, self.weights[1])
        self.suma_forget = numpy.sum(self.suma_forget, dtype=numpy.float32)
        self.y_forget = utility.sigmoid(self.suma_forget).astype(dtype=numpy.float32)

        input = input[:-1]              # drop y_prev

        self.suma_mem = numpy.multiply(input, self.weights[2])
        self.suma_mem = numpy.sum(self.suma_mem, dtype=numpy.float32)
        self.mem = self.y_forget * self.mem + self.y_in * utility.tanh(self.suma_mem).astype(dtype=numpy.float32)

        self.suma_out = numpy.multiply(input, self.weights[3])
        self.suma_out = numpy.sum(self.suma_out, dtype=numpy.float32)
        self.y_out = utility.sigmoid(self.suma_out).astype(dtype=numpy.float32)

        self.output = utility.tanh(self.mem) * self.y_out

        self.y_prev = self.output.astype(dtype=numpy.float32)

        return self.output
