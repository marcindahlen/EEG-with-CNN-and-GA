from utils import utility
import numpy


class NumpyNeuron(object):
    def __init__(self, window, from_existing_data=False, weights_data=[]):
        self.y_prev = 0
        self.suma_in, self.suma_out, self.suma_mem, self.suma_forget = 0, 0, 0, 0
        self.y_in, self.y_forget, self.state, self.y_out = 0, 0, 0, 0
        self.bias = 1
        self.mem = 0
        self.output = 0
        self.mean = 0
        self.sqrt_std_dev = 0.5477225575
        if not from_existing_data:                                                      # @TODO normal distribution might be not the best choice
            self.weights = [[] for i in range(4)]                                       # repetition to avoid ValueError: could not broadcast input array from shape
            self.weights[0] = numpy.random.beta(0.5, 0.5, window + 2)                   # +2 from weights for bias and previous output; normal distribution mu=0.5 sigma=0.166
            self.weights[1] = numpy.random.beta(0.5, 0.5, window + 2)
            self.weights[2] = numpy.random.beta(0.5, 0.5, window + 1)                   # +1 from weights for bias; normal distribution mu=0.5 sigma=0.166
            self.weights[3] = numpy.random.beta(0.5, 0.5, window + 1)

        if from_existing_data:
            if not weights_data:
                raise Exception('No weights data passed to the neuron constructor!')
            elif len(weights_data) != 4:
                raise Exception('List passed to the neuron constructor has wrong dimensions!')
            else:
                self.weights = weights_data

    def get_size(self):
        size = 0
        for gate in self.weights:
            size += len(gate)
        return size

    def get_weights(self):
        return self.weights

    def get_weights_vectorised(self):
        output = numpy.array(self.weights[0], dtype=numpy.float32)
        output = numpy.append(output, self.weights[1])
        output = numpy.append(output, self.weights[2])
        output = numpy.append(output, self.weights[3])

        return output

    def set_weights_from_vectorized(self,  weights_data=[]):
        for i in range(self.weights):
            for j in range(self.weights[i]):
                self.weights[i][j] = weights_data[0]
                weights_data = weights_data[1:]

    def set_weights(self, weights_data=[]):
        if not weights_data:
            raise Exception('No weights data passed to the neuron constructor!')
        elif len(weights_data) != 4:
            raise Exception('List passed to the neuron constructor has wrong dimensions!')
        else:
            self.weights = weights_data

    def calculate(self, input=[]):
        input = numpy.append(input, self.bias)
        input = numpy.append(input, self.y_prev)

        self.suma_in = numpy.multiply(input, self.weights[0])
        self.suma_in = numpy.sum(self.suma_in, dtype=numpy.float32)
        self.y_in = utility.sigmoid(self.suma_in)

        self.suma_forget = numpy.multiply(input, self.weights[1])
        self.suma_forget = numpy.sum(self.suma_forget, dtype=numpy.float32)
        self.y_forget = utility.sigmoid(self.suma_forget)

        input = input[:-1]              # drop y_prev

        self.suma_mem = numpy.multiply(input, self.weights[2])
        self.suma_mem = numpy.sum(self.suma_mem, dtype=numpy.float32)
        self.mem = self.y_forget * self.mem + self.y_in * utility.tanh(self.suma_mem)

        self.suma_out = numpy.multiply(input, self.weights[3])
        self.suma_out = numpy.sum(self.suma_out, dtype=numpy.float32)
        self.y_out = utility.sigmoid(self.suma_out)

        self.output = utility.tanh(self.mem) * self.y_out

        self.y_prev = self.output

        return self.output
