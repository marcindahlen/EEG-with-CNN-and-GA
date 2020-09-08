from functools import reduce
from typing import Tuple

import tensorflow as tf
import numpy

from layers.available_layers import Layer
from layers.ilayer import ILayer
from utils.utility import sigmoid, tanh


class LSTMLayer(ILayer):
    def __init__(self, in_shape: Tuple, size: int):
        self.outputs = None
        self.previous_outputs = numpy.zeros(size)
        self.memories = numpy.zeros(size)
        self.in_shape = in_shape
        self.size = size
        self.weights = self.init_weights(in_shape, size)
        self.type = Layer.LSTM

    def forward_pass(self, input) -> numpy.ndarray:
        input = tf.reshape(input, self.in_shape)
        input = numpy.append(input, [1])  # append bias
        input = numpy.append(input, [self.previous_outputs])

        input_gates = [numpy.matmul(input, self.weights[w][0]) for w in range(self.size)]
        input_gates = [numpy.sum(e) for e in input_gates]
        input_gates = [sigmoid(s) for s in input_gates]
        forget_gates = [numpy.matmul(input, self.weights[w][1]) for w in range(self.size)]
        forget_gates = [numpy.sum(e) for e in forget_gates]
        forget_gates = [sigmoid(s) for s in forget_gates]
        memory_gates = [numpy.matmul(input, self.weights[w][2]) for w in range(self.size)]
        memory_gates = [numpy.sum(e) for e in memory_gates]
        memory_gates = [numpy.delete(e, -1) for e in memory_gates]  # previous output not relevant in this gate
        memory_gates = [tanh(n) for n in memory_gates]

        self.memories = [forget_gates[cell] * self.memories[cell] +
                         input_gates[cell] * memory_gates[cell] for cell in range(self.size)]

        output_gates = [numpy.matmul(input, self.weights[w][3]) for w in range(self.size)]
        output_gates = [numpy.sum(e) for e in output_gates]
        output_gates = [numpy.delete(e, -1) for e in output_gates]  # previous output not relevant in this gate
        output_gates = [sigmoid(s) for s in output_gates]

        self.outputs = [tanh(self.memories[cell]) * output_gates[cell] for cell in range(self.size)]
        self.previous_outputs = self.outputs
        return self.outputs

    def get_all_weights(self) -> numpy.ndarray:
        return self.weights

    def set_all_weights(self, new_weights):
        if numpy.shape(new_weights) == numpy.shape(self.weights):
            self.weights = new_weights
        elif len(numpy.shape(new_weights)) == 1:
            self.weights = self.rebuild_weights(new_weights)

    def decomposed_weights(self):
        flat_length = reduce(lambda x, y: x * y, self.weights)
        return tf.reshape(self.weights, flat_length)

    def rebuild_weights(self, flat_weights):
        return tf.reshape(flat_weights, numpy.shape(self.weights))

    def init_weights(self, in_shape: Tuple, size: int) -> numpy.ndarray:
        self.in_shape = reduce(lambda x, y: x * y, in_shape)
        # size = (no_of_neurons, no_of_gates, no_of_weights + bias + previous_outputs)
        size = (size, 4, self.in_shape + 1 + 1 * size)
        return numpy.random.normal(loc=0, scale=0.25, size=size)
