from functools import reduce
from typing import Tuple

import tensorflow as tf
import numpy

from layers.ilayer import ILayer
from utils.utility import sigmoid


class SimpleLayer(ILayer):
    def __init__(self, in_shape: Tuple, size: int):
        self.output = None
        self.in_shape = None
        self.size = size
        self.weights = self.init_weights(in_shape, size)
        self.biases_weights = self.init_weights((1,), size)

    def forward_pass(self, input) -> numpy.ndarray:
        self.output = tf.reshape(input, self.in_shape)
        self.output = numpy.append(self.output, [1])
        self.output = [numpy.matmul(self.output, w) for w in range(self.size)]
        self.output = numpy.sum(self.output)
        self.output = sigmoid(self.output)

        return self.output

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
        return numpy.random.rand(size, self.in_shape + 1)       # + 1 bias weight
