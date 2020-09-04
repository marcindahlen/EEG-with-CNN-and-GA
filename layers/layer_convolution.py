from functools import reduce

import tensorflow as tf
import numpy

from layers.ilayer import ILayer


class Convolution(ILayer):
    def __init__(self, kernels_out: int, kernels_in: int, dimensions: list, filter_len: int):
        self.output = None
        self.dimensions = dimensions
        self.weights = self.init_weights(kernels_out, kernels_in, dimensions, filter_len)

    def forward_pass(self, input):
        self.dimensions = numpy.shape(input)
        if len(self.dimensions) == 3:
            self.output = tf.nn.conv2d(
                input, self.weights, 5, padding='SAME', data_format='NWC', dilations=None, name=None
            )
        if len(self.dimensions) == 5 or len(self.dimensions) == 12:
            self.output = tf.nn.conv3d(
                input, self.weights, 5, padding='SAME', data_format='NDHWC', dilations=None, name=None
            )
        else:
            raise Exception("Invalid input shape in convolution layer: " + str(len(self.dimensions)))

    def get_all_weights(self) -> numpy.ndarray:
        return self.weights

    def set_all_weights(self, new_weights):
        if numpy.shape(new_weights) == self.dimensions:
            self.weights = new_weights
        elif len(numpy.shape(new_weights)) == 1:
            self.weights = self.rebuild_weights(new_weights)
        else:
            raise Exception("Invalid new_weights shape in convolution layer: " + str(numpy.shape(new_weights)) +
                            " expected: " + str(self.dimensions))

    def decomposed_weights(self) -> numpy.ndarray:
        flat_length = reduce(lambda x, y: x * y, self.dimensions)
        return tf.reshape(self.weights, flat_length)

    def rebuild_weights(self, flat_weights) -> numpy.ndarray:
        return tf.reshape(flat_weights, self.dimensions)

    def init_weights(self, kernels_out: int, kernels_in: int, dimensions: list, filter_len: int) -> numpy.ndarray:
        if len(dimensions) == 3:  # first conv layer
            return numpy.random.rand(1, filter_len, 1, kernels_out)
        elif len(dimensions) == 5:  # nth conv layer or layer in "herded" data
            return numpy.random.rand(filter_len, filter_len, filter_len, kernels_in, kernels_out)
        elif len(dimensions) == 12:  # layer in multilayer eeg data
            return numpy.random.rand(filter_len, filter_len, filter_len, kernels_in, kernels_out)
