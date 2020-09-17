from functools import reduce
from typing import Tuple

import tensorflow as tf
import numpy

from layers.available_layers import Layer
from layers.ilayer import ILayer


class Convolution(ILayer):
    def __init__(self, kernels_out: int, kernels_in: int, dimensions: Tuple, filter_len: int):
        print("Convolution::init input - " + str(kernels_out) + ", " + str(kernels_in) + ", " + str(dimensions) + ", " + str(filter_len))
        self.output = None
        self.dimensions = dimensions
        self.weights = self.init_weights(kernels_out, kernels_in, dimensions, filter_len)
        print("Convolution::init weights - " + str(numpy.shape(self.weights)))
        self.type = Layer.convolution
        self.weight_length = len(self.decomposed_weights())

    def forward_pass(self, input: numpy.ndarray) -> numpy.ndarray:
        input_shape = numpy.shape(input)
        print("Convolution::forward_pass input_shape - " + str(input_shape))
        if len(input_shape) == 4:
            self.output = tf.nn.conv2d(
                input, self.weights, strides=5, padding='SAME', data_format='NHWC', dilations=None, name=None
            )
        elif len(input_shape) == 5:
            self.output = tf.nn.conv3d(
                input, self.weights, strides=[1, 5, 5, 5, 1], padding='SAME', data_format='NDHWC', dilations=None, name=None
            )
        else:
            raise Exception("Invalid input shape in convolution layer. Input: " + str(len(input_shape)) +
                            ". Weights: " + str(len(self.weights)))

        return self.output

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
        shape = numpy.shape(self.weights)
        flat_length = reduce(lambda x, y: x * y, shape)
        return tf.reshape(self.weights, flat_length)

    def rebuild_weights(self, flat_weights) -> numpy.ndarray:
        return tf.reshape(flat_weights, self.dimensions)

    def init_weights(self, kernels_out: int, kernels_in: int, dimensions: Tuple, filter_len: int) -> numpy.ndarray:
        if len(dimensions) == 4:  # first conv layer
            # [filter_height, filter_width, in_channels, out_channels]
            print("Convolution::init_weights input_shape - [filter_height, filter_width, in_channels, out_channels]")
            print("Convolution::init_weights input_shape - " + str(1) + ", " + str(filter_len) + ", " + str(1) + ", " + str(kernels_out))
            return numpy.random.normal(loc=0, scale=0.25, size=(1, filter_len, 1, kernels_out))
        elif len(dimensions) == 5:  # nth conv layer or layer in "herded" data
            # [filter_depth, filter_height, filter_width, in_channels, out_channels]
            print("Convolution::init_weights input_shape - [filter_depth, filter_height, filter_width, in_channels, out_channels]")
            print("Convolution::init_weights input_shape - " + str(filter_len) + ", " + str(filter_len) + ", " + str(filter_len) + ", " + str(kernels_in) + ", " + str(kernels_out))
            return numpy.random.normal(loc=0, scale=0.32, size=(filter_len, filter_len, filter_len, kernels_in,
                                                                kernels_out))
        else:
            raise Exception("Invalid input shape in convolution layer::init_weights. Dimensions: " +
                            str(len(dimensions)) + ". Expected: 4 or 5.")
