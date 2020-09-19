from functools import reduce
from typing import Tuple

import tensorflow as tf
import numpy

from layers.available_layers import Layer
from layers.ilayer import ILayer


class Convolution(ILayer):
    def __init__(self, in_shape: Tuple, out_shape: Tuple, filter_len: int):
        if len(in_shape) != len(out_shape):
            raise Exception("Convolution::init - input shape and output shape doesn't match!")
        self.output = None
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.weights = self.init_weights(in_shape, out_shape, filter_len)
        self.type = Layer.convolution
        self.weight_length = len(self.decomposed_weights())

    def forward_pass(self, input: numpy.ndarray) -> numpy.ndarray:
        input_shape = numpy.shape(input)
        input = input[None][:, :, None]
        if len(input_shape) == 2:
            self.output = tf.nn.conv2d(
                input, self.weights, strides=5, padding='SAME', data_format='NHWC', dilations=None, name=None
            )
        elif len(input_shape) == 3:
            self.output = tf.nn.conv2d(
                input, self.weights, strides=5, padding='SAME', data_format='NHWC', dilations=None, name=None
            )
        elif len(input_shape) == 4:
            self.output = tf.nn.conv3d(
                input, self.weights, strides=[1, 5, 5, 5, 1], padding='SAME', data_format='NDHWC', dilations=None,
                name=None
            )
        else:
            raise Exception("Invalid input shape in convolution layer. Input: " + str(len(input_shape)) +
                            ". Weights: " + str(len(self.weights)))

        return self.output

    def get_all_weights(self) -> numpy.ndarray:
        return self.weights

    def set_all_weights(self, new_weights):
        if numpy.shape(new_weights) == self.out_shape:
            self.weights = new_weights
        elif len(numpy.shape(new_weights)) == 1:
            self.weights = self.rebuild_weights(new_weights)
        else:
            raise Exception("Invalid new_weights shape in convolution layer: " + str(numpy.shape(new_weights)) +
                            " expected: " + str(self.out_shape))

    def decomposed_weights(self) -> numpy.ndarray:
        shape = numpy.shape(self.weights)
        flat_length = reduce(lambda x, y: x * y, shape)
        return tf.reshape(self.weights, flat_length)

    def rebuild_weights(self, flat_weights) -> numpy.ndarray:
        return tf.reshape(flat_weights, self.weights)

    def init_weights(self, in_shape: Tuple, out_shape: Tuple, filter_len: int) -> numpy.ndarray:
        if len(out_shape) == 2:  # 1-D input plus kernels
            # [filter_height, filter_width, in_channels, out_channels]
            return numpy.random.normal(loc=0, scale=0.25, size=(1, filter_len, in_shape[1], out_shape[1]))
        elif len(out_shape) == 3:  # 2-D input plus kernels
            # [filter_height, filter_width, in_channels, out_channels]
            return numpy.random.normal(loc=0, scale=0.25, size=(filter_len, filter_len, in_shape[2], out_shape[2]))
        elif len(out_shape) == 4:  # 3-D input plus kernels
            # [filter_height, filter_width, in_channels, out_channels]
            return numpy.random.normal(loc=0, scale=0.25, size=(filter_len, filter_len, filter_len, in_shape[3],
                                                                out_shape[3]))
        else:
            raise Exception("Invalid input shape in convolution layer::init_weights. Dimensions: " +
                            str(len(in_shape)) + ". Expected: 2 to 5.")
