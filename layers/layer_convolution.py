import tensorflow as tf
import numpy

from layers.ilayer import ILayer


class Convolution(ILayer):
    def __init__(self, kernels, dimensions):
        self.output = None
        self.dimensions = None
        self.weights = self.init_weights(kernels, dimensions)

    def forward_pass(self, input):
        self.dimensions = len(numpy.shape(input))
        if self.dimensions == 3:
            self.output = tf.nn.conv2d(
                input, self.weights, 5, padding='SAME', data_format='NWC', dilations=None, name=None
            )
        if self.dimensions == 5 or self.dimensions == 12:
            self.output = tf.nn.conv3d(
                input, self.weights, 5, padding='SAME', data_format='NDHWC', dilations=None, name=None
            )
        else:
            raise Exception("Invalid input shape in convolution layer: " + str(self.dimensions))

    def get_all_weights(self, weights):
        pass

    def set_all_weights(self):
        pass

    def decompose_weights(self):
        pass

    def rebuild_weights(self, flatline_weights):
        pass

    def init_weights(self, kernels: int, dimensions: int) -> numpy.ndarray:
        if dimensions == 3:                                 # first conv layer
            return numpy.random.rand(1, 20000, 1, kernels)
        elif dimensions == 5:                               # nth conv layer or layer in "herded" data
            return numpy.random.rand(1, 20000, 1, kernels)
        elif dimensions == 12:                              # layer in multilayer eeg data
            return numpy.random.rand(15, 15, 15)

