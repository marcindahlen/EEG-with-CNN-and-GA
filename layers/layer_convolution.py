import tensorflow as tf
import numpy


class Convolution(object):
    def __init__(self):
        self.output = 0
        self.weights_shape = []
        self.weights = numpy.ndarray(self.weights_shape, dtype=float)

    def forward_pass(self, input):
        if input.shape is (20000,):
            self.output = tf.keras.layers.Conv1D()  # TODO
            return self.output
        elif input.shape is (5, 4000):
            self.output = numpy.ndarray(input.shape, dtype=float)
            for x in input:
                self.output[x] = tf.keras.layers.Conv1D()  # TODO
        elif input.shape is (5, 5, 800):
            self.output = numpy.ndarray(input.shape, dtype=float)
            for x in input:
                for y in x:
                    self.output[x][y] = tf.keras.layers.Conv1D()  # TODO
        else:
            raise Exception("Error in AvgPool: not supported input shape!")

    def get_all_weights(self):
        pass

    def set_all_weights(self):
        pass

