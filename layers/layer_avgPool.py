import tensorflow as tf
import numpy


class AvgPool(object):
    def __init__(self):
        self.output = 0

    def forward_pass(self, input):
        """"""
        if input.shape is (100000,):
            self.output = tf.nn.avg_pool1d(input, [5], strides=None, data_format='SAME')
            return self.output
        elif input.shape is (5, 20000):
            self.output = numpy.ndarray(input.shape, dtype=float)
            for x in input:
                self.output[x] = tf.nn.avg_pool1d(input[x], [5], strides=None, data_format='SAME')
        elif input.shape is (5, 5, 4000):
            self.output = numpy.ndarray(input.shape, dtype=float)
            for x in input:
                for y in x:
                    self.output[x][y] = tf.nn.avg_pool1d(input[x][y], [5], strides=None, data_format='SAME')
        elif input.shape is (5, 5, 5, 800):
            self.output = numpy.ndarray(input.shape, dtype=float)
            for x in input:
                for y in x:
                    for z in y:
                        self.output[x][y][z] = tf.nn.avg_pool1d(input[x][y][z], [5], strides=None, data_format='SAME')
        else:
            raise Exception("Error in AvgPool: not supported input shape!")

