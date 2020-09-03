import tensorflow as tf
import numpy


class AvgPool(object):
    def __init__(self):
        self.input_shape = None
        self.output = None

    def forward_pass(self, input):
        """"""
        self.input_shape = numpy.shape(input)

        if self.input_shape is 1:
            pass
        else:
            pass