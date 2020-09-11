import tensorflow
import numpy

from layers.available_layers import Layer
from layers.ilayer import ILayer


class MaxPool(ILayer):
    def __init__(self):
        self.output = None
        self.dimensions = None
        self.type = Layer.MaxPool

    def forward_pass(self, input):
        self.dimensions = len(numpy.shape(input))

        if self.dimensions == 1:
            self.output = input[None][:, :, None]
            self.output = tensorflow.nn.max_pool1d(self.output, 5, 5, padding='SAME')
        elif self.dimensions == 2:
            self.output = input[None][:, :, None]
            self.output = tensorflow.nn.max_pool2d(self.output, ksize=[5, 5], strides=[5, 5], padding='SAME')
        elif self.dimensions == 3:
            self.output = input[None][:, :, None]
            self.output = tensorflow.nn.max_pool3d(self.output, [5, 5, 5], [5, 5, 5], padding='SAME')
        elif self.dimensions == 5:
            self.output = tensorflow.nn.max_pool3d(self.output, [5, 5, 5], [5, 5, 5], padding='SAME')
        else:
            raise Exception("Invalid input shape in AvgPool layer: " + str(self.dimensions))

        return self.output
