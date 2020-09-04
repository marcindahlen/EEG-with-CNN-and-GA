from functools import reduce
from typing import Tuple

import tensorflow as tf
import numpy

from layers.ilayer import ILayer


class LSTMLayer(ILayer):
    def __init__(self, in_shape, out_shape):
        self.output = None
        self.previous_outputs = None
        self.memories = None
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.weights = []

    def forward_pass(self, input) -> numpy.ndarray:
        input_gates = None
        memory_gates = None
        forget_gates = None
        output_gates = None

    def get_all_weights(self) -> numpy.ndarray:
        pass

    def set_all_weights(self, new_weights):
        pass

    def decomposed_weights(self):
        pass

    def rebuild_weights(self, flat_weights):
        pass

    def init_weights(self, in_shape: Tuple, size: int) -> numpy.ndarray:
        self.in_shape = reduce(lambda x, y: x * y, in_shape)
        size = (size, 4, self.in_shape + 1)   # (no_of_neurons, no_of_gates, no_of_weights + bias)
        return numpy.random.normal(loc=0, scale=0.25, size=size)