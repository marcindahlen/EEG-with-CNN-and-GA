import tensorflow as tf
import numpy

from layers.ilayer import ILayer


class SimpleLayer(ILayer):
    def __init__(self, in_shape, out_shape):
        self.output = None
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.weights = []

    def forward_pass(self, input):
        pass

    def get_all_weights(self, weights):
        pass

    def set_all_weights(self):
        pass

    def decomposed_weights(self):
        pass

    def rebuild_weights(self, flat_weights):
        pass

    def init_weights(self, kernels: int) -> list:
        pass
