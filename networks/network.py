import numpy

from layers.available_layers import Layer
from layers.layer_LSTMneurons import LSTMLayer
from layers.layer_avgPool import AvgPool
from layers.layer_convolution import Convolution
from layers.layer_maxPool import MaxPool
from layers.layer_simpleNeurons import SimpleLayer
from networks.inetwork import INetwork


class Network(INetwork):
    def __init__(self, layer_types, ins_outs_shapes):
        self.structure = ins_outs_shapes
        self.layers = self.initialize(layer_types, ins_outs_shapes)
        self.output = None
        self.score = None

    def forward_pass(self, input):
        self.output = input
        for layer in self.layers:
            if layer.type == Layer.convolution:
                if len(numpy.shape(input)) == 1:
                    pass
                else:
                    pass
            else:
                self.output = layer.forward_pass(self.output)

    def initialize(self, layer_types, ins_outs_shapes):
        layers = []
        for i, layer in enumerate(layer_types):
            shape_in = ins_outs_shapes[i][0]
            shape_out = ins_outs_shapes[i][1]
            if layer == Layer.MaxPool:
                layers.append(MaxPool())
            if layer == Layer.AvgPool:
                layers.append(AvgPool())
            if layer == Layer.Convolution:
                # (kernels_out: int, kernels_in: int, dimensions: Tuple, filter_len: int)
                kernels_in = shape_in[0] if len(shape_in) != 1 else 1
                kernels_out = shape_out[0]
                filter_len = 5
                dimensions = (1, filter_len, 1, kernels_out) if len(shape_in) == 1 else (filter_len, filter_len,
                                                                                         filter_len, kernels_in,
                                                                                         kernels_out)
                conv = Convolution(kernels_out, kernels_in, dimensions, filter_len)
                layers.append(conv)
            if layer == Layer.basic_neuron:
                # (in_shape: Tuple, size: int)
                neuron = SimpleLayer(shape_in, shape_out)
                layers.append(neuron)
            if layer == Layer.LSTM:
                # (in_shape: Tuple, size: int)
                neuron = LSTMLayer(shape_in, shape_out)
                layers.append(neuron)

        return layers

    def evaluate_self(self, target):
        if isinstance(target, list) or isinstance(target, numpy.ndarray):
            pass
        else:
            pass

    def get_score(self):
        pass

    def get_id(self):
        pass

    def save_state_binary(self):
        pass

    def load_state_binary(self):
        pass

    def save_state_text(self):
        pass

    def load_state_text(self):
        pass

    def mutate(self):
        pass

    def create_single_child(self):
        pass

    def multiplication_by_budding(self):
        pass

    def decompose_weights(self):
        pass

    def rebuild_weights(self, flat_weights):
        pass
