import numpy

from layers.available_layers import Layer
from layers.layer_LSTMneurons import LSTMLayer
from layers.layer_avgPool import AvgPool
from layers.layer_convolution import Convolution
from layers.layer_maxPool import MaxPool
from layers.layer_simpleNeurons import SimpleLayer
from networks.inetwork import INetwork
from utils import variables
from utils.utility import rmse, get_time
from utils.variables import mutation_probability_factor, multi_crossovers_limit

FILTER_LEN = 5


class Network(INetwork):
    def __init__(self, layer_types, ins_outs_shapes):
        self.layer_types = layer_types
        self.ins_outs_shapes = ins_outs_shapes
        self.structure = ins_outs_shapes
        self.layers = self.initialize(layer_types, ins_outs_shapes)
        self.output = None
        self.score = None
        self.weight_lengths_by_layer = self.get_weight_lengths_by_layer()

    def forward_pass(self, input):
        self.output = input
        for layer in self.layers:
            if layer.type == Layer.convolution:
                if len(numpy.shape(self.output)) == 1:
                    print("Network::forward_pass - " + str(layer.dimensions))
                    self.output = numpy.reshape(self.output, [1, 1, numpy.shape(self.output)[0], 1])
                elif len(numpy.shape(self.output)) == 2:
                    print("Network::forward_pass - " + str(layer.dimensions))
                    self.output = numpy.reshape(self.output, [1, 1, numpy.shape(self.output)[0], numpy.shape(self.output)[1], 1])
                elif len(numpy.shape(self.output)) == 4:
                    print("Network::forward_pass - " + str(layer.dimensions))
                else:
                    raise Exception("Network::forward_pass: wrong input shape - input: " +
                                    str(numpy.shape(self.output)) + " in layer " + str(layer.type))
                self.output = layer.forward_pass(self.output)
            else:
                self.output = layer.forward_pass(self.output)

        return self.output

    def initialize(self, layer_types, ins_outs_shapes):
        layers = []
        for i, layer in enumerate(layer_types):
            shape_in = ins_outs_shapes[i][0]
            shape_out = ins_outs_shapes[i][1]
            if layer == Layer.MaxPool:
                layers.append(MaxPool())
            if layer == Layer.AvgPool:
                layers.append(AvgPool())
            if layer == Layer.convolution:
                # (kernels_out: int, kernels_in: int, dimensions: Tuple, filter_len: int)
                kernels_in = shape_in[0] if type(shape_in) is not int else 1
                kernels_out = shape_out[0]
                dimensions = (1, FILTER_LEN, 1, kernels_out) if type(shape_in) is int else (FILTER_LEN, FILTER_LEN,
                                                                                            kernels_in, kernels_out)
                # above dimension are only cases for 1D and 2D input
                conv = Convolution(kernels_out, kernels_in, dimensions, FILTER_LEN)
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
            self.score = rmse(self.output, target)
        else:
            self.score = rmse([self.output], [target])

    def get_score(self) -> float:
        return self.score

    def get_weight_lengths_by_layer(self) -> list:
        weight_lengths = []
        for layer in self.layers:
            new_pair = (layer.type, layer.weight_length)
            weight_lengths.append(new_pair)
        return weight_lengths

    def set_weights(self, weights):
        for i, layer in self.layers:
            if i == 0:
                new_weights = weights[0:self.weight_lengths_by_layer[i][1]]
                layer.set_all_weights(new_weights)
            else:
                new_weights = weights[self.weight_lengths_by_layer[i - 1][1]:self.weight_lengths_by_layer[i][1]]
                layer.set_all_weights(new_weights)

    def get_weights(self) -> list:
        weights = []
        for layer in self.layers:
            weights.append(layer.decomposed_weights())
        return weights

    def save_weights(self):
        weights = self.get_weights()
        file = open(variables.net_memory_path + "_" + str(get_time()) + "_" + str(self.score) + ".csv")
        for val in weights:
            file.write(str(val))
        file.close()

    def load_weights(self, filename):
        # TODO
        pass

    def mutate(self, probability: float):
        probability = probability * mutation_probability_factor
        new_weights = []
        for w in self.get_weights():
            p = numpy.random.random() < probability
            if p:
                new_weights.append(numpy.random.normal(loc=0, scale=0.32))
            else:
                new_weights.append(w)
        self.set_weights(new_weights)

    def create_single_child(self, other: INetwork, mutate_prob) -> INetwork:
        mutate_prob = mutate_prob * mutation_probability_factor
        length = 0
        for key, val in self.weight_lengths_by_layer:
            length += val
        this_weights = self.get_weights()
        other_weights = other.get_weights()
        breakpoints = numpy.random.randint(0, length, size=multi_crossovers_limit)
        new_weights = []
        flag = True
        for i in range(length):
            if i in breakpoints:
                flag = not flag
            if flag:
                new_weights.append(this_weights[i])
            else:
                new_weights.append(other_weights[i])
        new_network = Network(self.layer_types, self.ins_outs_shapes)  # Yes, it's not efficient I know </3
        new_network.set_weights(new_weights)
        new_network.mutate(mutate_prob)

        return new_network
