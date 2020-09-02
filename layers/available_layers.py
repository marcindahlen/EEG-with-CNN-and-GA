from enum import Enum


class Layer(Enum):
    LSTM = 1
    basic_neuron = 2
    convolution = 3
    AvgPool = 4
    MaxPool = 5
