import utility
import random


class NumpyNeuron(object):
    """
    Class defines a single neuron of long-short term memory type.
    → https://cdn-images-1.medium.com/max/1250/1*laH0_xXEkFE0lKJu54gkFQ.png

    Using numpy arrays, since basic LSTM_neuron caused over 73 days
    of calculations on 4 i7 cores.
    """

    def __init__(self, window, from_existing_data=False, weights_data=[]):
        pass

    def get_size(self):
        """
        Returns number of all weights of this neuron.
        :return: int
        """

    def get_weights(self):
        """
        Method outputs neuron's weights in form of lists.
        There is some commotion in saving bias weights and previous value weight.
        :return: a list of four lists
        """

    def get_weights_structured(self):
        """
        Returns neuron's weights in the same order
        as method set_weights() would read them.
        :return: list of four lists
        """

    def set_weights(self, weights_data=[]):
        """
        Weights should be given as nested list:
        in a list there should be four lists, one for each gate,
        each list's last element should be bias weight,
        and the last list's last but one float should be previous output's weight.
        :param weights_data:
        :return: void
        """

    def calculate(self, input=[]):
        """
        Executes a single forward pass on a neuron.
        """

    def learn(self, target, learning_lambda, input=[]):
        """
        Executes a single learning pass on a neuron.
        In case of supervised learning.
        (that means target data is known)
        """