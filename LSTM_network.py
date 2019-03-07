import math
import variables
from LSTM_neuron import LstmNeuron

# @TODO konstruktor może, ale nie musi przyjmować zapamiętane wagi

# @TODO parser zapamiętanych wag z/do pliku

# @TODO potrzebna funkcja do zapisu stanu wag


class NeuralNetwork(object):
    """
    Class defines a simple network consisting of LSTM neurons grouped
    in layers as defined in variables.py
    I assume network can be trained to classify one's eeg data
    to a single scale's values group in psychology test.
    What i mean by value group is a group made by sticking
    together order of possible outcomes. If a scale have 100
    possible values and particular outcome is 37, i want network to
    classify this 37 to the group 4th of ten possible.
    """

    def __init__(self, from_existing_data=False):
        self.generationNo = 0
        self.score = math.nan #TODO czy może math.inf ?
        self.answer = []
        """
        Nested lists cheatsheet:
        matrix = [[1, 2], [3,4], [5,6], [7,8]]
        transpose = [[row[i] for row in matrix] for i in range(2)]
        [[1, 3, 5, 7], [2, 4, 6, 8]]
        """
        if not from_existing_data:
            self.topology = [[LstmNeuron(variables.network_input_window if layer-1 <= 0 else variables.network_topology[layer-1]) for neuron in range(variables.network_topology[layer])] for layer in variables.network_topology]
        if from_existing_data:
            pass

    def forward_pass(self, alpha_wave_data):
        """
        Given prepared and normalized data
        as a dictionary of numpy arrays it performs single
        pass over all fragments (windows) of data
        after each (window) giving some output, but only last
        one is regarded as important one and is
        further evaluated by other methods.
        Given data may be of length non divisible by
        window size and in such case n first values
        are omitted.

        :return a list of ten values, each value in <0, 1>
                the highest one at particular neuron,
                which resembles the class data is classified to.
        """
        self.generationNo += 1

    def adjust_data(self, alpha_wave_data):
        """
        @TODO for now it can't work this way since i have a dictionary as input, not list
        :param alpha_wave_data:
        :return:
        """
        while alpha_wave_data % variables.network_input_window != 0:
            del alpha_wave_data[0]
        return alpha_wave_data

    def evaluate_self(self, alpha_wave_data, target):
        """
        Given a dictionary which keys are filenames P01.txt - Pxx.txt
        and values are next channels of a single eeg recording,
        method calculates network's output classification.
        :param alpha_wave_data:
        :param target:
        :return:
        """
        pass

    def save_state_binary(self, file_path):
        """Saves wages values of all neurons
        in a binary file.
        First, there's need to make a single
        list from all of the wages of different neuron gates.
        → https://stackoverflow.com/questions/807863/how-to-output-list-of-floats-to-a-binary-file-in-python

        :return void
        """
        pass

    def load_state_binary(self, file_path):
        """Loads all neurons wages to
        a single list. From this list neurons' wages
        are set to new values. Topology must be matching.

        :return void"""
        pass

    def save_state_text(self, file_path):
        """Saves wages values of all neurons
        in a text file.

        :return void
        """
        pass

    def load_state_text(self, file_path):
        """"""
        pass

    def mutate(self):
        """With a [[given probability]]
        do or do not change a value of
        a particular wage - going once
        for all wages for all neurons
        in this network.

        :return void"""
        pass

    def create_single_child(self, other_network):
        """Given a particular other network
        both of them have their neurons' wages
        crossbreed producing a new network with different
        state. After that, new network is mutated
        to introduce new quality.
        @TODO czy proces krzyżowania też powinien charakteryzować się losowością?

        :return network
        """
        pass

    def multiplication_by_budding(self):
        """This method produces new network,
        by mutating this one. Original network
        doesn't change.

        :return network
        """
        pass

    def set_score(self, new_score):
        """Overrides the score with a new one."""
        self.score = new_score

    def get_id(self):
        """:return integer - generated ID based
            on generation number"""
