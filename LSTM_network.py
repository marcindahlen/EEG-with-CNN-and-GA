import math
from variables import Variables
from LSTM_neuron import LstmNeuron

# @TODO konstruktor może, ale nie musi przyjmować zapamiętane wagi

# @TODO parser zapamiętanych wag z/do pliku

# @TODO potrzebna funkcja do zapisu stanu wag


class NeuralNetwork(object):

    def __init__(self):
        self.generationNo = 0
        self.score = math.nan #TODO czy może math.inf ?
        """matrix = [[1, 2], [3,4], [5,6], [7,8]]
            transpose = [[row[i] for row in matrix] for i in range(2)]
            [[1, 3, 5, 7], [2, 4, 6, 8]]"""
        self.topology = [[LstmNeuron for neuron in range(Variables.network_topology[layer])] for layer in Variables.network_topology]
        self.save_state_text(Variables.network_dnas_path)

    def forward_pass(self, alpha_wave_data):
        """Given prepared and normalized data
        as a pandas series it performs single
        pass over all fragments (windows) of data
        after each giving some output, but only last
        one is regarded as important one and is
        further evaluated by other methods.
        Given data may be of length non divisible by
        window size and in such case n first values
        are omitted.

        :return a value in <0, 1> which correspond
                to interpreted classification cases
        """
        self.generationNo += 1

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
