import math
import random
import variables
from LSTM_neuron import LstmNeuron
from array import array

# @TODO konstruktor może, ale nie musi przyjmować zapamiętane wagi

# @TODO parser zapamiętanych wag z/do pliku

# @TODO potrzebna funkcja do zapisu stanu wag


class NeuralNetwork(object):
    """
    Class defines a simple network consisting of numpy-LSTM neurons grouped
    in layers as defined in variables.py
    I assume network can be trained to classify one's eeg data
    to a single scale's values group in psychology test.
    What i mean by value group is a group made by sticking
    together order of possible outcomes. If a scale have 100
    possible values and particular outcome is 37, i want network to
    classify this 37 to the group 4th of ten possible.
    """

    def __init__(self, examination_no, receptive_field = variables.network_input_window, existing_topology = [], from_existing_data = False):
        self.examination_no = examination_no    # the number of the psychological test's column
        self.generationNo = 0
        self.cycles = 0
        self.score = 1              # RMSE → https://www.statisticshowto.datasciencecentral.com/rmse/
        self.answer = dict()        # for each filename as key, store list of answers as lists of length 10
        self.question = []

        if not from_existing_data:
            if not existing_topology:
                self.topology = [[LstmNeuron(receptive_field if layer == 0 else variables.network_topology[layer-1]) for neuron in range(variables.network_topology[layer])] for layer in range(len(variables.network_topology))]
            else:
                self.topology = existing_topology
        if from_existing_data:
            pass

    def forward_pass(self, alpha_wave_data, iterations_no):
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
        @TODO what about forgetting pre-setup window size and instead make the window parametrised to fit data in 1:100 ratio (i.e.)?
        :param iterations_no:
        :param alpha_wave_data: a dictionary with filenames as keys containing dictionaries with channel numbers as keys
                (and channel numpy array data as values)
                i.e. data = b.input_examined['P08.txt'][16] → [0.45233266 0.37322515 0.22718053 ... 0.21095335 0.32860041 0.32860041]
        :return void
        """

        self.cycles += 1

        # @TODO iterations should be around 100? for now, around 21
        # erase previous answer, only last one (after all iterations) matters
        # 1. prepare "the question"
        # print("in network.calculate: " + str(len(self.question)))
        # 2. feed the neurons in the first layer
        # 3. feed the neurons in the second layer
        # 4. feed the neurons in the last layer and get "the answer"

    def evaluate_self(self, target):                # Probably the most important method of them all!!
        """
        Given the target values in form of dictionary of lists,
        each list consist of ten values: 9 zeros and 1 one,
        and having ready "answer" value in same form,
        where each list consist of ten values in (0, 1);
        method gives RMSE of last forward pass over
        all examined EEG data.

        Perfect RMSE = 0

        Worst case RMSE = 1

        :param target: dictionary of lists
        :return: RMSE
        """

    def get_score(self):
        """
        ":return float <0, 1>
        """
        return self.score

    def get_id(self):
        """
        :return string
        """
        return str(self.examination_no) + '_' + str(self.cycles) + '_' + str(self.score)

    def save_state_binary(self, file_path):
        """
        Saves wages values of all neurons
        in a binary file.
        → https://stackoverflow.com/questions/807863/how-to-output-list-of-floats-to-a-binary-file-in-python

        :return void
        """

    def load_state_binary(self, file_path):
        """

        :return void"""
        pass

    def save_state_text(self, file_path):
        """
        Saves wages values of all neurons
        in a text file.

        :return void
        """
        path = file_path + self.get_id() + '.txt'
        with open(path, "w+") as file:
            for layer in self.topology:
                for neuron in layer:
                    for weight in neuron.get_weights():
                        file.write(weight)

    def load_state_text(self, file_path):
        """

        :param file_path:
        :return: void
        """

        with open(file_path, "r") as file:
            for index, line in enumerate(file):
                pass

    # @TODO All mutation should allow inheritance of features like generationNumber and cycles passed.

    def mutate(self):
        """
        With some probability
        do or do not change a value of
        a particular wage - going once
        for all wages for all neurons
        in this network.

        :return void"""

    def create_single_child(self, other_network):
        """
        Given a particular other network
        both of them have their neurons
        crossbreed producing a new network with different
        state. After that, new network is mutated
        to introduce new quality.

        :return network
        """

    def multiplication_by_budding(self):
        """
        This method produces new network,
        by mutating this one. Original network
        doesn't change.

        :return network
        """

    def flatten_topology(self, topology):
        """

        :return:
        """

    def rebuild_topology(self, flat: 'flattened topology'):
        """

        :param flat:
        :return:
        """
