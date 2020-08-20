import math
import random
from utils import variables
from networks.neuron_LSTM import NumpyNeuron
import numpy

# @TODO konstruktor może, ale nie musi przyjmować zapamiętane wagi

# @TODO parser zapamiętanych wag z/do pliku

# @TODO potrzebna funkcja do zapisu stanu wag

# @TODO czy neurony pierwszej wartswy powinny widzieć cały zakres danych, czy konwolucja? czy Synopsa?? MOST IMPORTANT!!


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
                self.topology = [[NumpyNeuron(receptive_field if layer == 0 else variables.network_topology[layer - 1]) for neuron in range(
                    variables.network_topology[layer])] for layer in range(len(variables.network_topology))]
            else:
                self.topology = existing_topology
        if from_existing_data:
            pass

    def forward_pass(self, alpha_wave_data, iterations_no):
        """
        Given prepared and normalized data
        as a dictionary of numpy arrays it performs single
        pass over all fragments (windows) of data
        after each (window) giving some output, TODO ...???
        TODO The new approach after reading some papers should be to do everything (on data preparation/fourier level)
        TODO to read all relevant data (channel data) at once, without windows and sweeping!

        @TODO what about forgetting pre-setup window size and instead make the window parametrised to fit data in 1:100 ratio (i.e.)?
        :param iterations_no:
        :param alpha_wave_data: a dictionary with filenames as keys containing dictionaries with channel numbers as keys
                (and channel numpy array data as values)
                i.e. data = b.input_examined['P08.txt'][16] → [0.45233266 0.37322515 0.22718053 ... 0.21095335 0.32860041 0.32860041]
        :return void
        """

        self.cycles += 1
        self.answer = dict()        # erase previous values

        for a in alpha_wave_data:
            for i in range(iterations_no-1):            # @TODO iterations should be around 100? for now, around 21
                self.answer[a] = []                     # erase previous answer, only last one (after all iterations) matters

                # 1. prepare "the question"
                extension = []
                self.question = []
                for channel in variables.channels_to_consider:
                    extension = numpy.append(extension, alpha_wave_data[a][channel][i * variables.window_base_length: (i + 1) * variables.window_base_length])
                self.question = numpy.append(self.question, extension)

                # 2. loop over neurons in layers
                outputs = [[] for k in range(len(self.topology))]
                for k, layer in enumerate(self.topology):
                    for neuron in layer:
                        outputs[k] = numpy.append(outputs[k], neuron.calculate((self.question if k == 0 else outputs[k-1]).astype(dtype=numpy.float32)))

                self.answer[a] = outputs[-1]

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
        the_sum = dict()

        for key in target:              # @TODO correct answer is of much importance, should i increase its impact by an order of magnitude (or lower impact of mistake)?
            the_sum[key] = sum([math.pow(f - o, 2) for f, o in zip(self.answer[key], target[key])]) / len(target[key])
            the_sum[key] = math.sqrt(the_sum[key])

        self.score = sum(the_sum.values()) / len(the_sum)

        return self.score

    def get_score(self):
        """
        :return float <0, 1>
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
        → https://www.numpy.org/devdocs/user/basics.creation.html
        :return void
        """
        to_save = []
        for layer in self.topology:
            for neuron in layer:
                to_save = numpy.append(to_save, neuron.get_weights_vectorised())
        to_save.tofile(file_path + self.get_id())

    def load_state_binary(self, file_name):
        """
        → https://www.numpy.org/devdocs/user/basics.creation.html
        → https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.fromfile.html
        :return void"""
        to_load = numpy.fromfile(file_name, dtype=numpy.float32)
        for layer in self.topology:
            for neuron in layer:
                read = numpy.array(to_load[0:neuron.get_size()], dtype=numpy.float32)
                to_load = to_load[neuron.get_size():]
                neuron.set_weights_from_vectorized(read)

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
                    for weight in neuron.get_weights_vectorised():
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

        I expect to have about 50 populations
        of networks one after another.
        Each network have about 103000 weights or more.

        :return void"""

        def change_weights(weights: list) -> list:
            amount = int(len(weights) / variables.population_quantity)
            change_indexes = numpy.random.random_integers(len(weights), size=(amount,))         # @TODO to be parametrised as learning rate!!
            for index in change_indexes:
                weights[index] = 2 * numpy.random.random_sample() - 1
            return weights

        for layer in self.topology:
            for neuron in layer:
                neuron.set_weights_from_vectorized(change_weights(neuron.get_weights_vectorised))

    def create_single_child(self, other_network):
        """
        Given a particular other network
        both of them have their neurons
        crossbreed producing a new network with different
        state. After that, new network is mutated
        to introduce new quality.

        :return network
        """
        network_length = 0
        for layer in self.topology:
            network_length += len(layer)

        intersection = int(random.gauss(network_length / 2, network_length / 6))

        left_topology = self.flatten_topology(self.topology)
        left_topology = left_topology[:intersection]
        right_topology = self.flatten_topology(other_network.topology)
        right_topology = right_topology[intersection:]

        new_topology: list = left_topology.extend(right_topology)
        new_topology = self.rebuild_topology(new_topology)

        return NeuralNetwork(self.examination_no, new_topology).mutate()

    def multiplication_by_budding(self):
        """
        This method produces new network,
        by mutating this one. Original network
        doesn't change.

        :return network
        """

        return NeuralNetwork(self.examination_no, self.topology).mutate()

    def flatten_topology(self, topology):
        """

        :return:
        """
        flat_topology = []
        for layer in topology:
            for neuron in layer:
                flat_topology.append(neuron)

        return flat_topology

    def rebuild_topology(self, flat: list):
        """

        :param flat:
        :return:
        """
        print(len(flat))
        flat.reverse()

        new_topology = [[flat.pop() for neuron in layer] for layer in self.topology]

        return new_topology
