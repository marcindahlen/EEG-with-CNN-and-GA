import math
import random
from utils import variables
from networks.neuron_LSTM import NumpyNeuron
import numpy


class ConvNetwork(object):
    """
    → https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    """

    def __init__(self, examination_no, receptive_field=variables.network_input_window, existing_topology=[],
                 from_existing_data=False):
        self.examination_no = examination_no  # the number of the psychological test's column
        self.generationNo = 0
        self.cycles = 0
        self.score = 1  # RMSE → https://www.statisticshowto.datasciencecentral.com/rmse/
        self.answer = dict()  # for each filename as key, store list of answers as lists of length 10
        self.question = []

        if not from_existing_data:
            if not existing_topology:
                self.topology = [
                    [NumpyNeuron(receptive_field if layer == 0 else variables.network_topology[layer - 1]) for
                     neuron in range(
                        variables.network_topology[layer])] for layer in range(len(variables.network_topology))]
            else:
                self.topology = existing_topology
        if from_existing_data:
            pass

    def forward_pass(self, alpha_wave_data, iterations_no):
        """

        :return void
        """
        pass

    def evaluate_self(self, target):  # Probably the most important method of them all!!
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

        for key in target:  # @TODO correct answer is of much importance, should i increase its impact by an order of magnitude (or lower impact of mistake)?
            the_sum[key] = sum([math.pow(f - o, 2) for f, o in zip(self.answer[key], target[key])]) / len(
                target[key])
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
            change_indexes = numpy.random.random_integers(len(weights), size=(
            amount,))  # @TODO to be parametrised as learning rate!!
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