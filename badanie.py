"""
The class describes a single examination.

A single examination is a process of unsupervised
training of one LSTM neural network which takes
as input alpha waves recording and predict
the classification to a particular psychology test.
For each feature of each test there should
be separate network to classify examined's EEG.
The total number of features is 17.
Each network is trained by genetic algorithm.
That means, many networks are spawn to select
those with best performance and evolve them
further.

Idealnie, wynikiem pojedynczego badania są:
→   plik graficzny ilustrujący postęp nauki
    kolejnych generacji sieci,
→   plik z przewidywanymi klasyfikacjami grupy testowej
    dokonanymi przez najlepszą sieć,
→   plik (@TODO czy pliki?)
    binarny z zapisem stanu ostatniej generacji sieci.
"""

import variables
import pandas
import numpy
import os
import os.path


class Badanie(object):

    def __init__(self, examination_no):
        self.input_examined = dict()
        self.files_list = [name for name in os.listdir(variables.in_raw_path)]
        self.files_no = len(self.files_list)
        self.prepare_input()
        self.prepare_target(examination_no)

    def prepare_input(self):
        """
        From raw csv select proper channels, @TODO proper network could read all channels simultaneously?
        proper are channels 1 - 8 and 13 and 14, since i was told other channels contain too much noise
        trim useless frequencies,
        standardise data.
        @TODO is normalization needed?
        Out of all 17 channels, only following are proper
        :return void
        """
        for file in self.files_list:
            temporary_mem_channels = dict()
            self.input_examined[file] = numpy.genfromtxt(variables.in_raw_path + file, delimiter=',')
            self.input_examined[file] = numpy.delete(self.input_examined[file], variables.how_many_to_drop, axis=None)
            channel_size = self.count_channel_size(self.input_examined[file])
            channels_no = numpy.floor(self.input_examined[file] / channel_size)
            for channel in range(0, channels_no):
                temporary_mem_channels[channel] = self.input_examined[file][channel * channel_size : (channel + 1) * channel_size]
            self.input_examined[file] = temporary_mem_channels
        #at this point i have a dictionary with filenames as keys and containing dictionaries with channels numbers as keys (and channel numpy array data as values)

    def prepare_target(self, examination_no):
        """
        From excel file with columns:
        badany,	SPP,	SPH,	RPN,	Raven_A,	Raven_B,	Raven_C,	Raven_D,	Raven_E,	Raven_WO,	IVE_Impulsywnosc,	IVE_Ryzyko,	IVE_Empatia,	SSZ,	SSE,	SSU,	ACZ,	PKT
        read data,

        → https://www.mantidproject.org/Working_With_Functions:_Return_Values
        :return void
        """
        target_data = pandas.read_excel(variables.out_raw_filepath)

    def count_channel_size(self, eeg):
        """
        For each eeg recording there are 17 channels of streamed data,
        all stored in single file channel after channel.
        This method counts length of a single channel
        in a particular file, so channels could be extracted.
        I assume channels are separated by outlier data points.
        :return int
        """
        i = len(eeg)
        data_slice = eeg[i - 100000:i]      #last hundred thousand points from eeg, assumed they're always from last channel
        slice_mean = numpy.mean(data_slice)
        slice_dev =  numpy.std(data_slice)
        flag_rised = True
        min = min(eeg)
        while flag_rised:
            i -= 1
            if numpy.absolute(eeg[i] - slice_mean) > 3 * slice_dev:
                flag_rised = False
                return len(eeg) - i

    def initialize_networks(self):
        """
        Create if not exists a new binary files (plural!!)
        → https://stackoverflow.com/questions/807863/how-to-output-list-of-floats-to-a-binary-file-in-python
        describing initial wages for
        LSTM networks to be trained.
        Number of files (population) should
        correspond to number of target classes.
        First check if there is a file,
        if not → initialize networks and save states
        if yes → load states from files
        :return LSTM_network list
        """
        pass

    def single_pass_one_network(self, network):
        """
        Given a single network from the list
        it is run on all examined person's EEG,
        procuring an <0, 1> output for each (EEG).
        Produced list of outputs is compared with
        a list of corresponding target values
        (new list item = target_val - network_output).
        Then squares of new list items are summed.
        The bigger is final number, the worse was
        network's overall performance.
        :return a single float - network's rmse
        """
        pass

    def forward_pass_all_networks(self, networks):
        """
        On given list of networks
        (all networks of same type and size,
        they vary only in weights)
        each network calculates in single pass
        predicted classification of each
        examined's EEG.
        One networks performs as many single-passes
        as there are examined persons.

        Each network has a field to store
        it's mean performance
        measured as RMSE - root-mean-square error.

        :return one list with rmse scores, position
                on this list reference position of network
                and thus order of single-pass
        """
        pass

    def repopulate_network_generation(self, networks):
        """
        @TODO czy sieć nie powinna sama przechowywać swojego score?
        → https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects
        Given all networks were scored based on their
        performance and remember their score,
        the list containing them is sorted
        based on their score, from lowest to highest.
        The worsen [[part]] of networks is deleted and the
        list is repopulated with children of saved networks
        (pair multiplication includes mutation)
        and [[some amount]] of mutated saved networks (budding),
        and [[some amount]] of new random ones
        → https://en.wikipedia.org/wiki/Budding
        → https://unix.stackexchange.com/questions/196251/change-only-one-bit-in-a-file
        → https://www.hackerearth.com/practice/basic-programming/bit-manipulation/basics-of-bit-manipulation/tutorial/
        @TODO a może jednak nie binary a txt ← wtedy mutacja to random na [[jednej]] wadze
        @TODO flipping bits in binary files wymaga eksperymentów

        :return networks
        """
        pass
