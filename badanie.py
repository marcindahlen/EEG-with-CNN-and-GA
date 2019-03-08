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
import math
import variables
import pandas
import numpy
import os
import os.path


class Badanie(object):

    def __init__(self, examination_no):                                             # examination_no is the number of column in target data
        self.input_examined = dict()
        self.output_examined = dict()
        self.files_list = [name for name in os.listdir(variables.in_raw_path)]      # @TODO would be good to do NOT load same data every time
        self.files_no = len(self.files_list)
        self.prepare_input()
        self.prepare_target(examination_no)
        self.minmax_tuple = ()

    def prepare_input(self):
        """
        From raw csv select proper channels, @TODO proper network could read all channels simultaneously?
        proper are channels 1 - 8 and 13 and 14, since i was told other channels contain too much noise
        trim useless frequencies,
        standardise data.

        :return void
        """
        for file in self.files_list:
            temporary_mem_channels = dict()
            self.input_examined[file] = numpy.genfromtxt(variables.in_raw_path + file, delimiter=',')
            self.input_examined[file] = numpy.delete(self.input_examined[file], variables.how_many_to_drop, axis=None)
            channel_size = self.count_channel_size(self.input_examined[file])
            channels_no = numpy.floor(len(self.input_examined[file]) / channel_size).astype(int)
            print(channels_no)
            for channel in range(0, channels_no):
                temporary_mem_channels[channel] = self.input_examined[file][channel * channel_size : (channel + 1) * channel_size]
            self.input_examined[file] = temporary_mem_channels
        # at this point i have a dictionary with filenames as keys containing dictionaries with channel numbers as keys (and channel numpy array data as values)

        for examined_keys, examined_vals in self.input_examined.items():
            for channel_keys, channel_vals in examined_vals.items():
                for i in range(len(channel_vals)-1):
                    channel_vals[i] = channel_vals[i+1] - channel_vals[i]
                    channel_vals[len(channel_vals)-1] = 0
            for channel_key, channel_value in examined_vals.items():
                slice_mean = numpy.mean(channel_value)
                slice_dev = numpy.std(channel_value)
                for i in range(len(channel_value)):
                    if numpy.absolute(channel_value[i] - slice_mean) > 4 * slice_dev:
                        channel_value[i] = 0
        # at this point data is standarised around 0, with outsider values deleted

        for examined_keys, examined_vals in self.input_examined.items():
            minimum = math.inf
            maximum = 0
            for channel_keys, channel_vals in examined_vals.items():
                minimum = min(channel_vals) if min(channel_vals) < minimum else minimum
                maximum = max(channel_vals) if max(channel_vals) > maximum else maximum
            for channel_keys, channel_vals in examined_vals.items():
                for i in range(len(channel_vals)):
                    channel_vals[i] = (channel_vals[i] - minimum) / (maximum - minimum)
        # at this point data is normalised in <0, 1>                       @TODO or maybe i should normalise in <-1, 1> and introduce (-1, 1) to initialized weights in neurons ??

        # @TODO low- and highpass filters + fourier 8-12Hz

    def prepare_target(self, examination_no):
        """
        From excel file with columns:
        badany,	SPP,	SPH,	RPN,	Raven_A,	Raven_B,	Raven_C,	Raven_D,	Raven_E,	Raven_WO,	IVE_Impulsywnosc,	IVE_Ryzyko,	IVE_Empatia,	SSZ,	SSE,	SSU,	ACZ,	PKT
        read data,

        → https://www.mantidproject.org/Working_With_Functions:_Return_Values
        :return void
        """
        target_data = pandas.read_excel(variables.out_raw_filepath)
        target_data = target_data.iloc[2:, :]                         # delete P01BA and P01BB rows
        target_data = target_data.iloc[:, examination_no + 1].values

        print(target_data)
        minimum = min(target_data)
        maximum = max(target_data)
        self.minmax_tuple = (minimum, maximum)
        for i in range(len(target_data)):
            target_data[i] = (target_data[i] - minimum) / (maximum - minimum)

        for index, file in enumerate(self.files_list):
            self.output_examined[file] = [self.decide_no_belonging(target_data[index], x) for x in range(10)]

        # at this point i have a dictionary with filenames as keys containing a list as value.
        # The list contain ten values → 0 or 1, where 1 means the original value was in corresponding range
        # i.e. 39 becomes 0.18 when minmaxed* in <32, 71> → which becomes [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        #   *look "at this point data is normalised in <0, 1>" above or → https://en.wikipedia.org/wiki/Feature_scaling

    def count_channel_size(self, eeg):
        """
        For each eeg recording there are 17 channels of streamed data,
        all stored in single file channel after channel.
        This method counts length of a single channel
        in a particular file, so channels could be extracted.
        I assume channels are separated by outlier data points.
        @TODO probably wrong assumption that channels are equal in length
        @TODO and there are 16 channels not 17 - first 'item' is noise
        :return int
        """
        i = len(eeg)
        data_slice = eeg[i - 100000:i]      # last hundred thousand points from eeg, assumed they're always from last channel
        slice_mean = numpy.mean(data_slice)
        slice_dev = numpy.std(data_slice)
        flag_rised = True
        while flag_rised:
            i -= 1
            if numpy.absolute(eeg[i] - slice_mean) > 2 * 3 * slice_dev:     # find first 'impossible' value - impossible because: → https://en.wikipedia.org/wiki/Standard_score
                flag_rised = False
                return len(eeg) - i

    def decide_no_belonging(self, number, index):
        """
        Given minmaxed value of examined's test score
        and currently considered index in list being build,
        outputs 1 or 0 where 1 is a valid match
        :param number:
        :param index:
        :return: int: 1 or 0
        """
        index = index / 10
        return 1 if index <= number < index + 0.1 else 0

    def interprete_prediction(self, prediction):
        """
        Given list with 10 elements - 9 zeros and 1 one,
        this method converts information from the list
        to the form as in raw input (reverse process from prepare_target()
        :return: int
        """
        memory = 0
        for i, x in enumerate(prediction):
            memory += x * (i + 1) / 10
        memory = memory * (self.minmax_tuple[1] - self.minmax_tuple[0]) + self.minmax_tuple[0]
        return memory

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
