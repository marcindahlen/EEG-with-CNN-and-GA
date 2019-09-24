"""
Class created to initialize source data only once,
despite the number of examinations.

There is an assumption that data is given in one EEG channel
per file, at least 8 files for one person.
"""

from utils import variables
import math
import numpy
import pandas
import os
import os.path


class Datastorage(object):

    def __init__(self):
        self.input_examined = dict()            # main holder of data, nested dictionary: input_examined[examined_no][channel][datapoints]
        self.output_examined = dict()
        self.files_list = [name for name in os.listdir(variables.in_raw_channels_path)]
        self.files_no = len(self.files_list)
        self.examined_no = self.files_no / variables.channels_for_person
        print('   Znaleziono ' + str(self.files_no) + ' plików.')
        self.isDataInitialised = False
        self.channels_stats = dict()            # proper description in method: prepare_inputdata_insights()

    def prepare_input(self):
        """
        From raw csvs read channels,
        proper are channels 1 - 8 and 13 and 14, since i was told other channels contain too much noise.
        Trim useless frequencies (i was told to be interested only in alpha waves, that is 8-12Hz),
        standardise data.

        :return void
        """
        print('   Wczytywanie kanałów', end='... ')
        self.load_channels()
        print('   zakończone.')
        # at this point i have a dictionary with filenames as keys containing dictionaries with channel numbers as keys (and channel numpy array data as values)

        # @TODO data trimming to lowest length (drop initial varying n points for every channel)  <- unnecessary here if later trimmed by pure alpha wave length

        print('   Filtrowanie fal alfa', end='... ')
        self.data_fourier_transform()
        print('   zakończone.')

        # print("   Standaryzacja danych", end="...")
        # self.standardise_channel_data()
        # print("zakończona.")
        # at this point data is standardised around 0, with outsider values deleted

        print("   Normalizacja danych", end="...")
        self.normalise_channel_data()
        print("zakończona.")
        # at this point data is normalised in <0, 1>
        # @TODO or maybe i should normalise in <-1, 1> and introduce (-1, 1) to initialized weights in neurons ??

        self.prepare_inputdata_insights()

        self.isDataInitialised = True

    def load_channels(self):
        """
        The data is in form of nested dictionaries: input_examined[examined_no][channel][datapoints]
                                                    input_examined[0 → 33][1 → 10][0 → 100k++]
        TODO consider only those files included in variables.channels_to_consider

        :return: void
        """
        for index, file in enumerate(self.files_list):
            examined_person = math.floor((index + 1) / variables.channels_for_person)
            current_channel_no = index - examined_person * variables.channels_for_person + 1
            loaded_data = numpy.genfromtxt(variables.in_raw_path + file, delimiter=',', dtype=numpy.float32)
            data_to_save = numpy.delete(loaded_data, variables.how_many_to_drop, axis=None)
            self.input_examined[examined_person] = {}
            self.input_examined[examined_person][current_channel_no] = data_to_save

    def data_apply_filters(self):
        """
        TODO for time being i don't know the parameters for high- and low pass filters
        :return:
        """
        pass

    def data_fourier_transform(self):
        """
        Alpha waves have frequency between 8Hz and 12Hz
        → https://en.wikipedia.org/wiki/Alpha_wave
        This method use Fast Fourier Transform to convert
        EEG data info to it's frequency info, trim information
        concerning non-alpha wave frequencies, and convert
        frequency info back to wave datapoints.
        → https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/

        :return:
        """
        lower_data_limes = variables.alpha_low_frequency / variables.frequencyOfData
        higher_data_limes = variables.alpha_high_frequency / variables.frequencyOfData

        for examined_keys, examined_vals in self.input_examined.items():
            for channel_keys, channel_vals in examined_vals.items():
                channel_vals = numpy.fft.fft(channel_vals)
                channel_length = len(channel_vals)
                for i in range(channel_length):
                    channel_vals[i] = channel_vals[i] if math.floor(lower_data_limes) * channel_length < i < math.ceil(higher_data_limes) * channel_length else 0
                self.input_examined[examined_keys][channel_keys] = numpy.fft.ifft(channel_vals)     # TODO trim unnecessary parts (zeros from line above)

    def standardise_channel_data(self):
        """
        I need to carefully consider the need of standardisation and a way to proper justify it

        :return: void
        """
        # for examined_keys, examined_vals in self.input_examined.items():
        #     for channel_keys, channel_vals in examined_vals.items():
        #         for i in range(len(channel_vals)-1):
        #             channel_vals[i] = channel_vals[i+1] - channel_vals[i]
        #             channel_vals[-1] = 0      # TODO i need justification for this kind of standardisation

        # for examined_keys, examined_vals in self.input_examined.items():
        #     for channel_key, channel_value in examined_vals.items():          # @TODO it takes an eternity to process, needs optimization
        #         slice_mean = numpy.mean(channel_value)
        #         slice_dev = numpy.std(channel_value)
        #         for i in range(len(channel_value)):
        #             # if numpy.absolute(channel_value[i] - slice_mean) > 4 * slice_dev:
        #             # channel_value[i] = 0
        #             if channel_value[i] - slice_mean > 4 * slice_dev:
        #                 channel_value[i] = slice_mean + 3 * slice_dev
        #             elif channel_value[i] - slice_mean < -4 * slice_dev:
        #                 channel_value[i] = slice_mean - 3 * slice_dev         # TODO ok, but why? Outliers might be crucial

    def normalise_channel_data(self):
        """
        Min-max feature scaling (each single channel separately)
        → https://en.wikipedia.org/wiki/Normalization_(statistics)

        :return: void
        """
        for examined_keys, examined_vals in self.input_examined.items():
            for channel_keys, channel_vals in examined_vals.items():
                minimum = min(channel_vals)
                maximum = max(channel_vals)
                for i in range(len(channel_vals)):
                    channel_vals[i] = ((channel_vals[i] - minimum) / (maximum - minimum)).astype(dtype=numpy.float32)
                self.input_examined[examined_keys][channel_keys] = channel_vals

    def prepare_inputdata_insights(self):
        """
        Method meant to give some insights about input data.
        I calculates channels averages with deviations.
        Those informations are meaningless for the research,
        but useful to imagine how things look like.

        After this method completes, variable channels_stats
        contain nested dictionary. Each dictionary for particular channel
        contains name key and value: average length (float),
        standard deviation of length (float), average value (float),
        standard deviation of value (float), minimal and maximal value
        for this channel across examined ppl (tuple of floats),
        average minimal and average maximal for this channel (tuple)

        :return:
        """

    def print_inputdata_insights(self):
        """
        
        :return:
        """
        pass

    def assume_networkIterationsNo(self):
        """

        :return:
        """
        return int(self.minmax_channelLength_tuple[0] / variables.network_input_window)          # TODO could be better

    def show_summary(self):
        """

        :return:
        """
        if not self.isDataInitialised:
            print("No data loaded.")
        else:
            summary = dict()
            for key in self.input_examined:
                summary[key] = dict()
                print()
                for channel_key in self.input_examined[key]:
                    summary[key][channel_key] = len(self.input_examined[key][channel_key])
                    print(key + ' ' + str(channel_key) + ': ' + str(summary[key][channel_key]), end="; ")
        print()
        print("networkIterationsNo: " + str(self.assume_networkIterationsNo()))

    def prepare_target(self, examination_no):
        """
        From excel file with columns:
        badany,	SPP,	SPH,	RPN,	Raven_A,	Raven_B,	Raven_C,	Raven_D,	Raven_E,	Raven_WO,	IVE_Impulsywnosc,	IVE_Ryzyko,	IVE_Empatia,	SSZ,	SSE,	SSU,	ACZ,	PKT
        read data.

        → https://www.mantidproject.org/Working_With_Functions:_Return_Values
        :return void
        """
        print('   Wczytywanie danych wzorcowych', end='...')
        target_data = pandas.read_excel(variables.out_raw_filepath)
        target_data = target_data.iloc[2:, :]                         # delete P01BA and P01BB rows
        target_data = target_data.iloc[:, examination_no + 1].values

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
        print(' zakończone.')

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
