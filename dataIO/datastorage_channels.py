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
        self.output_ranges_x10 = dict()         # output_ranges_x10[examined_no][test_no][10x value 0 or 1]
        self.remember_output_reversal = dict()  # saved mins and maxs to be able to restore values in scale remember_output_reversal[test_no][minmax_tuple]
        self.output_single_x1 = dict()          # output_single_x1[examined_no][test_no]
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
        for channel_keys in self.input_examined[0].keys():
            self.channels_stats[channel_keys]['avg_length'] = 0
            self.channels_stats[channel_keys]['stddev_length'] = []
            self.channels_stats[channel_keys]['mean_value'] = 0
            self.channels_stats[channel_keys]['mean_stddev_value'] = 0
            self.channels_stats[channel_keys]['total_minmax'] = (math.inf, 0)
            self.channels_stats[channel_keys]['mean_minmax'] = (math.inf, 0)

        for examined_keys, examined_vals in self.input_examined.items():
            for channel_keys, channel_vals in examined_vals.items():
                self.channels_stats[channel_keys]['avg_length'] += len(channel_vals)
                self.channels_stats[channel_keys]['stddev_length'].append(len(channel_vals))
                self.channels_stats[channel_keys]['mean_value'] += numpy.mean(channel_vals)
                self.channels_stats[channel_keys]['mean_stddev_value'].append(numpy.std(channel_vals))
                self.channels_stats[channel_keys]['total_minmax'][0] = min(channel_vals) if self.channels_stats[channel_keys]['total_minmax'][0] > min(channel_vals) else self.channels_stats[channel_keys]['total_minmax'][0]
                self.channels_stats[channel_keys]['total_minmax'][1] = max(channel_vals) if self.channels_stats[channel_keys]['total_minmax'][1] < max(channel_vals) else self.channels_stats[channel_keys]['total_minmax'][1]

        for key in self.channels_stats.keys():
            self.channels_stats[key]['avg_length'] = self.channels_stats[key]['avg_length'] / self.examined_no
            self.channels_stats[key]['stddev_length'] = numpy.std(self.channels_stats[key]['stddev_length'])
            self.channels_stats[key]['mean_value'] = numpy.mean(self.channels_stats[key]['mean_value'])
            self.channels_stats[key]['mean_stddev_value'] = numpy.std(self.channels_stats[key]['mean_stddev_value'])
            self.channels_stats[key]['mean_minmax'][0] = self.channels_stats[key]['total_minmax'][0] if self.channels_stats[key]['total_minmax'][0] < self.channels_stats[key]['mean_minmax'][0] else self.channels_stats[key]['mean_minmax'][0]
            self.channels_stats[key]['mean_minmax'][1] = self.channels_stats[key]['total_minmax'][1] if self.channels_stats[key]['total_minmax'][1] > self.channels_stats[key]['mean_minmax'][1] else self.channels_stats[key]['mean_minmax'][1]

    def print_inputdata_insights(self):
        """
        Prints results of method prepare_inputdata_insights()

        :return:
        """
        for key in self.channels_stats.keys():
            header = "Channel no. " + str(key)
            print()
            print(header.center(40, '*'))
            print('Average length: ' + self.channels_stats[key]['avg_length'] + ' with deviation: ' + self.channels_stats[key]['stddev_length'])
            print('Mean value: ' + self.channels_stats[key]['mean_value'] + ' with \'mean\' deviation: ' + self.channels_stats[key]['mean_stddev_value'])
            print('Channel\'s global  minimum: ' + self.channels_stats[key]['mean_minmax'][0] + ', and average maximum: ' + self.channels_stats[key]['mean_minmax'][1])
            print('Average minimum: ' + self.channels_stats[key]['mean_minmax'][0] + ', and average maximum: ' + self.channels_stats[key]['mean_minmax'][1])

    # TODO output logic is WAY too complicated
    def prepare_target_ranges(self):
        """
        From excel file with columns:
        badany,	SPP,	SPH,	RPN,	Raven_A,	Raven_B,	Raven_C,	Raven_D,	Raven_E,	Raven_WO,
        IVE_Impulsywnosc,	IVE_Ryzyko,	IVE_Empatia,	SSZ,	SSE,	SSU,	ACZ,	PKT,
        read data into dictionary, where target data is mapped into ranges.

        Modified variables: output_ranges_x10[examined_no][test_no][10x value 0 or 1] and remember_output_reversal[test_no][minmax_tuple]

        :return void
        """
        print('   Wczytywanie danych do przedziałów', end='...')
        target_data = pandas.read_excel(variables.out_raw_filepath)
        target_data.drop(columns='badany')                            # drop first column

        for examined_no in self.input_examined.keys():
            self.remember_output_reversal[examined_no] = dict()
            for test_no in range(17):
                minimum = min(target_data.iloc[examined_no, :])
                maximum = max(target_data.iloc[examined_no, :])
                self.remember_output_reversal[examined_no][test_no] = (minimum, maximum)          # yes this could be more efficient but i find this convenient this way

        for examined_no in self.remember_output_reversal.keys():
            self.output_ranges_x10[examined_no] = dict()
            for test_no in self.remember_output_reversal[examined_no].keys():
                self.output_ranges_x10[examined_no][test_no] = self.get_ranged_list_outputs(self.remember_output_reversal[examined_no][test_no], target_data.iloc[examined_no, test_no])

        # at this point i have a dictionary with examined no. as keys containing a list as value.
        # The list contain ten values → 0 or 1, where 1 means the original value was in corresponding range
        # i.e. 39 becomes 0.18 when minmaxed* in <32, 71> → which becomes [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        # → https://en.wikipedia.org/wiki/Feature_scaling
        print(' zakończone.')

    def prepare_target_number(self, examination_no):
        """
        From excel file with columns:
        badany,	SPP,	SPH,	RPN,	Raven_A,	Raven_B,	Raven_C,	Raven_D,	Raven_E,	Raven_WO,
        IVE_Impulsywnosc,	IVE_Ryzyko,	IVE_Empatia,	SSZ,	SSE,	SSU,	ACZ,	PKT,
        read data into dictionary, where target data is saved as a number in <0, 1>.

        → https://www.mantidproject.org/Working_With_Functions:_Return_Values
        :return void
        """
        print('   Wczytywanie danych jako liczba', end='...')
        target_data = pandas.read_excel(variables.out_raw_filepath)
        target_data.drop(columns='badany')                            # drop first column

        for examined_no in self.input_examined.keys():
            self.remember_output_reversal[examined_no] = dict()
            for test_no in range(17):
                minimum = min(target_data.iloc[examined_no, :])
                maximum = max(target_data.iloc[examined_no, :])
                self.remember_output_reversal[examined_no][test_no] = (minimum, maximum)

        for examined_no in self.remember_output_reversal.keys():
            self.output_ranges_x10[examined_no] = dict()
            for test_no in self.remember_output_reversal[examined_no].keys():
                self.output_ranges_x10[examined_no][test_no] = (target_data.iloc[examined_no, test_no] - self.remember_output_reversal[examined_no][test_no][0]) / (self.remember_output_reversal[examined_no][test_no][0] - self.remember_output_reversal[examined_no][test_no][1])

    def get_ranged_list_outputs(self, minmax, score):
        """
        Given (min, max) tuple of examined's test score
        and considered score,
        outputs list of 1 or 0s
        where 1 stands for correct range out of 10 possible

        :param minmax: tuple of ints
        :param score: int
        :return: in ex.: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        """
        output = []
        input = (score - minmax[0]) / (minmax[1] - minmax[0])
        for i in range(10):
            index = i / 10
            output.append(1 if index <= input < index + 0.1 else 0)
        return output

    def interprete_prediction_from_list(self, prediction):
        """
        Given list with 10 elements - 9 zeros and 1 one,
        this method converts information from the list
        to the form as in raw input (reverse process from prepare_target_ranges()

        :return: int
        """
        pass

    def interprete_prediction_from_number(self):
        pass
