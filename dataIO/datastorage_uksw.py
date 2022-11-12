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
import sys
import gc

class UkswData(object):

    def __init__(self):
        self.input_examined = dict()            # main holder of data, nested dictionary: input_examined[examined_no][channel][datapoints]
        self.output_ranges_x10 = dict()         # output_ranges_x10[examined_no][test_no][10x value 0 or 1]
        self.remember_output_reversal = dict()  # saved mins and maxs to be able to restore values in scale remember_output_reversal[test_no][minmax_tuple]
        self.output_single_x1 = dict()          # output_single_x1[examined_no][test_no]
        self.files_list = [name for name in os.listdir(variables.uksw_in_raw_path)]
        self.files_no = len(self.files_list)
        self.examined_no = self.files_no / variables.channels_for_person
        print(f'   Found {str(self.files_no)} files.')
        self.channels_stats = dict()            # proper description in method: prepare_inputdata_insights()

        self.__load_channels()
        self.__fourier_transform()
        self.__prepare_inputdata_insights()
        self.print_inputdata_insights()
        self.__normalise_channel_data()

        print(f'UKSW data loaded successfully, taking {self.get_actual_size() / 1048576:.2f}MBs')

    def __load_channels(self):
        """
        The data is in form of nested dictionaries: input_examined[examined_no][channel][datapoints]
                                                    input_examined[0 → 33][1 → 10][0 → 100k++]
        :return: void
        """
        for index, file in enumerate(self.files_list):
            examined_person = math.floor((index + 1) / variables.channels_for_person)
            current_channel_no = index - examined_person * variables.channels_for_person
            if variables.limit_people and variables.limit_channels:
                if examined_person in variables.people_to_consider and current_channel_no in variables.channels_to_consider:
                    self.load_particular_channel(file, current_channel_no, examined_person)
            elif variables.limit_people and not variables.limit_channels:
                if examined_person in variables.people_to_consider:
                    self.load_particular_channel(file, current_channel_no, examined_person)
            elif not variables.limit_people and variables.limit_channels:
                if current_channel_no in variables.channels_to_consider:
                    self.load_particular_channel(file, current_channel_no, examined_person)
            else:
                self.load_particular_channel(file, current_channel_no, examined_person)

    def load_particular_channel(self, file, current_channel_no, examined_person):
        loaded_data = numpy.genfromtxt(variables.uksw_in_raw_path + file, delimiter=',', dtype=numpy.float32)
        data_to_save = numpy.delete(loaded_data, slice(0, variables.how_many_to_drop), axis=None)
        if not examined_person in self.input_examined:
            self.input_examined[examined_person] = {}
        self.input_examined[examined_person][current_channel_no] = data_to_save

    def apply_filters(self):
        """
        TODO for time being i don't know the parameters for high- and low pass filters
        :return:
        """
        pass

    def __fourier_transform(self):
        """
        Alpha waves have frequency between 8Hz and 12Hz
        → https://en.wikipedia.org/wiki/Alpha_wave
        This method use Fast Fourier Transform to convert
        EEG data info to it's frequency info, trim information
        concerning non-alpha wave frequencies, and convert
        frequency info back to wave datapoints.
        → https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/
        → https://www.youtube.com/watch?v=spUNpyF58BY

        What is important, numpy.fft.ifft() will give complex output even if it should
        be real (small imaginary addon will be always present!) Python marks imaginary
        part with a "j" letter!

        :return:
        """
        lower_data_limes = variables.alpha_low_frequency / variables.frequencyOfData
        higher_data_limes = variables.alpha_high_frequency / variables.frequencyOfData

        for examined_keys, examined_vals in self.input_examined.items():
            for channel_keys, channel_vals in examined_vals.items():
                channel_vals = numpy.fft.fft(channel_vals)
                channel_length = len(channel_vals)
                for i in range(channel_length):
                    channel_vals[i] = channel_vals[i] if math.floor(lower_data_limes * channel_length) < i < math.ceil(higher_data_limes * channel_length) else 0
                self.input_examined[examined_keys][channel_keys] = numpy.real(numpy.fft.ifft(channel_vals))   # TODO trim unnecessary parts (zeros from line above)
                # self.input_examined[examined_keys][channel_keys] = numpy.imag(channel_vals)

    def __normalise_channel_data(self):
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

    def __prepare_inputdata_insights(self):
        """
        Method meant to give some insights about input data.
        It calculates channels averages with deviations.
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
        first_key = variables.people_to_consider[0]                 # to know where to start indexing
        for channel_keys in self.input_examined[first_key].keys():
            if not channel_keys in self.channels_stats:
                self.channels_stats[channel_keys] = dict()
            self.channels_stats[channel_keys]['avg_length'] = 0
            self.channels_stats[channel_keys]['stddev_length'] = []
            self.channels_stats[channel_keys]['mean_value'] = 0
            self.channels_stats[channel_keys]['mean_stddev_value'] = []
            self.channels_stats[channel_keys]['total_minmax'] = [math.inf, 0]
            self.channels_stats[channel_keys]['mean_minmax'] = [math.inf, 0]

        for examined_keys, examined_vals in self.input_examined.items():
            for channel_keys, channel_vals in examined_vals.items():
                self.channels_stats[channel_keys]['avg_length'] += len(channel_vals)
                self.channels_stats[channel_keys]['stddev_length'].append(len(channel_vals))
                self.channels_stats[channel_keys]['mean_value'] += numpy.mean(channel_vals)
                self.channels_stats[channel_keys]['mean_stddev_value'].append(numpy.std(channel_vals))
                self.channels_stats[channel_keys]['total_minmax'][0] = min(channel_vals) if self.channels_stats[channel_keys]['total_minmax'][0] > min(channel_vals) else self.channels_stats[channel_keys]['total_minmax'][0]
                self.channels_stats[channel_keys]['total_minmax'][1] = max(channel_vals) if self.channels_stats[channel_keys]['total_minmax'][1] < max(channel_vals) else self.channels_stats[channel_keys]['total_minmax'][1]

        for key in self.channels_stats.keys():
            self.channels_stats[key]['avg_length'] = self.channels_stats[key]['avg_length'] / len(self.input_examined.keys())
            self.channels_stats[key]['stddev_length'] = numpy.std(self.channels_stats[key]['stddev_length'])
            self.channels_stats[key]['mean_value'] = numpy.mean(self.channels_stats[key]['mean_value'])
            self.channels_stats[key]['mean_stddev_value'] = numpy.std(self.channels_stats[key]['mean_stddev_value'])    # type change from list to float
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
            print(f'Average length: {str(self.channels_stats[key]["avg_length"])} with deviation: {str(self.channels_stats[key]["stddev_length"])}')
            print(f'Mean value: {str(self.channels_stats[key]["mean_value"])} with \'mean\' deviation: {str(self.channels_stats[key]["mean_stddev_value"])}')
            print(f'Channel\'s global  minimum: {str(self.channels_stats[key]["mean_minmax"][0])}, and average maximum: {str(self.channels_stats[key]["mean_minmax"][1])}')
            print(f'Average minimum: {str(self.channels_stats[key]["mean_minmax"][0])}, and average maximum: {str(self.channels_stats[key]["mean_minmax"][1])}')

    def adjust_input_data_lengths(self):
        """
        Neural networks need to have constant input size.
        The number of input length is defined in the variables file.
        After viewing input data insights I recommend 100k data points length
        for each channel, each person.
        :return:
        """
        for examined_keys, examined_vals in self.input_examined.items():
            for channel_keys, channel_vals in examined_vals.items():
                channel_length = len(channel_vals)
                if channel_length > variables.desired_data_length:
                    difference = channel_length - variables.desired_data_length
                    if difference % 2 == 0:
                        channel_vals = numpy.delete(channel_vals, slice(channel_length - difference/2, channel_length), axis=None)
                        channel_vals = numpy.delete(channel_vals, slice(0, difference/2), axis=None)
                    else:
                        channel_vals = numpy.delete(channel_vals, 0, axis=None)
                        channel_length = len(channel_vals)
                        difference = channel_length - variables.desired_data_length
                        channel_vals = numpy.delete(channel_vals, slice(channel_length - difference/2, channel_length), axis=None)
                        channel_vals = numpy.delete(channel_vals, slice(0, difference/2), axis=None)
                elif channel_length < variables.desired_data_length:
                    pass
                self.input_examined[examined_keys][channel_keys] = channel_vals

    def prepare_target_ranges(self):
        """
        From excel file with columns:
        badany,	SPP,	SPH,	RPN,	Raven_A,	Raven_B,	Raven_C,	Raven_D,	Raven_E,	Raven_WO,
        IVE_Impulsywnosc,	IVE_Ryzyko,	IVE_Empatia,	SSZ,	SSE,	SSU,	ACZ,	PKT,
        Kłamstwo, Neurotycznosc, Ekstrawersja, Psychotyzm
        read data into dictionary, where target data is mapped into ranges.

        Modified variables: output_ranges_x10[examined_no][test_no][10x value 0 or 1] and remember_output_reversal[test_no][minmax_tuple]

        I will have a dictionary of dictionaries containing a list as value each.
        # The list contain ten values → 0 or 1, where 1 means the original value was in corresponding range
        # i.e. 39 becomes 0.18 when minmaxed* in <32, 71> → which becomes [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        # → https://en.wikipedia.org/wiki/Feature_scaling
        :return void
        """
        print('   Loading target data for bins', end='...')
        target_data = pandas.read_excel(variables.out_raw_filepath)
        target_data.drop(columns='badany')                            # drop first column

        for examined_no in self.input_examined.keys():
            self.remember_output_reversal[examined_no] = dict()
            for test_no in range(21):
                minimum = min(target_data.iloc[examined_no, :])
                maximum = max(target_data.iloc[examined_no, :])
                self.remember_output_reversal[examined_no][test_no] = (minimum, maximum)          # yes this could be more efficient but i find this convenient this way

        for examined_no in self.remember_output_reversal.keys():
            self.output_ranges_x10[examined_no] = dict()
            for test_no in self.remember_output_reversal[examined_no].keys():
                self.output_ranges_x10[examined_no][test_no] = self.get_ranged_list_outputs(self.remember_output_reversal[examined_no][test_no], target_data.iloc[examined_no, test_no])

        print(' finished.')

    def prepare_target_number(self, examination_no):
        """
        From excel file with columns:
        badany,	SPP,	SPH,	RPN,	Raven_A,	Raven_B,	Raven_C,	Raven_D,	Raven_E,	Raven_WO,
        IVE_Impulsywnosc,	IVE_Ryzyko,	IVE_Empatia,	SSZ,	SSE,	SSU,	ACZ,	PKT,
        Kłamstwo, Neurotycznosc, Ekstrawersja, Psychotyzm
        read data into dictionary, where target data is saved as a number in <0, 1>.

        → https://www.mantidproject.org/Working_With_Functions:_Return_Values
        :return void
        """
        print('   Loading target data as a number', end='...')
        target_data = pandas.read_excel(variables.out_raw_filepath)
        target_data.drop(columns='badany')                            # drop first column

        for examined_no in self.input_examined.keys():
            self.remember_output_reversal[examined_no] = dict()
            for test_no in range(21):
                minimum = min(target_data.iloc[examined_no, :])
                maximum = max(target_data.iloc[examined_no, :])
                self.remember_output_reversal[examined_no][test_no] = (minimum, maximum)

        for examined_no in self.remember_output_reversal.keys():
            self.output_single_x1[examined_no] = dict()
            for test_no in self.remember_output_reversal[examined_no].keys():
                self.output_single_x1[examined_no][test_no] = (target_data.iloc[examined_no, test_no] - self.remember_output_reversal[examined_no][test_no][0]) / (self.remember_output_reversal[examined_no][test_no][0] - self.remember_output_reversal[examined_no][test_no][1])

    print(' finished.')

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

    def interprete_prediction_from_list(self, minmax, prediction_list):
        """
        Given list with 10 elements - 9 zeros and 1 one,
        this method converts information from the list
        to the form as in raw input (reverse process from prepare_target_ranges()

        :return: int
        """
        output = 0
        for index, number in enumerate(prediction_list):
            output = index * 10 if number == 1 else output
        output = output * (minmax[1] - minmax[0]) + (minmax[0] * (minmax[1] - minmax[0]))

        return output

    def interprete_prediction_from_number(self, minmax, prediction):
        return prediction * (minmax[1] - minmax[0]) + (minmax[0] * (minmax[1] - minmax[0]))

    def get_actual_size(self):
        """
        sys.getsizeof(object) returns size of object with attached memory addresses,
        but not attached objects' size in memory

        :return: amount of bytes taken in memory by this class' instance
        """
        memory_size = 0
        ids = set()
        objects = [self]
        while objects:
            new = []
            for obj in objects:
                if id(obj) not in ids:
                    ids.add(id(obj))
                    memory_size += sys.getsizeof(obj)
                    new.append(obj)
            objects = gc.get_referents(*new)
        return memory_size