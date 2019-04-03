"""
Class created to initialize source data only once,
despite the number of examinations.
"""

import variables
import math
import numpy
import pandas
import os
import os.path


class Datastorage(object):

    def __init__(self):
        self.input_examined = dict()
        self.output_examined = dict()
        self.files_list = [name for name in os.listdir(variables.in_raw_path)]
        self.files_no = len(self.files_list)
        print('   Znaleziono ' + str(self.files_no) + ' badanych.')
        self.isDataInitialised = False
        self.minmax_tuple = ()
        self.minmax_channelLength_tuple = ()

    def prepare_input(self):
        """
        From raw csv select proper channels, @TODO proper network could read all channels simultaneously?
        proper are channels 1 - 8 and 13 and 14, since i was told other channels contain too much noise
        trim useless frequencies,
        standardise data.

        :return void
        """
        print('   Wczytywanie kanałów', end='... ')
        self.load_channels()
        print('   zakończone.')
        # at this point i have a dictionary with filenames as keys containing dictionaries with channel numbers as keys (and channel numpy array data as values)

        # @TODO dat trimming to lowest length (drop initial varying n points for every channel)

        # @TODO low- and highpass filters
        print('   Filtrowanie fal alfa...', end='... ')
        self.data_fourier_transform()
        print('   zakończone.')

        print("   Standaryzacja danych", end="...")
        self.standardise_channel_data()
        print("zakończona.")
        # at this point data is standardised around 0, with outsider values deleted

        print("   Normalizacja danych", end="...")
        self.normalise_channel_data()
        print("zakończona.")
        # at this point data is normalised in <0, 1>
        # @TODO or maybe i should normalise in <-1, 1> and introduce (-1, 1) to initialized weights in neurons ??

        self.prepare_infoForNetworks()

        self.isDataInitialised = True

    def load_channels(self):
        """
        # @TODO wymaga poprawy!!
        :return:
        """
        for file in self.files_list:
            temporary_mem_channels = dict()
            self.input_examined[file] = numpy.genfromtxt(variables.in_raw_path + file, delimiter=',', dtype=numpy.float32)
            self.input_examined[file] = numpy.delete(self.input_examined[file], variables.how_many_to_drop, axis=None)
            channel_size = self.count_channel_size(self.input_examined[file])
            channels_no = numpy.floor(len(self.input_examined[file]) / channel_size).astype(int)
            print(channels_no, end=", ")
            for channel in range(0, channels_no):
                temporary_mem_channels[channel] = self.input_examined[file][channel * channel_size : (channel + 1) * channel_size]
            self.input_examined[file] = temporary_mem_channels

    def data_apply_filters(self):
        pass

    def data_fourier_transform(self):
        """
        Alpha waves have frequency between 8Hz and 12Hz
        → https://en.wikipedia.org/wiki/Alpha_wave
        This method use Fast Fourier Transform to convert
        EEG data info to it's frequency info, trim information
        concerning non-alpha wave frequencies, and convert
        frequency info back to wave datapoints.
        → → https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/

        :return:
        """
        progress = 0
        lower_data_limes = variables.alpha_low_frequency / variables.frequencyOfData
        higher_data_limes = variables.alpha_high_frequency / variables.frequencyOfData
        for examined_keys, examined_vals in self.input_examined.items():
            for channel_keys, channel_vals in examined_vals.items():
                channel_vals = numpy.fft.fft(channel_vals)
                channel_length = len(channel_vals)
                for i in range(channel_length):
                    channel_vals[i] = channel_vals[i] if math.floor(lower_data_limes)*channel_length < i < math.ceil(higher_data_limes)*channel_length else 0
                channel_vals = numpy.fft.ifft(channel_vals)
            progress += 1
            print(str(int((progress / len(self.input_examined) * 100))), end="%, ")


    def standardise_channel_data(self):
        progress = 0
        for examined_keys, examined_vals in self.input_examined.items():
            for channel_keys, channel_vals in examined_vals.items():
                for i in range(len(channel_vals)-1):
                    channel_vals[i] = channel_vals[i+1] - channel_vals[i]
                    channel_vals[-1] = 0
            progress += 1
            print(str(int((progress / len(self.input_examined) * 100))), end="%, ")
            for channel_key, channel_value in examined_vals.items():                # @TODO it takes eternity to process, needs optimization
                slice_mean = numpy.mean(channel_value)
                slice_dev = numpy.std(channel_value)
                for i in range(len(channel_value)):
                    # if numpy.absolute(channel_value[i] - slice_mean) > 4 * slice_dev:
                    # channel_value[i] = 0
                    if channel_value[i] - slice_mean > 4 * slice_dev:
                        channel_value[i] = slice_mean + 3 * slice_dev
                    elif channel_value[i] - slice_mean < -4 * slice_dev:
                        channel_value[i] = slice_mean - 3 * slice_dev

    def normalise_channel_data(self):
        for examined_keys, examined_vals in self.input_examined.items():
            minimum = math.inf
            maximum = 0
            for channel_keys, channel_vals in examined_vals.items():
                minimum = min(channel_vals) if min(channel_vals) < minimum else minimum
                maximum = max(channel_vals) if max(channel_vals) > maximum else maximum
            for channel_keys, channel_vals in examined_vals.items():
                for i in range(len(channel_vals)):
                    channel_vals[i] = ((channel_vals[i] - minimum) / (maximum - minimum)).astype(dtype=numpy.float32)

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
        data_slice = eeg[i - 100000:i]  # last hundred thousand points from eeg, assumed they're always from last channel
        slice_mean = numpy.mean(data_slice)
        slice_dev = numpy.std(data_slice)
        flag_rised = True
        while flag_rised:
            i -= 1
            if numpy.absolute(eeg[i] - slice_mean) > 2 * 3 * slice_dev:  # find first 'impossible' value - impossible because: → https://en.wikipedia.org/wiki/Standard_score
                flag_rised = False
                return len(eeg) - i

    def prepare_infoForNetworks(self):
        """

        :return:
        """
        minimum = math.inf
        maximum = 0
        suma = 0
        count = 0
        for key in self.input_examined:
            for channel_key in self.input_examined[key]:
                minimum = len(self.input_examined[key][channel_key]) if minimum > len(self.input_examined[key][channel_key]) else minimum
                maximum = len(self.input_examined[key][channel_key]) if maximum < len(self.input_examined[key][channel_key]) else maximum
                suma += len(self.input_examined[key][channel_key])
                count += 1
        self.minmax_channelLength_tuple = (minimum, maximum)
        worst = int((maximum - minimum) / maximum * 100)
        average = int((maximum - (suma / count)) / maximum * 100)
        print("   Część danych utracona z powodu różnicy ilości punktów kanałów. Najorszy przypadek: " + str(worst) + "% straty danych. Średnio: " + str(average) + "%")

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
