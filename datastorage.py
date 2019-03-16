"""
Class created to initialize source data only once,
despite the number of examinations.
"""

import variables
import math
import numpy
import os
import os.path


class Datastorage(object):

    def __init__(self):
        self.input_examined = dict()
        self.output_examined = dict()
        self.files_list = [name for name in os.listdir(variables.in_raw_path)]  # @TODO would be good to do NOT load same data every time
        self.files_no = len(self.files_list)
        print('   Znaleziono ' + str(self.files_no) + ' badanych.')
        self.minmax_tuple = ()
        self.isDataInitialised = False

    def prepare_input(self):
        """
        From raw csv select proper channels, @TODO proper network could read all channels simultaneously?
        proper are channels 1 - 8 and 13 and 14, since i was told other channels contain too much noise
        trim useless frequencies,
        standardise data.

        :return void
        """
        print('   Wczytywanie kanałów', end='... ')
        for file in self.files_list:
            temporary_mem_channels = dict()
            self.input_examined[file] = numpy.genfromtxt(variables.in_raw_path + file, delimiter=',')
            self.input_examined[file] = numpy.delete(self.input_examined[file], variables.how_many_to_drop, axis=None)
            channel_size = self.count_channel_size(self.input_examined[file])
            channels_no = numpy.floor(len(self.input_examined[file]) / channel_size).astype(int)
            print(channels_no, end=", ")
            for channel in range(0, channels_no):
                temporary_mem_channels[channel] = self.input_examined[file][channel * channel_size : (channel + 1) * channel_size]
            self.input_examined[file] = temporary_mem_channels
        print('   zakończone.')
        # at this point i have a dictionary with filenames as keys containing dictionaries with channel numbers as keys (and channel numpy array data as values)

        # @TODO low- and highpass filters + fourier 8-12Hz
        print('   >>fourier placeholder - not implemented yet<<')

        print("   Standaryzacja danych", end="...")
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
        print("zakończona.")
        # at this point data is standardised around 0, with outsider values deleted

        print("   Normalizacja danych", end="...")
        for examined_keys, examined_vals in self.input_examined.items():
            minimum = math.inf
            maximum = 0
            for channel_keys, channel_vals in examined_vals.items():
                minimum = min(channel_vals) if min(channel_vals) < minimum else minimum
                maximum = max(channel_vals) if max(channel_vals) > maximum else maximum
            for channel_keys, channel_vals in examined_vals.items():
                for i in range(len(channel_vals)):
                    channel_vals[i] = (channel_vals[i] - minimum) / (maximum - minimum)
        print("zakończona.")
        # at this point data is normalised in <0, 1>
        # @TODO or maybe i should normalise in <-1, 1> and introduce (-1, 1) to initialized weights in neurons ??

        self.isDataInitialised = True

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

