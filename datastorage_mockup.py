"""
Class needed to mockup data
for testing the evolution process
of neural networks.
"""

import variables
import numpy
import pandas
import os
import os.path


class Datamockup(object):

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
        → https://playground.tensorflow.org     # training set ideas
        Input is mocked as the input from real files,
        meaning format dict(dict(list)).
        First dictionary simulates the examined people files,
        second simulates channel (only one for simplicity),
        the list contains data generated from inverse Fourier transform.

        Network's task is to classify data to source frequency (from finite frequencies set)
        :return void
        """
        size = 8 * variables.window_base_length
        for index, file in enumerate(self.files_list):
            temporary_mem_channels = dict()
            for channel_no in range(3):
                temporary_mem_channels[channel_no] = [0 for x in range(size)]
                temporary_mem_channels[channel_no][index] = 1 + 1j if index < 10 else 0
                temporary_mem_channels[channel_no] = numpy.fft.ifft(temporary_mem_channels[channel_no])
            self.input_examined[file] = temporary_mem_channels

    def prepare_target(self, examination_no):
        """

        :return void
        """
        for index, file in enumerate(self.files_list):
            self.output_examined[file] = [0 for x in range(10)]
            self.output_examined[file][index] = 1 if index < 10 else 0

    def normalise_channel_data(self):
        for examined_keys, examined_vals in self.input_examined.items():
            minimum = numpy.inf
            maximum = 0
            for channel_keys, channel_vals in examined_vals.items():
                minimum = min(channel_vals) if min(channel_vals) < minimum else minimum
                maximum = max(channel_vals) if max(channel_vals) > maximum else maximum
            for channel_keys, channel_vals in examined_vals.items():
                for i in range(len(channel_vals)):
                    channel_vals[i] = ((channel_vals[i] - minimum) / (maximum - minimum)).astype(dtype=numpy.float32)

