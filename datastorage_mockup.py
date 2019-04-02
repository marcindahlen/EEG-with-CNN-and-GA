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

        :return void
        """
        size = 8 * variables.window_base_length
        for file in self.files_list:
            temporary_mem_channels = dict()
            for channel_no in range(3):
                temporary_mem_channels[channel_no] = []
            self.input_examined[file] = temporary_mem_channels

    def prepare_target(self, examination_no):
        """
        From excel file with columns:
        badany,	SPP,	SPH,	RPN,	Raven_A,	Raven_B,	Raven_C,	Raven_D,	Raven_E,	Raven_WO,	IVE_Impulsywnosc,	IVE_Ryzyko,	IVE_Empatia,	SSZ,	SSE,	SSU,	ACZ,	PKT
        read data,

        → https://www.mantidproject.org/Working_With_Functions:_Return_Values
        :return void
        """
        pass

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
        pass

    def prepare_infoForNetworks(self):
        """

        :return:
        """
        pass

    def assume_networkIterationsNo(self):
        """

        :return:
        """
        pass

    def show_summary(self):
        """

        :return:
        """
        pass

    def decide_no_belonging(self, number, index):
        """
        Given minmaxed value of examined's test score
        and currently considered index in list being build,
        outputs 1 or 0 where 1 is a valid match
        :param number:
        :param index:
        :return: int: 1 or 0
        """
        pass

    def interprete_prediction(self, prediction):
        """
        Given list with 10 elements - 9 zeros and 1 one,
        this method converts information from the list
        to the form as in raw input (reverse process from prepare_target()
        :return: int
        """
        pass
