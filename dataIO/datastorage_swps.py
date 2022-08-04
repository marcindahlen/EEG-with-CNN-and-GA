"""
Class created to initialize source data
regarding depression only once.
"""
import sys
import gc
from scipy.io import loadmat
from utils import variables
import math
import numpy


class SwpsData(object):

    def __init__(self):
        self.input_data_filename = variables.swps_in_raw_path + "psd_study-C_eyes-open_space-avg_winlen-2.0_step-0.5_tmin-2.0_tmax-60.0"
        self.output_data_filename = variables.swps_in_raw_path + "bdi"
        self.input_examined = dict()            # main holder of data, nested dictionary: input_examined[examined_no][channel][datapoints]
        self.output_ranges_x10 = dict()         # output_ranges_x10[examined_no][test_no][10x value 0 or 1]
        self.load_input()
        self.load_output()

        print(f'SWPS data loaded successfully, taking {self.get_actual_size() / 1048576:.2f}MBs')

    def load_input(self):
        psds_mat = loadmat(self.input_data_filename)
        keys = ['psd', 'freq', 'ch_names', 'subj_id']
        psds, *rest = [psds_mat[k] for k in keys]
        freq, ch_names, subj_id = [x.ravel() for x in rest]
        ch_names = [ch.replace(' ', '') for ch in ch_names]
        print(psds.shape)
        print(len(freq))
        print(len(ch_names))
        print(len(subj_id))
        print(len(rest))

    def load_output(self):
        data_in = numpy.genfromtxt(self.output_data_filename + '.csv', delimiter=',', dtype=numpy.float32)
        print(len(data_in))

    def get_actual_size(self):
        """

        :return:
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