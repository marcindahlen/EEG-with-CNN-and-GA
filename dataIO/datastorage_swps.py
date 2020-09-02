"""
Class created to initialize source data
regarding depression only once.
"""
from scipy.io import loadmat
from utils import variables
import math
import numpy
import os
import os.path


class Datastorage(object):

    def __init__(self):
        self.input_data_filename = variables.in_raw_path + "psd_study-C_eyes-open_space-avg_winlen-2.0_step-0.5_tmin-2.0_tmax-60.0"
        self.output_data_filename = variables.in_raw_path + "bdi"
        self.input_examined = dict()            # main holder of data, nested dictionary: input_examined[examined_no][channel][datapoints]
        self.output_ranges_x10 = dict()         # output_ranges_x10[examined_no][test_no][10x value 0 or 1]
        self.load_input()
        self.load_output()

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


storage = Datastorage()
