"""
Variables used across the experiment
"""

from math import floor

channels_to_consider = [7, 12, 13]
window_base_length = 10 * 10 * 16                                       # handcoded "reasonable" amount, should be adjusted in the process

network_input_window = window_base_length * len(channels_to_consider)   # "receptive field", "filter size" times channel number
                                                                        # → http://cs231n.github.io/convolutional-networks/#conv

network_topology = [10 * 16, 5 * 8, 10 * 1]                             # Network classifies eeg data to a particular slice of possible outcomes,
                                                                        # as an input there are 10 channels of data points, - see comments in 'badanie.py'
                                                                        # each channel is about 150k in length,
                                                                        # output is affiliation to one of ten classes

population_quantity = 32                                                # this is the number of input eeg examinations
how_many_networks_to_save = floor(population_quantity / 4)
mutation_probability_factor = 1

in_raw_path = "../in_raw/"
net_memory_path = "../net_memory/"
out_raw_filepath = "../out_raw/out_absData.xls"
out_charts_path = "../out_wykresy/"

how_many_to_drop = 20000                                                # I was told to drop this amount of initial data, so i'm dropping it
