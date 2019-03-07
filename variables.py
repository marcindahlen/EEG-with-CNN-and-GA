"""
Variables used across the experiment
"""

from math import floor

network_input_window = 24                                       #"receptive field", "filter size", → http://cs231n.github.io/convolutional-networks/#conv

network_topology = [10 * 16, 5 * 8, 10 * 1]                     #Network classifies eeg data to a particular slice of possible outcomes,
                                                                #as an input there are 10 channels of data points, - see comments in 'badanie.py'
                                                                #each channel is about 150k in length,
                                                                #output is affiliation to one of ten classes

accepted_min_rmse = 1
population_quantity = 32                                        #this is the number of input eeg examinations
how_many_networks_to_save = floor(population_quantity / 4)
mutation_probability_factor = 1

in_raw_path = "../in_raw/"
network_dnas_path = "../network_dnas/"
out_raw_filepath = "../out_raw/out_absData.xls"
out_charts_path = "../out_wykresy/"

how_many_to_drop = 20000                                        #I was told to drop this amount of initial data, so i'm dropping it
