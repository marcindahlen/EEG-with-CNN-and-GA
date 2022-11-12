"""
Variables used across the experiment
"""

from math import floor
from layers.available_layers import Layer

# only these particular channels will be considered
channels_to_consider = [3, 7]

# if True, only channels_to_consider are loaded
limit_channels = False

# only these particular people will be considered
people_to_consider = [4, 5]

# if True, only people_to_consider will be considered
limit_people = False

# only these test will be processed and learned (this is a list of test's numbers)
testscale_to_consider = [2]

# if True, only testscale_to_consider will be considered
limit_tests = True

channels_for_person = 10

# this is the number of input eeg examinations
population_quantity = 64
how_many_networks_to_save = floor(population_quantity / 4)

# auto tuned evolution parameters will be affected by this
mutation_probability_factor = 1

# how many points to cut 'dna' for networks to exchange weights
multi_crossovers_limit = 2

single_channel_network_layers_LSTM = [Layer.AvgPool, Layer.convolution, Layer.AvgPool, Layer.convolution,
                                      Layer.convolution, Layer.LSTM, Layer.LSTM, Layer.LSTM]
single_channel_network_layers_LSTM_IO = [(100000, 20000),
                                         (20000, (32, 4000)),
                                         ((32, 4000), (32, 800)),
                                         ((32, 800), (64, 160)),
                                         ((64, 160), (128, 32)),
                                         ((128, 32), 1024),
                                         (1024, 1024),
                                         (1024, 10)]

single_channel_network_layers_basic = [Layer.AvgPool, Layer.convolution, Layer.AvgPool, Layer.convolution,
                                       Layer.convolution, Layer.basic_neuron, Layer.basic_neuron, Layer.basic_neuron]
single_channel_network_layers_basic_IO = [(100000, 20000),
                                          (20000, (32, 4000)),
                                          ((32, 4000), (32, 800)),
                                          ((32, 800), (64, 160)),
                                          ((64, 160), (128, 32)),
                                          ((128, 32), 1024),
                                          (1024, 1024),
                                          (1024, 10)]

herding_network_layers = []
herding_network_layers_IO = []

random_guessing_test_network_layers = [Layer.AvgPool, Layer.MaxPool, Layer.convolution, Layer.basic_neuron]

single_LSTM_for_all_channels_layers = []

uksw_in_raw_path = "data/uksw_in_raw/"
swps_in_raw_path = "data/swps_in_raw/"
uksw_out_raw_path = "data/uksw_out_raw/out_absData.xls"
out_charts_path = "data/out_charts/"

# I was told to drop this amount of initial data, so I'm dropping it (for each channel)
how_many_to_drop = 20000
# By rule of thumb, I set this number as a max length for input data (networks need standardisation)
desired_data_length = 100000

# how many data points are recorded per second (eeg)
frequencyOfData = 256
alpha_low_frequency = 8
alpha_high_frequency = 12

examination_test_names = {0: 'SPP',
                          1: 'SPH',
                          2: 'RPN',
                          3: 'Raven A',
                          4: 'Raven B',
                          5: 'Raven C',
                          6: 'Raven D',
                          7: 'Raven E',
                          8: 'Raven WO',
                          9: 'IVE Impulsywnosc',
                          10: 'IVE Ryzyko',
                          11: 'IVE Empatia',
                          12: 'SSZ',
                          13: 'SSE',
                          14: 'SSU',
                          15: 'ACZ',
                          16: 'PKT',
                          17: 'Lie',
                          18: 'Neuro',
                          19: 'Ekstr',
                          20: 'Psycho'}

# used to send emails with summary after completion
email_addresses = ['marcindahlen@gmail.com']
should_send_mail = True
