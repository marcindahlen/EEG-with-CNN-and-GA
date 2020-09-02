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
population_quantity = 32
how_many_networks_to_save = floor(population_quantity / 4)
mutation_probability_factor = 1

single_channel_network_layers = [Layer.AvgPool, Layer.convolution, Layer.AvgPool, Layer.convolution, Layer.convolution,
                                 Layer.LSTM, Layer.LSTM, Layer.LSTM]
herding_network_layers = []
random_guessing_test_network_layers = []
single_LSTM_for_all_channels_layers = []

in_raw_path = "../../in_raw/"
net_memory_path = "../../net_memory/"
out_raw_filepath = "../../out_raw/out_absData.xls"
out_charts_path = "../../out_wykresy/"

# I was told to drop this amount of initial data, so i'm dropping it (for each channel)
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
