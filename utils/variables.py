"""
Variables used across the experiment
"""
from math import floor

#TODO
channels_to_consider = [3, 7]                                           # only these particular channels will be considered
limit_channels = False                                                   # if True, only channels_to_consider are loaded
# TODO
people_to_consider = [4, 5]                                             # only these particular people will be considered
limit_people = False                                                     # if True, only people_to_consider will be considered
# TODO
testscale_to_consider = [2]                                             # only these test will be processed and learned
limit_tests = True                                                      # if True, only testscale_to_consider will be considered

channels_for_person = 10

population_quantity = 32                                                # this is the number of input eeg examinations
how_many_networks_to_save = floor(population_quantity / 4)
mutation_probability_factor = 1

in_raw_path = "../../in_raw/"
net_memory_path = "../../net_memory/"
out_raw_filepath = "../../out_raw/out_absData.xls"
out_charts_path = "../../out_wykresy/"

how_many_to_drop = 20000                                                # I was told to drop this amount of initial data, so i'm dropping it (for each channel)
desired_data_length = 100000

frequencyOfData = 256                                                   # how many data points are recorded per second (eeg)
alpha_low_frequency = 8
alpha_high_frequency = 12

examination_names = {0: 'SPP',
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

email_addresses = ['marcindahlen@gmail.com']
