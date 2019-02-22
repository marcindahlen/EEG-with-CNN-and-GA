"""
Variables used across the experiment
"""

from math import floor

network_input_window = 24
network_topology = [16, 8, 1]
accepted_min_rmse = 1
population_quantity = 32                                        #this is the number of input eeg examinations
how_many_networks_to_save = floor(population_quantity / 4)
mutation_probability_factor = 1
in_raw_path = "../in_raw/"
network_dnas_path = "../network_dnas/"
out_raw_path = "../out_raw/"
out_charts_path = "../out_wykresy/"
how_many_to_drop = 20000
