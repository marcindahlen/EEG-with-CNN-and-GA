"""
I assume following files' hierarchy:
    root/
        data/
            out_charts/
            out_networks/
            swps_in_raw/
                .
                .
                .
            uksw_in_raw/
                P01.txt
                .
                .
                .
                P32.txt
            uksw_out_raw/
                main_alpha-index_base.xls
                out_absData.xls
        dataIO/
        layers/
        networks/
        nodes/
        populations/
        tests/
        utils/
        main.py
"""

# imports
from dataIO.datastorage_uksw import UkswData
from dataIO.datastorage_swps import SwpsData

# load input data
# input_data_uksw = UkswData()

input_data_swps = SwpsData()

# load output data

# test system by spawning networks and single pass each for test, note results for comparison                          1

# examine pooling-convolution-fullyConnected network, each population examining one separate data channel              2

# initialise populations of networks

# evolve conv networks in a population

# evolution done, save stats

# examine pooling-convolution-LSTM network, where LSTM layers are single for all channels                              3

# initialise populations of networks

# evolve conv networks in a population

# evolution done, save stats

# examine "EEG herding", join eeg channels data as a single image of 0s and 1s,
# evolve pooling-convolution-fullyConnected network                                                                    4

# initialise populations of networks

# evolve conv networks in a population

# evolution done, save stats

# examine pooling-LSTM-fullyConnected network, each population examining one separate data channel                     5

# initialise populations of networks

# evolve conv networks in a population

# evolution done, save stats

# print and save stats                                                                                                 6

# generate and save charts, send emails if needed                                                                      7
