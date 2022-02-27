"""
I assume following files hierarchy:
    root/
        uksw_in_raw/
            P01.txt
            .
            .
            .
            P32.txt
        swps_in_raw/
            .
            .
            .
        in_raw_channels/
            P01CH01
            P01CH02
            .
            .
            .
            P32Ch14
        uksw_out_raw/
            main_alpha-index_base.xls
            out_absData.xls
        out_charts/

        python/
            dataIO/
            layers/
            networks/
            nodes/
            populations/
            tests/
            utils/
            notepad.py
            main.py
"""

# imports
from dataIO.datastorage_uksw import UkswData

"""
# load input data
input_data_uksw = Datastorage()
input_data_uksw.load_channels()
input_data_uksw.fourier_transform()
input_data_uksw.prepare_inputdata_insights()
input_data_uksw.print_inputdata_insights()
input_data_uksw.normalise_channel_data()

# load output data
target_data = data_eeg.prepare_target_ranges()

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
"""


