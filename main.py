
from dataIO.datastorage import Datastorage

# load input data
data = Datastorage()
data.load_channels()
data.fourier_transform()
data.prepare_inputdata_insights()
data.print_inputdata_insights()
data.normalise_channel_data()

# load output data
target_data = data.prepare_target_ranges()

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

# generate and save charts                                                                                             7


