
from dataIO.datastorage_channels import Datastorage

# load input data
data = Datastorage()
data.load_channels()
data.fourier_transform()
data.prepare_inputdata_insights()
data.print_inputdata_insights()
data.normalise_channel_data()

# load output data
target_data = data.prepare_target_ranges()

# spawn convolution networks in a population

# evolve conv networks in a population

# evolution done, draw charts: best accuracy each epoch, other stats

# save best performing network, save metadata (stats) for further comparisons

# spawn simple full-connected networks in a population

# evolve simple networks in a population

# evolution done, draw charts: best accuracy each epoch, other stats

# save best performing network, save metadata (stats) for further comparisons

# spawn LSTM, window-reading networks in a population

# evolve LSTM networks in a population

# evolution done, draw charts: best accuracy each epoch, other stats

# save best performing network, save metadata (stats) for further comparisons

# populations comparison