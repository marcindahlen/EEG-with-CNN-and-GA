from functools import reduce

import numpy

# https://stackoverflow.com/questions/58146901/how-to-use-only-max-average-pooling-layer-in-tensorflow-for-1d-array
import tensorflow

data_1d = numpy.random.rand(15)
data_1d = data_1d[None][:, :, None]
data_3d = numpy.random.rand(15, 15, 15)
data_3d = data_3d[None][:, :, None]

print(data_1d)
print(data_3d)
print("")

new = tensorflow.reshape(data_1d, 15)
print(new)
print("")

new = tensorflow.reshape(new, (1, 5, 3, 1))  # batch_shape + [height, width, channels]
print(new)
print("")
