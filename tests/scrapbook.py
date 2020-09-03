import numpy as np

# https://stackoverflow.com/questions/58146901/how-to-use-only-max-average-pooling-layer-in-tensorflow-for-1d-array

a = np.random.randn(4).astype('float32')
print(a)
print("  ")
print("  ")

bac = a[None][:, :, None]
print(bac)
print("  ")
print("  ")

cbd = a[None][None, None, :, :, None]
print(cbd)
print("  ")
print("  ")

a = np.random.randn(4, 4, 4).astype('float32')
print(a)
print("  ")
print("  ")

bac = a[None][:, :, None]               # addin two dims
print(bac)
print("  ")
print("  ")

cbd = a[None][None, None, :, :, None]   # adding 4 dims
print(cbd)
