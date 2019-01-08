#@TODO testy fouriera forward + inverse

import cmath
import fourier

test_data = []

for x in range(0, 100):
    test_data.append(cmath.sin(x))

print(test_data)

test_data = fourier.Fourier.fft(test_data)

print(test_data)
