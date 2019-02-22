#@TODO testy fouriera forward + inverse

import cmath
#import fourier
import numpy

test_data = []

for x in range(0, 100):
    test_data.append(cmath.sin(x/100*cmath.pi))

print(test_data)
print()
print()

test_data = numpy.fft.fft(test_data)

print(test_data)
print()
print()

test_data = numpy.fft.ifft(test_data)

print(test_data)
print()
print()
