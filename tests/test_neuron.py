from networks.numpy_neuron import NumpyNeuron
import numpy

data = [numpy.random.random() for x in range(100)]

neuron_1 = NumpyNeuron(100)
neuron_2 = NumpyNeuron(100)

output_1 = neuron_1.calculate(data)
output_2 = neuron_2.calculate(data)

print(output_1)
print(output_2)
