#@TODO testy fouriera forward + inverse + plotly


import numpy
import plotly.plotly
import plotly.graph_objs
import variables
import os

test_data = numpy.genfromtxt(variables.in_raw_path + 'P07.txt', delimiter=',')
test_data = numpy.delete(test_data, variables.how_many_to_drop, axis=None)
x = [x for x in range(0, len(test_data))]

"""
x = [x/100*cmath.pi for x in range(0, 1000)]
for y in x:
    test_data.append(cmath.sin(y))
"""

trace = plotly.graph_objs.Scatter(x=x, y=numpy.real(test_data))
plot_data = [trace]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "sin" + '.html', auto_open=False)

test_data = numpy.fft.fft(test_data)
trace = plotly.graph_objs.Scatter(x=x, y=numpy.real(test_data))
plot_data = [trace]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "fft" + '.html', auto_open=False)


test_data = numpy.fft.ifft(test_data)
trace = plotly.graph_objs.Scatter(x=x, y=numpy.real(test_data))
plot_data = [trace]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "ifft" + '.html', auto_open=False)
del test_data

files_list = [name for name in os.listdir(variables.in_raw_path)]
print(files_list)
order = 0


def process_file(filename):
    data = numpy.genfromtxt(variables.in_raw_path + filename, delimiter=',')
    data = numpy.delete(data, variables.how_many_to_drop, axis=None)
    x = [x for x in range(0, len(data))]
    trace = plotly.graph_objs.Scatter(x=x, y=numpy.real(data))
    plot_data = [trace]
    figure = plotly.graph_objs.Figure(data=plot_data)
    plotly.offline.plot(figure, filename=variables.out_charts_path + filename + '.html', auto_open=False)


for file in files_list:
    process_file(file)

