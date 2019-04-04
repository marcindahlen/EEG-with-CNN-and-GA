#@TODO testy fouriera forward + inverse + plotly


import numpy
import plotly.plotly
import plotly.graph_objs
import variables
import os

test_data = numpy.genfromtxt(variables.in_raw_path + 'P07.txt', delimiter=',')
test_data = numpy.delete(test_data, variables.how_many_to_drop, axis=None)
x = [x for x in range(0, len(test_data))]

trace = plotly.graph_objs.Scatter(x=x, y=numpy.real(test_data))
plot_data = [trace]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "test" + '.html', auto_open=False)

test_data = numpy.fft.fft(test_data)
trace = plotly.graph_objs.Scatter(x=x, y=numpy.real(test_data))
plot_data = [trace]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "fft" + '.html', auto_open=False)


test_data = numpy.delete(test_data, variables.how_many_to_drop, axis=None)
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

print("nested >>")
nested_waves_lists = [[0 for j in range(256)] for i in range(10)]
for i, x in enumerate(nested_waves_lists):
    for j, y in x:
        nested_waves_lists[i][j] = 1+1j if i == j else 0
for y in nested_waves_lists:
    y = numpy.fft.ifft(y)
    x = [n for n in range(len(y))]
    trace_real = plotly.graph_objs.Scatter(x=x, y=numpy.real(y))
    trace_imag = plotly.graph_objs.Scatter(x=x, y=numpy.imag(y))
    plot_data = [trace_real, trace_imag]
    figure = plotly.graph_objs.Figure(data=plot_data)
    plotly.offline.plot(figure, filename=variables.out_charts_path + str(y) + "ifft" + '.html', auto_open=False)



