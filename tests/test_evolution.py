"""
→ https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/
"""

import numpy
from dataIO.datastorage_mockup import Datamockup
from populations.population import Populacja
import plotly.plotly
import plotly.graph_objs
from utils import variables


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def test_Hertz():
    print("Hertzs testing >>")
    nested_waves_lists = [[0 for j in range(256)] for i in range(10)]
    for i, x in enumerate(nested_waves_lists):
        for j, y in enumerate(x):
            nested_waves_lists[i][j] = 1 + 1j if i == j else 0
    for i, y in enumerate(nested_waves_lists):
        y = numpy.fft.ifft(y)
        x = [n for n in range(len(y))]
        trace_real = plotly.graph_objs.Scatter(x=x, y=numpy.real(y))
        trace_imag = plotly.graph_objs.Scatter(x=x, y=numpy.imag(y))
        plot_data = [trace_real, trace_imag]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + str(i) + "ifft" + '.html', auto_open=False)
    print("end testing <<")


data = Datamockup()
data.prepare_input()
data.normalise_channel_data()
data.prepare_target(5)

population = Populacja(5, data)

wynik = population.forward_pass_all_networks(8)
print(wynik)

population.evolve_network_generation()