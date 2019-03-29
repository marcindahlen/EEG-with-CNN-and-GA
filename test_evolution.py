"""
→ https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/
"""

import numpy
import math
import plotly.plotly
import plotly.graph_objs
import variables

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


sin = [math.sin(x) for x in frange(0, 1000, 0.1)]
count = len(sin)
xs = [x for x in range(count)]

f = numpy.fft.fft(sin)

print(sin)

print(f)

trace = plotly.graph_objs.Scatter(x=xs, y=sin)
plot_data = [trace]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "test_evo_sin" + '.html', auto_open=False)

y_fft_real = [x.real for x in f]
y_fft_imaginary = [x.imag for x in f]
trace_real = plotly.graph_objs.Scatter(x=xs, y=y_fft_real)
trace_imag = plotly.graph_objs.Scatter(x=xs, y=y_fft_imaginary)
plot_data = [trace_real, trace_imag]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "test_evo_fft" + '.html', auto_open=False)
