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


recipe = [1 if 7 <= x <= 12 else 0 for x in range(2056)]
count = len(recipe)
xs = [x for x in range(count)]

f = numpy.fft.ifft(recipe)

print(recipe)

print(f)

y_fft_real = [x.real for x in f]
y_fft_imaginary = [x.imag for x in f]
trace_real = plotly.graph_objs.Scatter(x=xs, y=y_fft_real)
trace_imag = plotly.graph_objs.Scatter(x=xs, y=y_fft_imaginary)
plot_data = [trace_real, trace_imag]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "test_evo_sin" + '.html', auto_open=False)

y_fft_real = [x.real for x in recipe]
y_fft_imaginary = [x.imag for x in recipe]
trace_real = plotly.graph_objs.Scatter(x=xs, y=y_fft_real)
trace_imag = plotly.graph_objs.Scatter(x=xs, y=y_fft_imaginary)
plot_data = [trace_real, trace_imag]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "test_evo_fft" + '.html', auto_open=False)
