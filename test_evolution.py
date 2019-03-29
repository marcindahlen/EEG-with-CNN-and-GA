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

f = numpy.fft.fft(sin)

print(sin)

print(f)