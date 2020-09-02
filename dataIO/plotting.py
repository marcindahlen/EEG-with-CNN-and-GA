import matplotlib
import matplotlib.pyplot

from utils import variables


def make_single_dots_chart(nazwa, dane):
    """Given the data and chart file name,
    method automates plotting simple scatter chart"""
    out_path = variables.out_charts_path + nazwa + '.jpg'
    domain = [i for i in range(len(dane))]
    pyplot.plot(domain, dane, 'o')
    pyplot.savefig(out_path)


def make_multi_chart(nazwa, dane=[]):
    pass

