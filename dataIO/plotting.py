from matplotlib import pyplot

from utils import variables


def make_single_dots_chart(name, data):
    """Given the data and chart file name,
    method automates plotting simple scatter chart"""
    out_path = variables.out_charts_path + name + '.jpg'
    domain = [i for i in range(len(data))]
    pyplot.plot(domain, data, 'o')
    pyplot.savefig(out_path)


def make_multi_chart(name, data=[]):
    pass

