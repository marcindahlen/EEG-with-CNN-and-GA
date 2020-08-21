import matplotlib


"""
    x = [i for i in range(len(output_scores))]
            trace = graph_objs.Scatter(x=x, y=output_scores)
            plotly.offline.plot(trace, filename=variables.out_charts_path + 'name.html', auto_open=False) # @TODO filenaming needs parametrisation
"""

data_output_dir = "C:/marcin/psychologia_badania/EEG_csv_x32/out_wykresy"

def makeDotsChart(dane, dziedzina, nazwa):
    """Given the data and chart file name,
    method automates plotting simple scatter chart"""
