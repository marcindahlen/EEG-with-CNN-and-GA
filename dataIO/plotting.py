import plotly as plot
import plotly.graph_objs as graph

#https://plot.ly/python/static-image-export/

"""
    x = [i for i in range(len(output_scores))]
            trace = graph_objs.Scatter(x=x, y=output_scores)
            plotly.offline.plot(trace, filename=variables.out_charts_path + 'name.html', auto_open=False) # @TODO filenaming needs parametrisation
"""

data_output_dir = "C:/marcin/psychologia_badania/EEG_csv_x32/out_wykresy"

def makeDotsChart(dane, dziedzina, nazwa):
    """Given the data and chart file name,
    method automates plotting simple scatter chart"""
    trace_A = graph.Scatter(
        x = dziedzina,
        y = dane,
        mode = 'markers'
    )
    plot_data = [trace_A]

    figure = graph.Figure(
        data=plot_data
    )

    #plot.offline.plot(figure, filename=data_output_dir + nazwa + '.html', auto_open=False)
    plot.io.write_image(figure, data_output_dir + nazwa + '.jpeg')
