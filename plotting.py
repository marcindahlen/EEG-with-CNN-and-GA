import plotly as plot
import plotly.graph_objs as graph

#https://plot.ly/python/static-image-export/

data_output_dir = "C:/marcin/psychologia_badania/EEG_csv_x32/out_wykresy"

def rysuj_wykres_kropki(dane, dziedzina, nazwa):
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