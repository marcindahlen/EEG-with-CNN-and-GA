from dataIO.datastorage_channels import Datastorage

# x = [i for i in range(len(the_data))]
# trace = plotly.graph_objs.Scatter(x=x, y=the_data)
# plot_data = [trace]
# figure = plotly.graph_objs.Figure(data=plot_data)
# plotly.offline.plot(figure, filename=variables.out_charts_path + "badanie" + '.html', auto_open=False)

def loading_channels():
    data = Datastorage()
    data.load_channels()

    the_data = data.input_examined[3][1][100:200]
    print(the_data)

    return the_data


def test_answer():
    assert loading_channels() != False
