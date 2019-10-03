from dataIO.datastorage_channels import Datastorage

# x = [i for i in range(len(the_data))]
# trace = plotly.graph_objs.Scatter(x=x, y=the_data)
# plot_data = [trace]
# figure = plotly.graph_objs.Figure(data=plot_data)
# plotly.offline.plot(figure, filename=variables.out_charts_path + "badanie" + '.html', auto_open=False)

class test_data():
    def __init__(self):
        self.data = Datastorage()
        self.data.load_channels()

    def test_loading(self):
        the_data = self.data.input_examined[1][1][100:200]
        print(the_data)
        assert the_data != False
