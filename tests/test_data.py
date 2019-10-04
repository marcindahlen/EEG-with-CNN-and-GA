import pytest
from dataIO.datastorage_channels import Datastorage
from utils import variables
import plotly

class TestData:

    @pytest.fixture
    def load_filecontent(self):
        pass # refuses to work this way

    def test_loading(self):
        data = Datastorage()
        data.load_channels()
        particular_data = data.input_examined[3][2][100:200]
        print(particular_data)
        assert any(particular_data) != False

    def test_fourier_transform(self):
        data = Datastorage()
        data.load_channels()

        x = [i for i in range(len(data.input_examined[3][2]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[3][2])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataBeforeFourier" + '.html', auto_open=False)

        data.fourier_transform()

        particular_data = data.input_examined[3][2][100:200]
        x = [i for i in range(len(data.input_examined[3][2]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[3][2])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataBeforeFourier" + '.html', auto_open=False)

        print(particular_data)
        assert any(particular_data) != False

    def test_normalistaion(self):
        data = Datastorage()
        data.load_channels()
        data.fourier_transform()
        data.normalise_channel_data()
        particular_data = data.input_examined[3][2][100:200]
        print(particular_data)
        assert any(particular_data) != False