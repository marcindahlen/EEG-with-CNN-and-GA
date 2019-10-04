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

        x = [i for i in range(len(data.input_examined[3][2]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[3][2])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataAfterLoading" + '.html',
                            auto_open=False)

        particular_data = data.input_examined[3][2][100:200]
        print(particular_data)
        assert any(particular_data) != False

    def test_standardisation(self):
        data = Datastorage()
        data.load_channels()
        data.standardise_channel_data()

        x = [i for i in range(len(data.input_examined[3][2]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[3][2])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataAfterStandardisation" + '.html',
                            auto_open=False)

        particular_data = data.input_examined[3][2][100:200]
        assert any(particular_data) != False

    def test_fourier_transform(self):
        data = Datastorage()
        data.load_channels()
        data.standardise_channel_data()
        data.fourier_transform()

        x = [i for i in range(len(data.input_examined[3][2]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[3][2])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataAfterFourier" + '.html', auto_open=False)

        particular_data = data.input_examined[3][2][100:200]
        print(particular_data)
        assert any(particular_data) != False

    def test_normalisation(self):
        data = Datastorage()
        data.load_channels()
        data.standardise_channel_data()
        data.fourier_transform()
        data.normalise_channel_data()

        x = [i for i in range(len(data.input_examined[3][2]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[3][2])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataAfterNormalisation" + '.html',
                            auto_open=False)

        particular_data = data.input_examined[3][2][100:200]
        print(particular_data)
        assert any(particular_data) != False


    def test_summary(self):
        data = Datastorage()
        data.load_channels()
        data.standardise_channel_data()
        data.fourier_transform()
        data.normalise_channel_data()
        data.prepare_inputdata_insights()

        data.print_inputdata_insights()