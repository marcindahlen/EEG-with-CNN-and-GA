import pytest
from dataIO.datastorage import Datastorage
from utils import variables
import plotly

class TestData:

    pytest.person_no = variables.people_to_consider[0]
    pytest.channel_no = variables.channels_to_consider[0]

    @pytest.fixture
    def load_filecontent(self):
        pass # refuses to work this way

    def test_loading(self):
        data = Datastorage()
        data.load_channels()

        x = [i for i in range(len(data.input_examined[pytest.person_no][pytest.channel_no]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[pytest.person_no][pytest.channel_no])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataAfterLoading" + '.html',
                            auto_open=False)

        particular_data = data.input_examined[pytest.person_no][pytest.channel_no][100:200]
        print(particular_data)
        assert any(particular_data) != False

    def test_standardisation(self):
        data = Datastorage()
        data.load_channels()

        x = [i for i in range(len(data.input_examined[pytest.person_no][pytest.channel_no]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[pytest.person_no][pytest.channel_no])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataAfterStandardisation" + '.html',
                            auto_open=False)

        particular_data = data.input_examined[pytest.person_no][pytest.channel_no][100:200]
        assert any(particular_data) != False

    def test_fourier_transform(self):
        data = Datastorage()
        data.load_channels()
        data.fourier_transform()

        x = [i for i in range(len(data.input_examined[pytest.person_no][pytest.channel_no]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[pytest.person_no][pytest.channel_no])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataAfterFourier" + '.html', auto_open=False)

        particular_data = data.input_examined[pytest.person_no][pytest.channel_no][100:200]
        print(particular_data)
        assert any(particular_data) != False

    def test_normalisation(self):
        data = Datastorage()
        data.load_channels()
        data.fourier_transform()
        data.normalise_channel_data()

        x = [i for i in range(len(data.input_examined[pytest.person_no][pytest.channel_no]))]
        trace = plotly.graph_objs.Scatter(x=x, y=data.input_examined[pytest.person_no][pytest.channel_no])
        plot_data = [trace]
        figure = plotly.graph_objs.Figure(data=plot_data)
        plotly.offline.plot(figure, filename=variables.out_charts_path + "testDataAfterNormalisation" + '.html',
                            auto_open=False)

        particular_data = data.input_examined[pytest.person_no][pytest.channel_no][100:200]
        print(particular_data)
        assert any(particular_data) != False


    def test_summary(self):
        data = Datastorage()
        data.load_channels()
        data.fourier_transform()

        data.prepare_inputdata_insights()
        data.print_inputdata_insights()

        data.normalise_channel_data()