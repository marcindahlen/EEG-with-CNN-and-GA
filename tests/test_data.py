from networks.population import Populacja
from dataIO.datastorage import Datastorage
from utils import variables
import plotly.graph_objs


data = Datastorage()
data.prepare_input()

b = Populacja(0, data)

the_data = data.input_examined['P08.txt'][16]
print(the_data)

the_data = b.output_examined['P08.txt']
print(the_data)

x = [i for i in range(len(the_data))]
trace = plotly.graph_objs.Scatter(x=x, y=the_data)
plot_data = [trace]
figure = plotly.graph_objs.Figure(data=plot_data)
plotly.offline.plot(figure, filename=variables.out_charts_path + "badanie" + '.html', auto_open=False)
