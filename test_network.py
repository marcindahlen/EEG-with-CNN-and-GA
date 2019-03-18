from badanie import Badanie
from datastorage import Datastorage
import variables
import plotly.graph_objs

# b = Badanie(1)

# wyniki = b.forward_pass_all_networks()

data = Datastorage()

data.prepare_input()

data.show_summary()