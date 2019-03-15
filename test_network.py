from badanie import Badanie
import variables
import plotly.graph_objs

b = Badanie(1)

wyniki = b.forward_pass_all_networks()

print(wyniki)