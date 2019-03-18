"""
Main node of the experiment.

I assume following files hierarchy:
    root/
        in_raw/
            P01.txt
            .
            .
            .
            P32.txt
        out_raw/
            main_alpha-index_base.xls
            out_absData.xls
        out_wykresy/

        python/
            badanie.py
            .
            .
            .
            variables.py
        net_memory/

"""

from badanie import Badanie
import variables
from plotly import graph_objs
import plotly.plotly

"""
x = [i for i in range(len(output_scores))]
        trace = graph_objs.Scatter(x=x, y=output_scores)
        plotly.offline.plot(trace, filename=variables.out_charts_path + 'name.html', auto_open=False) # @TODO filenaming needs parametrisation
"""

"""
    → https://superuser.com/questions/679679/how-to-increase-pythons-cpu-usage
"""