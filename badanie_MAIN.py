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
            populacja.py
            .
            .
            .
            variables.py
        net_memory/

"""

from populacja import Populacja
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
    → https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
"""

"""
    → https://en.wikipedia.org/wiki/Data_stream_management_system#Synopses  !!!
    It is possible to use another method for data reading, other than windows,
    and given the nature of eeg it might be highly desirable to use it.
"""