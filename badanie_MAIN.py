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