"""
I assume following files hierarchy:
    root/
        in_raw/
            P01.txt
            .
            .
            .
            P32.txt
        in_raw_channels/
            P01CH01
            P01CH02
            .
            .
            .
            P32Ch14
        out_raw/
            main_alpha-index_base.xls
            out_absData.xls
        out_wykresy/

        python/
            dataIO/
            networks/
            tests/
            utils/
            notepad.py
            main.py
        net_memory/

"""

"""
"The total economic burden of MDD (major depression) is now estimated to be $210.5 billion per year."
- http://www.workplacementalhealth.org/Mental-Health-Topics/Depression/Quantifying-the-Cost-of-Depression
+ https://onlinelibrary.wiley.com/doi/full/10.1002/wps.20692
"""

"""
    → https://superuser.com/questions/679679/how-to-increase-pythons-cpu-usage
    → https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
"""

"""
    → https://en.wikipedia.org/wiki/Data_stream_management_system#Synopses  !!!
    It is possible to use another method for data reading, other than windows,
    and given the nature of eeg it might be highly desirable to use it.
    
    The idea behind compression techniques is to maintain only a synopsis of the data, 
    but not all (raw) data points of the data stream. The algorithms range from selecting 
    random data points called sampling to summarisation using histograms, wavelets 
    or sketching. One simple example of a compression is the continuous calculation 
    of an average. Instead of memorizing each data point, the synopsis only holds 
    the sum and the number of items. The average can be calculated by dividing 
    the sum by the number. However, it should be mentioned that synopses cannot reflect 
    the data accurately. Thus, a processing that is based on synopses may produce inaccurate results.
"""

"""
    I can use two types of neuron: classic and LSTM (/GRUs)
    But LSTMs are pointless in one-time-look on data, they don't have occasion to remember anything.
    
    What about LSTM gives final output after looking at all channels?
    
    There are some cases:
    neurons: 
            classic vs LSTM (vs GRU)
    viewpoint:  
            single whole channel → verdict; 
            few whole channels → verdict; 
            scanning channel in parts → verdict;
            scanning channels in parallel → verdict;
    output (classification):
            a number;
            a vector;
    A nice table of comparison emerges from the above:
            multichannel simple convolution 10outputs            (data → convolution → pooling → ANN → output vec)
            multichannel simple convolution singleOutput         (data → convolution → pooling → ANN → single number)
            multichannel simple noConv 10outputs                 (data → pooling → ANN → output vec)
            multichannel simple noConv singleOutput              (data → pooling → ANN → single number)
            multichannel lstm convolution 10outputs              (data → convolution → data-chunks → lstm step-by-step → output vec)
            multichannel lstm convolution singleOutput           (data → convolution → data-chunks → lstm step-by-step → single number)
            multichannel lstm noConv 10outputs                   (data → pooling → lstm step-by-step → output vec)
            multichannel lstm noConv singleOutput                (data → pooling → lstm step-by-step → single number)
            
        lstm step-by-step → a smaller net than classic ANN, used window after window on the input
        
        The learning method is genetic algorithm, ideally besides training weights it also optimises pooling size and method.
        
    Outputs could be evaluated in many different ways.
    There could be 10 output neurons, each one responsible for its own range of correctness:
    for real output 14, neuron no. 2 should fire, rest should be 'silent'.
    But there could be also a single output neuron giving answers in its range <0, 1> where
    real output 14 would be a 0.14 answer. 
    Second approach could be more "trainable".
    This adds new depth to "a nice table of comparison".
    
    →https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
"""

"""
    Additional comparison trained network vs untrained network.
"""

"""
Decide to use 3-fold cross validation vs 6-fold cross validation (shuffling!)
→ a bias-variance trade-off associated with the choice of k in k-fold cross-validation. 
Typically, given these considerations, one performs k-fold cross-validation using k = 5 or k = 10, 
as these values have been shown empirically to yield test error rate estimates that suffer neither 
from excessively high bias nor from very high variance.

Citation about estimating network performance on larger data sets most welcome.
"""