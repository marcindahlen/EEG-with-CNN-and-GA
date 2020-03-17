"""

Idealnie, wynikiem pojedynczego badania są:
→   plik graficzny ilustrujący postęp nauki
    kolejnych generacji sieci,
→   plik z przewidywanymi klasyfikacjami grupy testowej
    dokonanymi przez najlepszą sieć,
→   plik (@TODO czy pliki?)
    binarny z zapisem stanu ostatniej generacji sieci.
"""
from utils import variables
from networks.LSTM_network import NeuralNetwork
import time


class Population(object):

    def __init__(self, examination_no, datastorage, nodeType):
        self.nodeType = nodeType
        self.examination_no = examination_no
        self.datastorage = datastorage

