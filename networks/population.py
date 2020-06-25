"""
Idealnie, wynikiem pojedynczego badania są:
→   plik graficzny ilustrujący postęp nauki
    kolejnych generacji sieci,
→   plik z przewidywanymi klasyfikacjami grupy testowej
    dokonanymi przez najlepszą sieć
    """
from networks import convolution_node
from utils import variables
from networks.LSTM_network import NeuralNetwork
import time


class Population(object):

    def __init__(self, examination_no, datastorage, nodeType):
        self.nodeType = nodeType
        self.examination_no = examination_no
        #self.datastorage = datastorage
        self.savedRMSEs = []

    def create_new(self):
        """"""
        if self.nodeType == convolution_node:
            pass
        else:
            pass
        pass

    def get_network_code(self, network_no):
        """"""
        pass

    def save_network_code_toFile(self, network_code):
        """"""
        pass

    def forward_pass_all_nets(self, inputdata):
        """
        After receiving datapoints of all persons and channels
        feeds all networks and saves their RMSEs
        :param inputdata:
        :return: list of RMSE
        """
        pass

    def sort_networks_by_RMSEs(self):
        """"""
        pass

    def set_averages_for_RMSEs(self):
        """"""
        pass


