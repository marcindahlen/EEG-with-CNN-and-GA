"""
Idealnie, wynikiem pojedynczego badania są:
→   plik graficzny ilustrujący postęp nauki
    kolejnych generacji sieci,
→   plik z przewidywanymi klasyfikacjami grupy testowej
    dokonanymi przez najlepszą sieć
    """
from networks import node_convolution
from utils import variables
from networks.network_LSTM import NeuralNetwork
import time


class Population(object):

    def __init__(self, examination_no):
        self.nodeType = nodeType
        self.examination_no = examination_no
        self.savedRMSEs = []

    def create_new(self):
        """"""
        if self.nodeType == node_convolution:
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


