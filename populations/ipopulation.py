class IPopulation(object):
    """"""

    def __init__(self, examination_no):
        self.nodeType = nodeType
        self.examination_no = examination_no
        self.savedRMSEs = []

    def get_network_code(self, network_no):
        """"""
        pass

    def save_network_code_tofile(self, network_code):
        """"""
        pass

    def evolve(self):
        """"""
        pass

    def calculate(self, inputdata):
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