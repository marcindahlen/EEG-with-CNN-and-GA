
class IPopulation(object):
    """"""

    def __init__(self, nets_layers: list, nets_shape: list, data_source, data_target, test_data,
                 population_quantity: int):
        self.population_quantity = population_quantity
        self.test_data = test_data
        self.data_target = data_target
        self.data_source = data_source
        self.nets_shape = nets_shape
        self.nets_layers = nets_layers
        self.networks = self.init_networks(nets_layers, nets_shape, population_quantity)

    def init_networks(self, nets_layers, nets_shape, population_quantity) -> list:
        """"""
        pass

    def forward_pass(self):
        """"""
        pass

    def sort_by_score(self):
        """"""
        pass

    def evolve(self):
        """"""
        pass

    def get_best_rmse(self):
        """"""
        pass
