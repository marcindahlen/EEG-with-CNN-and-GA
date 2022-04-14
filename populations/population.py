from networks.network import Network
from populations.ipopulation import IPopulation


class Population(IPopulation):
    def __init__(self, nets_layers: list, nets_shape: list, data_source, data_target: list, test_data,
                 population_quantity: int):
        self.population_quantity = population_quantity
        self.test_data = test_data
        self.data_target = data_target
        self.data_source = data_source
        self.nets_shape = nets_shape
        self.nets_layers = nets_layers
        self.networks = self.init_networks(nets_layers, nets_shape, population_quantity)
        self.current_target = data_target[0]
        self.outputs = list()
        self.rmse_list = list()

    def init_networks(self, nets_layers, nets_shape, population_quantity) -> list:
        return [Network(nets_layers, nets_shape) for i in range(population_quantity)]

    def forward_pass(self):
        pass

    def sort_by_score(self):
        self.rmse_list.append(list())
        for net in self.networks:
            net.evaluate_self(self.current_target)
        self.networks = sorted(self.networks, key=lambda net: net.get_score())
        for net in self.networks:
            self.rmse_list[-1].append(net.get_score())

    def evolve(self):
        pass

    def get_best_rmse(self):
        self.sort_by_score()
        newest_rmse = self.rmse_list[-1]
        newest_rmse = sorted(newest_rmse)
        return newest_rmse[-1]

