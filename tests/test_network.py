from networks.population import Populacja
from dataIO.datastorage import Datastorage

data = Datastorage()

data.prepare_input()

data.prepare_target(1)
b = Populacja(1, data)

print("Assumed iterations: " + str(data.assume_networkIterationsNo()))

wyniki = b.forward_pass_all_networks(data.assume_networkIterationsNo())
print(wyniki)

b.evolve_network_generation()

wyniki = b.forward_pass_all_networks(data.assume_networkIterationsNo())
print(wyniki)
