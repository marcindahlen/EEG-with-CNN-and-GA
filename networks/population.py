"""
The class describes a single examination.

A single examination is a process of unsupervised
training of one LSTM neural network which takes
as input alpha waves recording and predict
the classification to a particular psychology test.
For each feature of each test there should
be separate network to classify examined's EEG.
The total number of features is 17.
Each network is trained by genetic algorithm.
That means, many networks are spawn to select
those with best performance and evolve them
further.

Idealnie, wynikiem pojedynczego badania są:
→   plik graficzny ilustrujący postęp nauki
    kolejnych generacji sieci,
→   plik z przewidywanymi klasyfikacjami grupy testowej
    dokonanymi przez najlepszą sieć,
→   plik (@TODO czy pliki?)
    binarny z zapisem stanu ostatniej generacji sieci.
"""
from utils import variables
from networks.numpy_network import NeuralNetwork
import time


class Populacja(object):

    def __init__(self, examination_no, datastorage):                                             # examination_no is the index of column in target data
        self.examination_no = examination_no
        self.examination_names = {0: 'SPP',
                                  1: 'SPH',
                                  2: 'RPN',
                                  3: 'Raven A',
                                  4: 'Raven B',
                                  5: 'Raven C',
                                  6: 'Raven D',
                                  7: 'Raven E',
                                  8: 'Raven WO',
                                  9: 'IVE Impulsywnosc',
                                  10: 'IVE Ryzyko',
                                  11: 'IVE Empatia',
                                  12: 'SSZ',
                                  13: 'SSE',
                                  14: 'SSU',
                                  15: 'ACZ',
                                  16: 'PKT'}
        print('Populacja ' + self.examination_names[self.examination_no])
        self.db = datastorage
        start = time.time()
        time.clock()
        print('   Inicjalizacja populacji sieci', end="... ")
        self.network_list = [NeuralNetwork(examination_no) for i in range(variables.population_quantity)]
        print("zakończona po " + str(int(time.time() - start)) + "s")

    def forward_pass_all_networks(self, iterations_no):
        """
        For each network in [[self.network_list]]
        perform single forward pass over
        available examined persons' data.
        Evaluate all networks and sort
        list containing them accordingly.
        → https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects

        :return sorted list @TODO or void?
        """
        if not self.db.isDataInitialised:
            raise Exception('No data was loaded, see dataBase.prepare_input()!')

        print("   Sieci klasyfikują", end="... ")
        output_scores = []                          # list of ints
        counter = 0
        start = time.time()
        time.clock()
        for network in self.network_list:
            network.forward_pass(self.db.input_examined, iterations_no)
            output_scores.append(network.evaluate_self(self.db.output_examined))
            counter += 1
            percent = int((counter / len(self.network_list)) * 100)
            print(str(percent) + '%', end=' ', flush=True)

        print("iteracja trwała " + str(int(time.time() - start)) + "s")

        self.network_list.sort(key=lambda network: network.score, reverse=True)

        return output_scores.sort(reverse=True)

    def evolve_network_generation(self):
        """
        Given all networks were scored based on their
        performance and remember their score,
        this method makes changes to the list of networks.
        First 4 networks are saved.
        Other networks are deleted.
        First 2 networks generate offspring - as many as needed to fill
        up half of population cap.
        Other half is created by budding of 4 initial networks.
        Two new networks are added above the limit to introduce diversity.

        → https://en.wikipedia.org/wiki/Budding
        → https://unix.stackexchange.com/questions/196251/change-only-one-bit-in-a-file
        → https://www.hackerearth.com/practice/basic-programming/bit-manipulation/basics-of-bit-manipulation/tutorial/
        @TODO a może jednak nie binary a txt ← wtedy mutacja to random na [[jednej]] wadze
        @TODO flipping bits in binary files wymaga eksperymentów

        :return void
        """
        print("   Ewolucja sieci", end='... ')
        self.network_list = self.network_list[:4]

        while len(self.network_list) <= variables.population_quantity / 2:
            if len(self.network_list) % 2 == 0:
                self.network_list.append(self.network_list[0].create_single_child(self.network_list[1]))
            else:
                self.network_list.append(self.network_list[1].create_single_child(self.network_list[0]))

        while len(self.network_list) <= variables.population_quantity:
            self.network_list.append(self.network_list[0].multiplication_by_budding())
            self.network_list.append(self.network_list[1].multiplication_by_budding())
            self.network_list.append(self.network_list[2].multiplication_by_budding())
            self.network_list.append(self.network_list[3].multiplication_by_budding())

        self.network_list.append(NeuralNetwork(self.examination_no))
        self.network_list.append(NeuralNetwork(self.examination_no))
        print("zakończona.")

