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
import math
import variables
from numpy_network import NeuralNetwork
import pandas
import time


class Populacja(object):

    def __init__(self, examination_no, datastorage):                                             # examination_no is the index of column in target data
        self.examination_no = examination_no
        self.output_examined = dict()
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
        self.prepare_target(examination_no)
        self.minmax_tuple = ()
        start = time.time()
        time.clock()
        print('   Inicjalizacja populacji sieci', end="... ")
        self.network_list = [NeuralNetwork(examination_no) for i in range(variables.population_quantity)]
        print("zakończona po " + str(int(time.time() - start)) + "s")

    def prepare_target(self, examination_no):
        """
        From excel file with columns:
        badany,	SPP,	SPH,	RPN,	Raven_A,	Raven_B,	Raven_C,	Raven_D,	Raven_E,	Raven_WO,	IVE_Impulsywnosc,	IVE_Ryzyko,	IVE_Empatia,	SSZ,	SSE,	SSU,	ACZ,	PKT
        read data,

        → https://www.mantidproject.org/Working_With_Functions:_Return_Values
        :return void
        """
        print('   Wczytywanie danych wzorcowych', end='...')
        target_data = pandas.read_excel(variables.out_raw_filepath)
        target_data = target_data.iloc[2:, :]                         # delete P01BA and P01BB rows
        target_data = target_data.iloc[:, examination_no + 1].values

        minimum = min(target_data)
        maximum = max(target_data)
        self.minmax_tuple = (minimum, maximum)
        for i in range(len(target_data)):
            target_data[i] = (target_data[i] - minimum) / (maximum - minimum)

        for index, file in enumerate(self.db.files_list):
            self.output_examined[file] = [self.decide_no_belonging(target_data[index], x) for x in range(10)]

        # at this point i have a dictionary with filenames as keys containing a list as value.
        # The list contain ten values → 0 or 1, where 1 means the original value was in corresponding range
        # i.e. 39 becomes 0.18 when minmaxed* in <32, 71> → which becomes [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        #   *look "at this point data is normalised in <0, 1>" above or → https://en.wikipedia.org/wiki/Feature_scaling
        print(' zakończone.')

    def decide_no_belonging(self, number, index):
        """
        Given minmaxed value of examined's test score
        and currently considered index in list being build,
        outputs 1 or 0 where 1 is a valid match
        :param number:
        :param index:
        :return: int: 1 or 0
        """
        index = index / 10
        return 1 if index <= number < index + 0.1 else 0

    def interprete_prediction(self, prediction):
        """
        Given list with 10 elements - 9 zeros and 1 one,
        this method converts information from the list
        to the form as in raw input (reverse process from prepare_target()
        :return: int
        """
        memory = 0
        for i, x in enumerate(prediction):
            memory += x * (i + 1) / 10
        memory = memory * (self.minmax_tuple[1] - self.minmax_tuple[0]) + self.minmax_tuple[0]

        return memory

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
            output_scores.append(network.evaluate_self(self.output_examined))
            counter += 1
            percent = int((counter / len(self.network_list)) * 100)
            print(str(percent) + '%', end=' ', flush=True)

        print("iteracja trwała " + str(int(time.time() - start)) + "s")

        self.network_list.sort(key=lambda network: network.score, reverse=True)

        # print("   Wyniki bieżącej populacji sieci: " + str(output_scores))
        return output_scores


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

