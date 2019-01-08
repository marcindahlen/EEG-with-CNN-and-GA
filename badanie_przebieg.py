"""Klasa opisuje przebieg pojedynczego badania.
Pojedyncze badanie to próba nauki sieci neuronowej
(unsupervised - genetic algorithm)
klasyfikacji wejściowego strumienia danych EEG do
jednej z ocen w danym badaniu testowym. Przykładowo,
dla każdego pliku wejściowego danych z EGG tworzona
i uczona jest osobna sieć dla testów takich jak RavenSPP
czy IVE-Impulsywnosc.

Idealnie, wynikiem pojedynczego badania są:
plik graficzny ilustrujący postęp nauki
kolejnych generacji sieci,
plik z przewidywanymi klasyfikacjami grupy testowej
dokonanymi przez najlepszą sieć,
plik (@TODO czy pliki?)
binarny z zapisem stanu ostatniej generacji sieci."""

import pandas


class Badanie(object):

    def __init__(self, input_csv_folder_path, target_data_path):
        pass

    def prepare_input(self, input_csv_path):
        """From raw csv trim useless frequencies.
        → https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.tolist.html
        :return pandas labeled dataframe
        """
        pass

    def prepare_target(self, target_data_path):
        """From excel file with two columns:
        [[badany]] and [[test_score]]
        → https://www.mantidproject.org/Working_With_Functions:_Return_Values
        :return minmaxed pandas series as train data
                AND second pandas series as test data
                """
        pass

    def initialize_networks(self):
        """Create if not exists a new binary files (plural!!)
        → https://stackoverflow.com/questions/807863/how-to-output-list-of-floats-to-a-binary-file-in-python
        describing initial wages for
        LSTM networks to be trained.
        Number of files (population) should
        correspond to number of target classes.
        First check if there is a file,
        if not → initialize networks and save states
        if yes → load states from files
        :return LSTM_network list
        """
        pass

    def single_pass_one_network(self, network):
        """Given a single network from the list
        it is run on all examined person's EEG,
        procuring an <0, 1> output for each (EEG).
        Produced list of outputs is compared with
        a list of corresponding target values
        (new list item = target_val - network_output).
        Then squares of new list items are summed.
        The bigger is final number, the worse was
        network's overall performance.
        :return a single float - network's rmse
        """
        pass

    def forward_pass_all_networks(self, networks):
        """On given list of networks
        (all networks of same type and size,
        they vary only in weights)
        each network calculates in single pass
        predicted classification of each
        examined's EEG.
        One networks performs as many single-passes
        as there are examined persons.

        Each network has a field to store
        it's mean performance
        measured as RMSE - root-mean-square error.

        :return one list with rmse scores, position
                on this list reference position of network
                and thus order of single-pass
        """
        pass

    def repopulate_network_generation(self, networks):
        """ @TODO czy sieć nie powinna sama przechowywać swojego score?
        → https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects
        Given all networks were scored based on their
        performance and remember their score,
        the list containing them is sorted
        based on their score, from lowest to highest.
        The worsen [[part]] of networks is deleted and the
        list is repopulated with children of saved networks
        (pair multiplication includes mutation)
        and [[some amount]] of mutated saved networks (budding),
        and [[some amount]] of new random ones
        → https://en.wikipedia.org/wiki/Budding
        → https://unix.stackexchange.com/questions/196251/change-only-one-bit-in-a-file
        → https://www.hackerearth.com/practice/basic-programming/bit-manipulation/basics-of-bit-manipulation/tutorial/
        @TODO a może jednak nie binary a txt ← wtedy mutacja to random na [[jednej]] wadze
        @TODO flipping bits in binary files wymaga eksperymentów

        :return networks
        """
        pass
