import pandas
import plotting
import fourier

in_path = "../in_raw/P06.txt"
data_output_dir = "../out_wykresy"

in_data = pandas.read_csv(in_path, index_col=False, header=0)
in_data = in_data.transpose()[0]

in_data = in_data.iloc[20000:]

print(in_data.index)

#rysuj_wykres_kropki(dane, dziedzina, nazwa)

frequency_step = fourier.fourier(in_data)

print(frequency_step)