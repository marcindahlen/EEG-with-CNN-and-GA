import numpy
import pandas
import plotting

in_path = "../in_raw/P04.txt"
data_output_dir = "../out_wykresy"

in_data = pandas.read_csv(in_path, index_col=False, header=0, dtype='a')
in_data = in_data.transpose()[0]

in_data = in_data.iloc[20000:]

in_data = in_data.astype('float32')

print(in_data.index)

print(in_data.size)

print(in_data[-2])

if (in_data[5037].dtype == "float32"):
    print("1")
else:
    print("0")

mem = 0
for x in range(in_data.size):
    if (in_data[5037].dtype == "float32"):
        mem += 1

print(mem)
mem = mem / in_data.size
mem = mem * 100
print(mem)


#rysuj_wykres_kropki(dane, dziedzina, nazwa)