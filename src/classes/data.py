import numpy as np

from src.utils import demodulate
from src.utils import get_frequency
from src.utils import R_effective
from src.utils import extract_file, extract_files_dir

class Data:
    def __init__(self, file_name) -> None:
        data = extract_file(file_name)

        self.time, self.pressure, self.voltage = data
        self.frequency = get_frequency(self.time)

class DataSet:
    def __init__(self, directory) -> None:
        time, pressure, voltage, self.frequencies, self.resistances = extract_files_dir(directory)
        self.resistances = R_effective(self.resistances)

        # Demodulate the data to extract the amplitude at the fundamental frequency
        self.pressure_p2third = np.array([demodulate(t, np.power(p, 2/3), freq) for t, p, freq in zip(time, pressure, self.frequencies)])
        self.voltage  = np.array([demodulate(t, V, freq) for t, V, freq in zip(time, voltage, self.frequencies)])

        # Split the data into arrays of the same frequency
        self.frequencies = np.split(self.frequencies, np.unique(self.frequencies, return_index=True)[1][1:])
        self.resistances = np.split(self.resistances, np.unique(self.frequencies, return_index=True)[1][1:])
        self.pressure_p2third    = np.split(self.pressure_p2third, np.unique(self.frequencies, return_index=True)[1][1:])
        self.voltage     = np.split(self.voltage, np.unique(self.frequencies, return_index=True)[1][1:])