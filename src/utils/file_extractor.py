import re
import os
import numpy as np
import re
import os
import numpy as np

def extract_file(file_path):
    data = np.loadtxt(file_path, comments='#', delimiter=',', skiprows=1, usecols=[0, 1, 2], dtype=float).T
    data[1] *= 1e5 #Convert bar to Pascal
    return data[:3]

def extract_files_dir(directory):
    loaded_data = []
    frequencies = []
    resistances = []
    
    for file_name in [file for file in os.listdir(directory) if not file.startswith('.')]:
        # Check if the file name matches the expected pattern
        match = re.search(r'(\d+)Hz_(\d+)Ohms(_\d+)?\.csv', file_name)
        if match:
            frequency, resistance, _ = match.groups()
            frequencies.append(float(frequency))
            resistances.append(float(resistance))

            file_path = os.path.join(directory, file_name)
            loaded_data.append(extract_file(file_path))

    sort_indices = np.lexsort((resistances, frequencies))
    time, pressure, voltage = np.swapaxes(np.array(loaded_data)[sort_indices], 0, 1)
    frequencies  = np.array(frequencies)[sort_indices]
    resistances  = np.array(resistances)[sort_indices]
    
    return time, pressure, voltage, frequencies, resistances