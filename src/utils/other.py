import os
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils import extract_files_dir

def ask_user_file():
    """
    Ask the user for the thickness, frequency and resistance to define the file to load.
    """
    folders   = np.array(np.sort([int(file[:-2]) for file in os.listdir('data') if not file.startswith('.')]), dtype=str)
    thickness = input("\033[92mEnter the thickness in um\033[0m\nChoose between: " + ", ".join(folders) + "\n")

    _, _, _, frequencies, resistances = extract_files_dir(f"data/{str(thickness)}um")

    frequencies = np.array(np.int64(np.unique(frequencies)), dtype=str)
    resistances = np.array(np.int64(np.unique(resistances)/1000), dtype=str)

    frequency = input("\033[92mEnter the frequency in Hz\033[0m\nChoose between: " + ", ".join(frequencies) + "\n")
    resistance = input("\033[92mEnter the resistance in kOhms\033[0m\nChoose between: " + ", ".join(resistances) + "\n")

    return float(thickness), float(frequency), float(resistance)*1e3

def ask_user_folder():
    """
    Ask the user for the thickness to define which folder to use.
    """
    folders   = np.array(np.sort([int(file[:-2]) for file in os.listdir('data') if not file.startswith('.')]), dtype=str)
    thickness = input("\033[92mEnter the thickness in um\033[0m\nChoose between: " + ", ".join(folders) + "\n")

    return float(thickness)

def save_tex_fig(filename):
    """
    Save the current plot to a .tex file.
    """
    def tikzplotlib_fix_ncols(obj):
        """
        workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
        """
        if hasattr(obj, "_ncols"):
            obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)

    tikzplotlib_fix_ncols(plt.gcf())

    Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
    Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])

    tikzplotlib.save(filename + ".tex")
