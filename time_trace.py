import numpy as np
import matplotlib.pyplot as plt

from src.classes import Data

from src.utils import save_tex_fig
from src.utils import ask_user_file
from src.utils import get_fourier
from src.utils.models import model_quant_V_trace
from src.utils.coefficient import get_specs, compute_A, compute_C

thickness, frequency, R_decade_box = ask_user_file()

data            = Data(f'data/{int(thickness)}um/SingleMeasurement_{int(frequency)}Hz_{int(R_decade_box)}Ohms.csv')
specs           = get_specs(f'{int(thickness)}um')

A               = compute_A(specs)
C               = compute_C(specs)

model_trace     = model_quant_V_trace(data.time, data.pressure,
                                A = A['nominal'], C = C['nominal'], R_decade_box = R_decade_box,
                                filter_window=301)

model_bounds     = np.array([
    model_quant_V_trace(data.time, data.pressure,
                                A = A['lower'], C = C['lower'], R_decade_box = R_decade_box,
                                filter_window=301),
    model_quant_V_trace(data.time, data.pressure,
                                A = A['lower'], C = C['upper'], R_decade_box = R_decade_box,
                                filter_window=301),
    model_quant_V_trace(data.time, data.pressure,
                                A = A['upper'], C = C['lower'], R_decade_box = R_decade_box,
                                filter_window=301),
    model_quant_V_trace(data.time, data.pressure,
                                A = A['upper'], C = C['upper'], R_decade_box = R_decade_box,
                                filter_window=301)
])

min_idx      = np.argwhere(data.time > 1/frequency)[0][0]
max_idx      = np.argwhere(data.time > 1/frequency*3)[0][0]
max_nb_points= 500
skip         = np.max([1, len(data.time[:max_idx])//max_nb_points])

plt.figure()
plt.plot(data.time[min_idx:max_idx:skip], data.voltage[min_idx:max_idx:skip], label="Measurement", color='black')
plt.plot(data.time[min_idx:max_idx:skip], model_trace[min_idx:max_idx:skip], label="Model", color='black', alpha=0.4)
plt.fill_between(data.time[min_idx:max_idx:skip], np.min(model_bounds, axis=0)[min_idx:max_idx:skip],
                    np.max(model_bounds, axis=0)[min_idx:max_idx:skip], 
                    color='black', alpha=0.1, label='Confidence interval Model')
plt.xlabel(rf"Time [s]")
plt.ylabel(r"$\hat{V}$ [V]")
plt.legend(frameon=False, loc='lower center', ncols=3, bbox_to_anchor=(0.5, 1.16))
plt.tight_layout()
plt.savefig(f"results/trace.png")
save_tex_fig(f"results/trace")
print("\033[93mSaved to results/trace\033[0m")