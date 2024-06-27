import numpy as np
import matplotlib.pyplot as plt

from src.classes import Data

from src.utils import save_tex_fig
from src.utils import ask_user_file
from src.utils import model_quant_V_trace
from src.utils import get_specs, compute_C, compute_A
from src.utils import smooth, smooth_derivative

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

##########################################
## Plot model trace and measurement
##########################################

min_idx      = np.argwhere(data.time > 1/frequency)[0][0]
max_idx      = np.argwhere(data.time > 1/frequency*3)[0][0]
max_nb_points= 500
skip         = np.max([1, len(data.time[:max_idx])//max_nb_points])

plt.figure()

plt.plot(data.time[min_idx:max_idx:skip] * 1e3, data.voltage[min_idx:max_idx:skip], label="Measurement", color="black")

plt.plot(data.time[min_idx:max_idx:skip] * 1e3, model_trace[min_idx:max_idx:skip], "--", label="Model", color='grey')
plt.fill_between(data.time[min_idx:max_idx:skip] * 1e3, np.min(model_bounds, axis=0)[min_idx:max_idx:skip],
                    np.max(model_bounds, axis=0)[min_idx:max_idx:skip], 
                    color='black', alpha=0.1) #, label='Confidence interval Model'
plt.xlabel(rf"Time [ms]")
plt.ylabel(r"$V$ [V]")
plt.legend(frameon=False, loc='lower center', ncols=2, bbox_to_anchor=(0.5, 1.06))
plt.tight_layout()
plt.savefig(f"results/trace.png")
save_tex_fig(f"results/trace")
print("\033[93mSaved to results/trace\033[0m")


##########################################
## Plot the pressure and its derivative
##########################################


min_idx      = np.argwhere(data.time > 1/frequency)[0][0]
max_idx      = np.argwhere(data.time > 1/frequency*6)[0][0]
max_nb_points= 500
skip         = np.max([1, len(data.time[:max_idx])//max_nb_points])

plt.figure()
plt.plot(data.time[min_idx:max_idx:skip] * 1e3,
         smooth(data.time[min_idx:max_idx:skip], data.pressure[min_idx:max_idx] / 1e3, window_length=1001)[::skip], color="black")
plt.xlabel(rf"Time [ms]")
plt.ylabel(r"$P$ [kPa]")
plt.tight_layout()
plt.savefig(f"results/pressure.png")
save_tex_fig(f"results/pressure")
print("\033[93mSaved to results/pressure\033[0m")

plt.figure()
plt.plot(data.time[min_idx:max_idx:skip] * 1e3,
         smooth_derivative(data.time[min_idx:max_idx:skip], np.power(data.pressure[min_idx:max_idx] / 1e3, 2/3), window_length=1001)[::skip], color="black")
plt.xlabel(rf"Time [ms]")
plt.ylabel(r"$\partial_t P^{2/3}$ [kPa$^{2/3}$/s]")
plt.tight_layout()
plt.savefig(f"results/derivative.png")
save_tex_fig(f"results/derivative")
print("\033[93mSaved to results/derivative\033[0m")