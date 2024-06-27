import numpy as np
import matplotlib.pyplot as plt

from src.classes import Data

from src.utils import save_tex_fig
from src.utils import ask_user_file
from src.utils import get_fourier
from src.utils.models import model_quant_V_freq
from src.utils.coefficient import get_specs, compute_C, compute_A

##########################################
## Plot model and measurement
##########################################

thickness, frequency, R_decade_box = ask_user_file()

data            = Data(f'data/{int(thickness)}um/SingleMeasurement_{int(frequency)}Hz_{int(R_decade_box)}Ohms.csv')
specs           = get_specs(f'{int(thickness)}um')

A               = compute_A(specs)
C               = compute_C(specs)

voltate_fft     = get_fourier(data.voltage)
model_fft       = model_quant_V_freq(data.time, data.pressure,
                                A = A['nominal'], C = C['nominal'], R_decade_box = R_decade_box,
                                filter_window=21)

model_bounds     = np.array([
    model_quant_V_freq(data.time, data.pressure,
                                A = A['lower'], C = C['lower'], R_decade_box = R_decade_box,
                                filter_window=21),
    model_quant_V_freq(data.time, data.pressure,
                                A = A['lower'], C = C['upper'], R_decade_box = R_decade_box,
                                filter_window=21),
    model_quant_V_freq(data.time, data.pressure,
                                A = A['upper'], C = C['lower'], R_decade_box = R_decade_box,
                                filter_window=21),
    model_quant_V_freq(data.time, data.pressure,
                                A = A['upper'], C = C['upper'], R_decade_box = R_decade_box,
                                filter_window=21)
])

max_idx         = np.argwhere(data.frequency > frequency*4.5)[0][0]

plt.figure()

# plt.plot(data.frequency[:max_idx], voltate_fft[:max_idx], label="Measurement", color=plt.cm.plasma(0.3))
plt.scatter(data.frequency[:max_idx], voltate_fft[:max_idx]/specs['gam'].nominal_value, label="Measurement", color="black", zorder=2)
plt.plot(data.frequency[:max_idx], model_fft[:max_idx], '--', label="Model", color="grey", zorder=1)

plt.fill_between(data.frequency[:max_idx], np.min(model_bounds, axis=0)[:max_idx],
                    np.max(model_bounds, axis=0)[:max_idx],
                    color='black', alpha=0.1)
plt.xlabel(rf"$\omega / 2\pi$ [Hz]")
plt.ylabel(r"$\hat{V}$ [V]")
plt.legend(frameon=False, loc='upper left', ncols=1, bbox_to_anchor=(.6, 1))
plt.tight_layout()
plt.savefig(f"results/frequency_domain.png")
save_tex_fig(f"results/frequency_domain")
print("\033[93mSaved to results/frequency_domain\033[0m")



##########################################
## Plot pressure input
##########################################

pressure_fft    = get_fourier(np.power(data.pressure / 1e3, 2/3))

plt.figure()

plt.plot(data.frequency[:max_idx], 2*np.pi*data.frequency[:max_idx]*pressure_fft[:max_idx], label="Measurement", color="black")

plt.xlabel(rf"$\omega / 2\pi$ [Hz]")
plt.ylabel(r"$\omega \mathscr{F}_t(\delta p^{2/3}) $kPa$^{2/3}/$s")
plt.legend(frameon=False, loc='upper left', ncols=1, bbox_to_anchor=(.6, 1))
plt.tight_layout()

fundamental_freq = np.argmax((data.frequency*pressure_fft)[:max_idx])

current_ticks = plt.xticks()[0]
current_labels = plt.xticks()[1]

new_tick = data.frequency[fundamental_freq]
new_label = r'$p_\omega$'

plt.xticks(np.append(current_ticks, new_tick), np.append(current_labels, new_label))

plt.savefig(f"results/pressure_fft.png")
save_tex_fig(f"results/pressure_fft")
print("\033[93mSaved to results/pressure_fft\033[0m")