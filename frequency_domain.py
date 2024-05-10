import numpy as np
import matplotlib.pyplot as plt

from src.classes import Data

from src.utils import save_tex_fig
from src.utils import ask_user_file
from src.utils import get_fourier
from src.utils.models import model_quant_V_freq
from src.utils.coefficient import get_specs, compute_A, compute_C

thickness, frequency, R_decade_box = ask_user_file()

data         = Data(f'data/{int(thickness)}um/SingleMeasurement_{int(frequency)}Hz_{int(R_decade_box)}Ohms.csv')
specs        = get_specs(f'{int(thickness)}um')

voltate_fft  = get_fourier(data.voltage)
model_fft    = model_quant_V_freq(data.time, data.pressure,
                                A = compute_A(specs)['nominal'], C = compute_C(specs)['nominal'], R_decade_box = R_decade_box,
                                filter_window=21)

max_idx      = np.argwhere(data.frequency > frequency*4.5)[0][0]

plt.figure()
plt.plot(data.frequency[:max_idx], voltate_fft[:max_idx], label="Measurement", color='black')
plt.plot(data.frequency[:max_idx], model_fft[:max_idx], label="Model", color='black', alpha=0.2)
plt.xlabel(rf"$\omega / 2\pi$ [Hz]")
plt.ylabel(r"$\hat{V}$ [V]")
plt.legend(frameon=False, loc='upper left', ncols=1, bbox_to_anchor=(.6, 1))
plt.tight_layout()
plt.savefig(f"results/frequency_domain.png")
save_tex_fig(f"results/frequency_domain")
print("\033[93mSaved to results/frequency_domain\033[0m")