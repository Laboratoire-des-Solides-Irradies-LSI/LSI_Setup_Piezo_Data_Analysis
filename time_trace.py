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

max_idx      = np.argwhere(data.time > 1/frequency*2)[0][0]

plt.figure()
plt.plot(data.time[:max_idx], data.voltage[:max_idx], label="Measurement", color='black')
plt.plot(data.time[:max_idx], model_trace[:max_idx], label="Model", color='black', alpha=0.4)
plt.fill_between(data.time[:max_idx], np.min(model_bounds, axis=0)[:max_idx],
                    np.max(model_bounds, axis=0)[:max_idx], 
                    color='black', alpha=0.1, label='Confidence interval Model')
plt.xlabel(rf"Time [s]")
plt.ylabel(r"$\hat{V}$ [V]")
# plt.legend()
plt.tight_layout()
plt.savefig(f"results/trace.png")
save_tex_fig(f"results/trace")