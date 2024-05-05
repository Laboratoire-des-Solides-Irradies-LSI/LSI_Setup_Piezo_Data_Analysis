import numpy as np
import matplotlib.pyplot as plt

from src.classes import DataSet

from src.utils import save_tex_fig
from src.utils import ask_user_folder
from src.utils import fit_V, model_quali_V
from src.utils import get_specs, compute_C

##########################################
## Load the data and fit the model
##########################################

thickness     = ask_user_folder()
dataset       = DataSet(f'data/{int(thickness)}um')

specs         = get_specs(f'{int(thickness)}um')
C             = compute_C(specs)

A_fit, _      = fit_V(np.concatenate(dataset.frequencies)*2*np.pi * np.concatenate(dataset.resistances),
                 np.concatenate(dataset.voltage) / np.concatenate(dataset.pressure),
                 C=C['nominal'])

colors        = plt.cm.viridis(np.linspace(0, 1, len(dataset.frequencies)))

##########################################
## Plot the voltage
##########################################

plt.figure()

for R, freq, pressure, voltage, color in zip(dataset.resistances, np.unique(dataset.frequencies), dataset.pressure, dataset.voltage, colors):
    plt.scatter(R, voltage / pressure, s=2, color=color, zorder=2, label=f'{freq:.0f} Hz')


x = np.logspace(np.log10(np.min(dataset.resistances[0])), np.log10(np.max(dataset.resistances[0])), 100)

for f in np.unique(dataset.frequencies):
    plt.plot(x,  model_quali_V(x*2*np.pi*f, A_fit, C['nominal']),  color='gray', alpha=.2, zorder=1)

plt.xscale('log')
plt.xlabel(rf"$R\ [\Omega]$")
plt.ylabel(r'$\frac{V}{ P^{2/3}}\ [$V$\cdot$Pa$^{-2/3}]$')
plt.legend()
plt.tight_layout()
plt.savefig(f"results/voltage.png")
save_tex_fig(f"results/voltage")

##########################################
## Plot the voltage scaling law
##########################################

plt.figure()

for R, freq, pressure, voltage, color in zip(dataset.resistances, np.unique(dataset.frequencies), dataset.pressure, dataset.voltage, colors):
    plt.scatter(R*2*np.pi*freq, voltage / pressure, s=2, color=color, zorder=2, label=f'{freq:.0f} Hz')


x = np.logspace(np.log10(np.min(dataset.resistances[0])*2*np.pi*np.min(dataset.frequencies)),
                np.log10(np.max(dataset.resistances[0])*2*np.pi*np.max(dataset.frequencies)),
                100)

for f in np.unique(dataset.frequencies):
    plt.plot(x,  model_quali_V(x, A_fit, C['nominal']),  color='gray', alpha=.2, zorder=1)

plt.xscale('log')
plt.xlabel(rf"$R\ [\Omega]$")
plt.ylabel(r'$\frac{V}{ P^{2/3}}\ [$V$\cdot$Pa$^{-2/3}]$')
plt.legend()
plt.tight_layout()
plt.savefig(f"results/voltage_scaling.png")
save_tex_fig(f"results/voltage_scaling")

##########################################
## Plot the power
##########################################

plt.figure()

for R, freq, pressure, voltage, color in zip(dataset.resistances, np.unique(dataset.frequencies), dataset.pressure, dataset.voltage, colors):
    plt.scatter(R, (voltage/pressure)**2 / R, s=2, color=color, zorder=2, label=f'{freq:.0f} Hz')


x = np.logspace(np.log10(np.min(dataset.resistances[0])), np.log10(np.max(dataset.resistances[0])) + 2, 100)

for f in np.unique(dataset.frequencies):
    plt.plot(x,  model_quali_V(x*2*np.pi*f, A_fit, C['nominal'])**2 / x,  color='gray', alpha=.2, zorder=1)

plt.xscale('log')
plt.xlabel(rf"$R\ [\Omega]$")
plt.ylabel(r'$\frac{P_e}{ P^{4/3}}\ [$W$\cdot$Pa$^{-4/3}]$')
plt.legend()
plt.tight_layout()
plt.savefig(f"results/power.png")
save_tex_fig(f"results/power")