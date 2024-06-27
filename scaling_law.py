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
                 np.concatenate(dataset.voltage) / np.power(np.concatenate(dataset.pressure), 2/3),
                 C=C['nominal'])

colors        = plt.cm.plasma(np.linspace(0, 1, len(dataset.frequencies)+1))

##########################################
## Plot the voltage
##########################################

plt.figure()

for R, freq, pressure, voltage, color in zip(dataset.resistances, np.unique(dataset.frequencies), dataset.pressure, dataset.voltage, colors):
    plt.plot(R, voltage / np.power(pressure, 2/3),
                marker="o", linestyle='None', markersize=np.sqrt(4), 
                color=color, zorder=2, label=rf'\makebox[2mm][r]{{{freq:.0f}}} Hz')


x = np.logspace(np.log10(np.min(dataset.resistances[0])), np.log10(np.max(dataset.resistances[0])), 100)

for f in np.unique(dataset.frequencies):
    plt.plot(x,  model_quali_V(x*2*np.pi*f, A_fit, C['nominal']),  color='gray', alpha=.2, zorder=1)

plt.xscale('log')
plt.xlabel(rf"$R\ [\Omega]$")
plt.ylabel(r'$\frac{V}{ P^{2/3}}\ [$V$\cdot$Pa$^{-2/3}]$')
plt.legend(frameon=False, loc='upper left', ncols=2, bbox_to_anchor=(.05, .95))
plt.tight_layout()
plt.savefig(f"results/voltage.png")
save_tex_fig(f"results/voltage")
print("\033[93mSaved to results/voltage\033[0m")

##########################################
## Plot the voltage scaling law
##########################################

plt.figure()

for R, freq, pressure, voltage, color in zip(dataset.resistances, np.unique(dataset.frequencies), dataset.pressure, dataset.voltage, colors):
    plt.plot(R*2*np.pi*freq, voltage / np.power(pressure, 2/3),
                marker="o", linestyle='None', markersize=np.sqrt(4), 
                color=color, zorder=2, label=rf'\makebox[2mm][r]{{{freq:.0f}}} Hz')


x = np.logspace(np.log10(np.min(dataset.resistances[0])*2*np.pi*np.min(dataset.frequencies)),
                np.log10(np.max(dataset.resistances[0])*2*np.pi*np.max(dataset.frequencies)),
                100)


plt.fill_between(x, 
                 model_quali_V(x, A_fit, C['nominal']) * 0.95,
                 model_quali_V(x, A_fit, C['nominal']) * 1.05,
                   color='gray', alpha=.2, zorder=1)

plt.xscale('log')
plt.xlabel(rf"$R\omega\ [\Omega\cdot$Rad$\cdot$s$^{{-1}}]$")
plt.ylabel(r'$\frac{V}{ P^{2/3}}\ [$V$\cdot$Pa$^{-2/3}]$')
plt.legend(frameon=False, loc='upper left', ncols=2, bbox_to_anchor=(.05, .95))
plt.tight_layout()
plt.savefig(f"results/voltage_scaling.png")
save_tex_fig(f"results/voltage_scaling")
print("\033[93mSaved to results/voltage_scaling\033[0m")

##########################################
## Plot the power
##########################################

plt.figure()

for R, freq, pressure, voltage, color in zip(dataset.resistances, np.unique(dataset.frequencies), dataset.pressure, dataset.voltage, colors):
    plt.plot(R, (voltage/pressure)**2 / R,
            marker="o", linestyle='None', markersize=np.sqrt(4), 
            color=color, zorder=2, label=rf'\makebox[2mm][r]{{{freq:.0f}}} Hz')


x           = np.logspace(np.log10(np.min(dataset.resistances[0])), np.log10(np.max(dataset.resistances[0])) + 2, 100)

for f in np.unique(dataset.frequencies):
    plt.plot(x,  model_quali_V(x*2*np.pi*f, A_fit, C['nominal'])**2 / x,  color='gray', alpha=.2, zorder=1)

frequencies = np.linspace(np.min(dataset.frequencies)*0.9, np.max(dataset.frequencies)*1.1, 100)

plt.plot(1/(C['nominal']*2*np.pi*frequencies),
         model_quali_V(1/C['nominal'], A_fit, C['nominal'])**2 * C['nominal'] * 2*np.pi*frequencies,
         "--", color='black', alpha=0.4)

plt.xscale('log')
plt.xlabel(rf"$R\ [\Omega]$")
plt.ylabel(r'$\frac{P_e}{ P^{4/3}}\ [$W$\cdot$Pa$^{-4/3}]$')
plt.legend(frameon=False, loc='upper left', ncols=2, bbox_to_anchor=(.6, .95))
plt.tight_layout()
plt.savefig(f"results/power.png")
save_tex_fig(f"results/power")
print("\033[93mSaved to results/power\033[0m")