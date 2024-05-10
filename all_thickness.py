import numpy as np
import os
import matplotlib.pyplot as plt

from src.classes import DataSet

from src.utils import save_tex_fig
from src.utils import ask_user_folder
from src.utils import fit_V, model_quali_V
from src.utils import get_specs, compute_A, compute_C


##########################################
## Plot the scaling law for the voltage
##########################################

all_thick       = np.array(np.sort([int(file[:-2]) for file in os.listdir('data') if not file.startswith('.')]), dtype=str)
As_fit          = []
As_theory       = []
As_theory_err   = []

plt.figure()
colors          = plt.cm.viridis(np.linspace(0, 1, len(all_thick)+1))

for (i, thickness) in enumerate(all_thick):
    dataset     = DataSet(f'data/{int(thickness)}um')

    specs       = get_specs(f'{int(thickness)}um')
    C           = compute_C(specs)
    A           = compute_A(specs)

    A_fit, _    = fit_V(np.concatenate(dataset.frequencies)*2*np.pi * np.concatenate(dataset.resistances),
                    np.concatenate(dataset.voltage) / np.concatenate(dataset.pressure),
                    C=C['nominal'])
    
    As_fit.append(A_fit)
    As_theory.append(A['nominal'])
    As_theory_err.append((A['upper'] - A['lower'])/2)

    plt.plot(np.concatenate(dataset.resistances)*2*np.pi*np.concatenate(dataset.frequencies),
                np.concatenate(dataset.voltage) / np.concatenate(dataset.pressure),
                marker="o", linestyle='None', markersize=np.sqrt(4), 
                color=colors[i], zorder=2, label=rf'\makebox[6mm][r]{{{thickness}}} $\mu$m')


    x           = np.logspace(np.log10(np.min(dataset.resistances[0])*2*np.pi*np.min(dataset.frequencies)),
                    np.log10(np.max(dataset.resistances[0])*2*np.pi*np.max(dataset.frequencies)),
                    100)

    for f in np.unique(dataset.frequencies):
        plt.plot(x,  model_quali_V(x, A_fit, C['nominal']),  color='gray', alpha=.2, zorder=1)

plt.xscale('log')
plt.xlabel(rf"$R\omega\ [\Omega\cdot$Rad$\cdot$s$^{{-1}}]$")
plt.ylabel(r'$\frac{V}{ P^{2/3}}\ [$V$\cdot$Pa$^{-2/3}]$')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"results/voltage_all_scaling.png")
save_tex_fig(f"results/voltage_all_scaling")
print("\033[93mSaved to results/voltage_all_scaling\033[0m")


##########################################
## Plot the scaling law for the voltage
##########################################

plt.figure()

for (i, thickness) in enumerate(all_thick):
    dataset     = DataSet(f'data/{int(thickness)}um')

    specs       = get_specs(f'{int(thickness)}um')
    C           = compute_C(specs)
    A           = compute_A(specs)
    

    plt.plot(np.concatenate(dataset.resistances)*2*np.pi*np.concatenate(dataset.frequencies),
                (np.concatenate(dataset.voltage) / np.concatenate(dataset.pressure))**2 / np.concatenate(dataset.resistances),
                marker="o", linestyle='None', markersize=np.sqrt(4), 
                color=colors[i], zorder=2, label=rf'\makebox[6mm][r]{{{thickness}}} $\mu$m')


    x           = np.logspace(np.log10(np.min(dataset.resistances[0])*2*np.pi*np.min(dataset.frequencies)),
                    np.log10(np.max(dataset.resistances[0])*2*np.pi*np.max(dataset.frequencies)) + 2,
                    100)

    for f in np.unique(dataset.frequencies):
        plt.plot(x,  model_quali_V(x, A_fit, C['nominal'])**2 / x,  color='gray', alpha=.2, zorder=1)

plt.xscale('log')
plt.xlabel(rf"$R\omega\ [\Omega\cdot$Rad$\cdot$s$^{{-1}}]$")
plt.ylabel(r'$\frac{P_e}{ P^{4/3}}\ [$W$\cdot$Pa$^{-4/3}]$')
plt.legend(frameon=False, loc='lower center', ncols=3, bbox_to_anchor=(0.5, 1.16))
plt.tight_layout()
plt.savefig(f"results/power_all_scaling.png")
save_tex_fig(f"results/power_all_scaling")
print("\033[93mSaved to results/power_all_scaling\033[0m")


##########################################
## Plot the A measurement vs the thickness
##########################################

plt.figure()
l = np.linspace(10, 110, 1000)
plt.scatter(all_thick, As_fit, color='black', label="Measurement", marker='o')
plt.scatter(all_thick, As_theory, color='black', label="Model", marker='>')
plt.errorbar(all_thick, As_theory, color='black', yerr=As_theory_err, capsize=5, ecolor='gray', fmt='none')
plt.ylim(0, np.max(np.concatenate([As_fit, As_theory]))*1.5)
plt.xlabel(rf"$\ell$ [um]")
plt.ylabel(rf"$A$")
plt.legend(frameon=False, loc='lower center', ncols=3, bbox_to_anchor=(0.5, 1.16))
plt.tight_layout()
plt.savefig(f"results/thickness.png")
save_tex_fig(f"results/thickness")
print("\033[93mSaved to results/thickness\033[0m")