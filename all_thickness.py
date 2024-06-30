import numpy as np
import os
import matplotlib.pyplot as plt

from src.classes import DataSet

from src.utils import save_tex_fig
from src.utils import fit_V, model_quali_V
from src.utils import get_specs, compute_A, compute_C


##########################################
## Plot the scaling law for the voltage
##########################################

all_thick       = np.array(np.sort([int(file[:-2]) for file in os.listdir('data') if not file.startswith('.')]), dtype=float)
As_fit          = []
As_theory       = []
As_theory_err   = []

plt.figure()
colors          = plt.cm.plasma(np.linspace(0, 1, len(all_thick)+1))

for (i, thickness) in enumerate(all_thick):
    dataset     = DataSet(f'data/{int(thickness)}um')

    specs       = get_specs(f'{int(thickness)}um')
    C           = compute_C(specs)
    A           = compute_A(specs)

    A_fit, _    = fit_V(np.concatenate(dataset.frequencies)*2*np.pi * np.concatenate(dataset.resistances),
                    np.concatenate(dataset.voltage) / np.concatenate(dataset.pressure_p2third),
                    C=C['nominal'])
    
    As_fit.append(A_fit)

    As_theory.append(A['nominal'])
    As_theory_err.append((A['upper'] - A['lower'])/2)

    plt.plot(np.concatenate(dataset.resistances)*2*np.pi*np.concatenate(dataset.frequencies),
                np.concatenate(dataset.voltage) / np.power(np.concatenate(dataset.pressure_p2third), 2/3),
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
plt.legend(frameon=False, loc='upper left', ncols=1, bbox_to_anchor=(.05, .95))
plt.tight_layout()
plt.savefig(f"results/voltage_all_scaling.png")
save_tex_fig(f"results/voltage_all_scaling")
print("\033[93mSaved to results/voltage_all_scaling\033[0m")


##########################################
## Plot the A measurement vs the thickness
##########################################

plt.figure()
l = np.linspace(np.min(all_thick), np.max(all_thick), 100)

plt.scatter(all_thick**(-2/3), As_fit, color='black', label="Measurement", marker='o')
plt.errorbar(all_thick**(-2/3), As_theory, color='black', yerr=As_theory_err, capsize=5, ecolor='gray', fmt='none')
plt.scatter(all_thick**(-2/3), As_theory, color='black', label="Model", marker='>')

plt.plot(l**(-2/3), np.mean(As_theory / all_thick**(-2/3)) *l**(-2/3), "--", color="black", alpha=0.4)

color = plt.cm.plasma(0.4)

plt.xlabel(rf"$\ell^{{-2/3}}$ [um]")
plt.ylabel(rf"$A$")
# plt.ylim(0, np.max(As_theory)*1.4)
plt.legend(frameon=False, loc='upper left', ncols=1, bbox_to_anchor=(.05, 1))
plt.tight_layout()
plt.savefig(f"results/thickness.png")
save_tex_fig(f"results/thickness")
print("\033[93mSaved to results/thickness\033[0m")