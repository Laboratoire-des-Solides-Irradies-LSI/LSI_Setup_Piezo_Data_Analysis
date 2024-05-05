import numpy as np
import os
import matplotlib.pyplot as plt

from src.classes import DataSet

from src.utils import save_tex_fig
from src.utils import ask_user_folder
from src.utils import fit_V, model_quali_V
from src.utils import get_specs, compute_A, compute_C

##########################################
## Plot the scaling lows
##########################################

all_thick     = np.array(np.sort([int(file[:-2]) for file in os.listdir('data') if not file.startswith('.')]), dtype=str)
As_fit        = []
As_theory     = []
As_theory_err = []

plt.figure()
colors        = plt.cm.viridis(np.linspace(0, 1, len(all_thick)))

for (i, thickness) in enumerate(all_thick):
    dataset       = DataSet(f'data/{int(thickness)}um')

    specs         = get_specs(f'{int(thickness)}um')
    C             = compute_C(specs)
    A             = compute_A(specs)

    A_fit, _      = fit_V(np.concatenate(dataset.frequencies)*2*np.pi * np.concatenate(dataset.resistances),
                    np.concatenate(dataset.voltage) / np.concatenate(dataset.pressure),
                    C=C['nominal'])
    
    As_fit.append(A_fit)
    As_theory.append(A['nominal'])
    As_theory_err.append((A['upper'] - A['lower'])/2)

    plt.scatter(np.concatenate(dataset.resistances)*2*np.pi*np.concatenate(dataset.frequencies),
                np.concatenate(dataset.voltage) / np.concatenate(dataset.pressure),
                s=2, color=colors[i], zorder=2, label=f'{thickness} $\mu$m')


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
plt.savefig(f"results/voltage_all_scaling.png")
save_tex_fig(f"results/voltage_all_scaling")

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
plt.legend()
plt.tight_layout()
plt.savefig(f"results/thickness.png")
save_tex_fig(f"results/thickness")