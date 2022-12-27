from merger_clean import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import time
from numba import jit
params = {'legend.fontsize': 7,
          'legend.handlelength': 2}
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)



# sim = 'm2s8'
# sim2 = 'm2s8b'

path = '/home/painchess/sims/'

s = -1
sim1 = Simulation(sim_names[s], omegas[s], sigmas[s], old_path)
# s = 4
# sim1 = Simulation(sim_names[s], omegas[s], sigmas[s], path)
reds = sim1.get_redshifts()

#sim1 = Simulation(sim, 0.2, 0.8, path)

snapshots = [1, 2, 3]
mlim, bins, ximin, ximax = 1e13, 20, 1e-2, 1
xis = np.logspace(np.log10(ximin), np.log10(ximax), bins + 1)
dxis = xis[1:] - xis[:-1]
#m_reds = np.loadtxt('/home/painchess/asus_fedora/simulation_reading/simu_redshifts.txt')[::-1]

colors = ['blue', 'red', 'green', 'orange']
for i in range(len(snapshots)):
    snap = snapshots[i]
    nmgs, tds = sim1.dndxi(snap, mlim, bins, ximin, ximax, wpos=False)
    dz = reds[snap + 1] - reds[snap]
    y = nmgs / dz / dxis / tds[1:]
    poisson = np.sqrt(nmgs) / dz / tds[1:] / dxis

    plt.plot(xis[1:-1], ell_mrate_per_n(5 * mlim, reds[snap], xis, om0=sim1.om0, sig8=sim1.sig8), '-', color='C{}'.format(i), linewidth=1.5)
    plt.scatter(xis[1:], y, color='C{}'.format(i), s=30, label='snap = {}, z={:1.1f}'.format(44-snap, reds[snap]))

    # plt.plot(xis[1:-1], ell_mrate_per_n(5 * mlim, reds[snap], xis, om0=sim1.om0, sig8=sim1.sig8), '--', color=col, linewidth=1)
    # plt.plot(xis[1:], y, color=col, marker='o', ms= 3, ls=' ', label='snap = {}, z={:1.1f}'.format(118-snap, reds[snap]), )

    plt.fill_between(xis[1:], y - poisson, y + poisson, color='C{}'.format(i), alpha=0.4)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\xi =M_1/M_2$', size=15)
plt.ylabel(r'dN/dz/d$\xi$ [mergers/halo/dz]')
plt.ylim(1e-1, 1e3)
plt.xlim(ximin, ximax)
plt.title(r'$M_0$ = {:2.1e} {}'.format(mlim, sim_names[s]))
plt.legend()
plt.show()
