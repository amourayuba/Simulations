from simulation import *
import matplotlib.pyplot as plt
import matplotlib as mpl

params = {'legend.fontsize': 7,
          'legend.handlelength': 2}
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

sim_names = ['M25S07', 'M25S08', 'M25S09', 'M03S07','M03S08', 'M03S09', 'M35S07', 'M35S08', 'M35S09',
                 'Illustris', 'bolshoiP', 'bolshoiW', 'M03S08b', 'm25s85', 'm2s8', 'm4s7', 'm4s8', 'm2s9',
                 'm3s8_50', 'm3s8', 'm35s75', 'm4s9', 'm3s9', 'm25s75', 'm2s1', 'm3s7', 'm3s85', 'm2s7', 'm25s8',
                 'm35s8', 'm25s9', 'm35s85', 'm3s75', 'm35s9', 'm35s7']
omegas = [0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.35, 0.35, 0.35, 0.309, 0.307, 0.27, 0.3, 0.25, 0.2, 0.4, 0.4, 0.2,  0.3
              ,0.3, 0.35, 0.4, 0.3, 0.25, 0.2, 0.3, 0.3, 0.2, 0.25, 0.35, 0.25, 0.35, 0.3, 0.35, 0.35]
sigmas = [0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.816, 0.82, 0.82, 0.8, 0.85, 0.8, 0.7, 0.8, 0.9, 0.8
              ,0.8, 0.75, 0.9, 0.9, 0.75, 1.0, 0.7, 0.85, 0.7, 0.8, 0.8, 0.9, 0.85, 0.75, 0.9, 0.7]

sims = dict(zip(sim_names, list(zip(omegas, sigmas))))
sim = 'M35S09'
# sim2 = 'm2s8b'

path = '/home/painchess/sims/'

sim1 = Simulation(sim, sims[sim][0], sims[sim][1], path)
# s = 4
# sim1 = Simulation(sim_names[s], omegas[s], sigmas[s], path)
reds = sim1.get_redshifts()

# sim1 = Simulation(sim, 0.2, 0.8, path)

snapshots = [2, 3, 6, 8, 10]
mlim, bins, ximin, ximax = 5e13, 15, 1e-2, 1
xis = np.logspace(np.log10(ximin), np.log10(ximax), bins + 1)
dxis = xis[1:] - xis[:-1]
# m_reds = np.loadtxt('/home/painchess/asus_fedora/simulation_reading/simu_redshifts.txt')[::-1]

colors = ['blue', 'red', 'green', 'orange']
for i in range(len(snapshots)):
    snap = snapshots[i]
    nmgs, tds = sim1.dndxi(snap, mlim, bins, ximin, ximax, wpos=False)
    dz = reds[snap + 1] - reds[snap]
    y = nmgs / dz / dxis / tds[1:]
    poisson = np.sqrt(nmgs) / dz / tds[1:] / dxis

    plt.plot(xis[1:-1], ell_mrate_per_n(5 * mlim, reds[snap], xis, om0=sim1.om0, sig8=sim1.sig8), '-',
             color='C{}'.format(i), linewidth=1.5)
    plt.scatter(xis[1:], y, color='C{}'.format(i), s=30, label='snap = {}, z={:1.1f}'.format(118 - snap, reds[snap]))

    # plt.plot(xis[1:-1], ell_mrate_per_n(5 * mlim, reds[snap], xis, om0=sim1.om0, sig8=sim1.sig8), '--', color=col, linewidth=1)
    # plt.plot(xis[1:], y, color=col, marker='o', ms= 3, ls=' ', label='snap = {}, z={:1.1f}'.format(118-snap, reds[snap]), )

    plt.fill_between(xis[1:], y - poisson, y + poisson, color='C{}'.format(i), alpha=0.4)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\xi =M_1/M_2$', size=15)
plt.ylabel(r'dN/dz/d$\xi$ [mergers/halo/dz]')
plt.ylim(1e-1, 1e5)
plt.xlim(ximin, ximax)
plt.title(r'$M_0$ = {:2.1e} {}'.format(mlim, sim))
plt.legend()
plt.show()
