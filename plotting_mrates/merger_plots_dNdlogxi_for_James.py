import sys
sys.path.append('/home/painchess/projects_clean/Halo_Analytical_Calculations')
import merger_rate as mr
import numpy as np
from simulation import Simulation
import matplotlib.pyplot as plt
import matplotlib as mpl

params = {'legend.fontsize': 7,
          'legend.handlelength': 2}
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

sim_names = ['M25S07', 'M25S08', 'M25S09', 'M03S07', 'M03S08', 'M03S09', 'M35S07', 'M35S08', 'M35S09',
                 'Illustris', 'bolshoiP', 'bolshoiW', 'M03S08b', 'm25s85', 'm2s8', 'm4s7', 'm4s8', 'm2s9',
                 'm3s8_50', 'm3s8', 'm35s75', 'm4s9', 'm3s9', 'm25s75', 'm2s1', 'm3s7', 'm3s85', 'm2s7', 'm25s8',
                 'm35s8', 'm25s9', 'm35s85', 'm3s75', 'm35s9', 'm35s7']
omegas = [0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.35, 0.35, 0.35, 0.309, 0.307, 0.27, 0.3, 0.25, 0.2, 0.4, 0.4, 0.2, 0.3
        , 0.3, 0.35, 0.4, 0.3, 0.25, 0.2, 0.3, 0.3, 0.2, 0.25, 0.35, 0.25, 0.35, 0.3, 0.35, 0.35]
sigmas = [0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.816, 0.82, 0.82, 0.8, 0.85, 0.8, 0.7, 0.8, 0.9, 0.8
        , 0.8, 0.75, 0.9, 0.9, 0.75, 1.0, 0.7, 0.85, 0.7, 0.8, 0.8, 0.9, 0.85, 0.75, 0.9, 0.7]

sims = dict(zip(sim_names, list(zip(omegas, sigmas))))



# sim_selec = np.array([-2, -3])
# styles = ['-', '--', '-.', '-.-']
# markers = ['.', '^', 'o', 'c']
# ximin, ximax, resol = 1e-2, 0.8, 40
# xis = np.logspace(np.log10(ximin), np.log10(ximax), resol+1)
# dxis = xis[1:]-xis[:-1]
# mlims = [1e12, 3e12, 1e13, 3e13]
# snaps = [3, 5, 7, 9, 11]
# for k in range(len(mlims)):
#     mlim = mlims[k]
#     plt.figure()
#     for i in range(len(sim_selec)):
#         sim = sim_selec[i]
#         sim1 = Simulation(sim_names[sim], omegas[sim], sigmas[sim])
#         om = omegas[sim]
#         s8 = sigmas[sim]
#         reds = sim1.get_redshifts(mrate_path)
#         for j in range(len(snaps)):
#             snap = snaps[j]
#             dz = reds[snap +1]-reds[snap]
#             nmgs, tnds = sim1.dndxi(mrate_path, snap, mlim, bins=resol, wpos=False)
#             y = nmgs/dz/dxis/tnds[1:]
#             plt.plot(xis[1:], xis[1:]*y, color='C{}'.format(i), linestyle = (0, (5, j)), marker=j,
#                         label=r'{}'.format(sim_names[sim])[:6*(j==0)]+'  z = {:2.1f}'.format(reds[snap])[:10*(i==0)])
#             #plt.plot(xis[1:-1], ell_mrate_per_n(5*mlim, reds[snap], xis, om0=om, sig8=s8), color='C{}'.format(i),
#             # linestyle = (0, (5, j)), linewidth=2, alpha=0.2)
#     plt.title('M > {:1.2e}'.format(mlim), size=20)
#     plt.xlabel(r'$\xi =M_1/M_2$', size=15)
#     plt.ylabel(r'dN/dz/dlog$\xi$ [mergers/halo/dz]')
#     plt.xscale('log')
#     plt.yscale('log')
#     ax = plt.gca()
#     ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
#     #ax.yaxis.set_minor_locator(ticker.MaxNLocator(nbins=7))
#     ax.yaxis.set_minor_locator(ticker.LogLocator(base=2, numticks=8))
#     ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
#     ax.xaxis.set_minor_locator(ticker.LogLocator(base=2, numticks=8))
#     ax.yaxis.set_major_locator(ticker.NullLocator())
#     ax.yaxis.set_major_formatter(ticker.NullFormatter())
#     ax.xaxis.set_major_locator(ticker.NullLocator())
#     ax.xaxis.set_major_formatter(ticker.NullFormatter())
#     plt.legend()
#     plt.savefig('dN_dlogxi_{:1.1e}.png'.format(mlim))
#     plt.show()
#
#     # ax.yaxis.set_major_locator(ticker.NullLocator())
#     # ax.yaxis.set_major_formatter(ticker.NullFormatter())
#     #
#     # colors = ['blue', 'red', 'green', 'orange']



sim_selec = np.arange(9)
#styles = ['-', '--', '-.', '-.-']
#markers = ['.', '^', 'o', 'c']
ximin, ximax, resol = 1e-2, 1, 10
xis = np.logspace(np.log10(ximin), np.log10(ximax), resol+1)
dxis, ximeans = xis[1:]-xis[:-1], np.sqrt(xis[1:]*xis[:-1])
#mlims = [1e12, 3e12, 1e13, 3e13]
mlims = [1e13]
sel_reds = np.array([0.05, 0.55, 0.8])
fig, axs = plt.subplots(3,3, sharex=True, sharey=True, figsize=[11.5,11.5])


mrate_path = '/home/painchess/asus_fedora/merger_rate'
old_path = '/home/painchess/disq2/ahf-v1.0-101/'
localpath = '/home/painchess/sims/'
externalpath = '/home/painchess/mounted/HR_sims/niagara/'
illustris_path = '/home/painchess/mounted/TNG300-625/output'

for k in range(len(mlims)):
    mlim = mlims[k]

    #plt.figure()
    #for i in np.delete(np.arange(9, len(sim_selec)), [1, 2]):
    for i in range(len(sim_selec)):
        ax = axs[i // 3, i % 3]
        #lt.figure() #make a figure for each simulation and for each mass
        sim = sim_selec[i]
        sim1 = Simulation(sim_names[sim], omegas[sim], sigmas[sim], localpath)
        om = omegas[sim]
        s8 = sigmas[sim]
        if sim_names[sim][0] == 'I':
            m_path = illustris_path
        else:
            m_path = mrate_path
        reds = sim1.get_redshifts()
        for j in range(len(sel_reds)):
            snap = np.min(np.where(reds >= sel_reds[j]))
            dz = reds[snap +1]-reds[snap]

            nmgs, tnds = sim1.dndxi(snap, mlim, bins=resol, wpos=False)
            y = nmgs/dz/dxis/tnds[:-1]
            ps = np.sqrt(1/nmgs + 1/tnds[:-1])*nmgs/tnds[:-1]
            tps = ps/dz/dxis
            #ps = np.sqrt(nmgs) / dz / tnds[1:] / dxis
            ax.plot(ximeans,  y, 'o', ms=2,  color='C{}'.format(j),
                     label=r'z = {:1.2f}'.format(reds[snap]))
            ax.fill_between(ximeans, y-tps, y+tps,  color='C{}'.format(j), alpha=0.4)
            ax.plot(xis[1:-1], mr.ell_mrate_per_n(5*mlim, reds[snap], xis, om0=om, sig8=s8), color='C{}'.format(j),
              linewidth=1.5)
        #plt.title('M > {:1.2e} {}'.format(mlim, sim_names[sim]), size=13)
        ax.set_title(sim_names[sim], size=17)
        if i==0:
            ax.legend(fontsize=15)
        if i//3 == 2:
            ax.set_xlabel(r'$\xi =M_1/M_2$', size=20)

        if i == 3:
            ax.set_ylabel(r'dN/dz/d$\xi$ [mergers/halo/dz]', size=25)
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax = plt.gca()
        # ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
        # #ax.yaxis.set_minor_locator(ticker.MaxNLocator(nbins=7))
        # ax.yaxis.set_minor_locator(ticker.LogLocator(base=2, numticks=8))
        # ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        # ax.xaxis.set_minor_locator(ticker.LogLocator(base=2, numticks=8))
        # ax.yaxis.set_major_locator(ticker.NullLocator())
        # ax.yaxis.set_major_formatter(ticker.NullFormatter())
        # ax.xaxis.set_major_locator(ticker.NullLocator())
        # ax.xaxis.set_major_formatter(ticker.NullFormatter())

# plt.savefig('dN_dxi_{}_{:1.1e}.png'.format(sim_names[sim], mlim), facecolor='white', transparent=False, bbox_inches='tight')
# plt.savefig('dN_dxi_{}_{:1.1e}.pdf'.format(sim_names[sim], mlim), dpi=300, facecolor='white', transparent=False, bbox_inches='tight')
#plt.savefig('dN_dxi_{:1.1e}.png'.format(mlim), facecolor='white', transparent=False, bbox_inches='tight')
#plt.savefig('dN_dxi_{:1.1e}.pdf'.format(mlim), dpi=300, facecolor='white', transparent=False, bbox_inches='tight')

plt.show()

