import sys
sys.path.append('/home/painchess/projects_clean/Halo_Analytical_Calculations')
import merger_rate as mr
from simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
params = {'legend.fontsize': 8,
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

mrate_path = '/home/painchess/asus_fedora/merger_rate'
old_path = '/home/painchess/disq2/ahf-v1.0-101/'
localpath = '/home/painchess/sims/'
externalpath = '/home/painchess/mounted/HR_sims/niagara/'
illustris_path = '/home/painchess/mounted/TNG300-625/output'


resol, ximin, ximax = 100, 1e-2, 1

log_fracs = [0, 0.25, 0.5]  # log space fraction between ximin and ximax
max_fracs = [0.25, 0.5, 0.75]
ximins = (resol * np.array(log_fracs)).astype(int)
ximaxs = (resol * np.array(max_fracs)).astype(int)
dexis = np.logspace(np.log10(ximin), np.log10(ximax), resol)
sim_selec1s = np.array([[0, 3, 6], [1,4,7], [2,5,8]])
#sim_selec1 = np.array([0, 3, 6])
sim_selec2s = np.array([[0, 1, 2], [3,4,5], [6,7,8]])

om1 = 0.3
om1s = [0.25,0.3,0.35]
sim_selec2 = np.array([6, 7, 8])
sim_selec3 = np.array([0, -1, -2, -4])
zmin, zmax = 0.01, 0.45
sim_res, pois_res = np.zeros((len(sim_selec1s[0]), resol - 1)), np.zeros((len(sim_selec1s[0]), resol - 1))
c_param = '\sigma_8'

fig, axs = plt.subplots(1, 3, figsize=[17, 6], sharey=True)

for p in range(3):
    #om1 = om1s[l]
    #sim_selec1 = sim_selec1s[p]
    for i in range(len(sim_selec1s[0])):
        if c_param == '\Omega_m':
            sim_selec1 = sim_selec1s[p]
            j = sim_selec1[i]
        elif om1 == 0.3:
            sim_selec2 = sim_selec2s[p]
            j = sim_selec2[i]
        else:
            j = sim_selec3[i]
        sim1 = Simulation(sim_names[j], omegas[j], sigmas[j], localpath)
        reds = sim1.get_redshifts()
        snap_min, snap_max = np.min(np.where(reds >= zmin)) + 1, np.min(np.where(reds >= zmax)) + 1
        snapshots = np.arange(snap_min, snap_max, 1)
        if sim1.name[0] == 'M':
            mlim = 5e13
        elif sim1.name[0] == 'I':
            mlim = 1e13
            mrate_path = illustris_path
        elif sim1.name[0] == 'm':
            mlim = 3e13
        else:
            mlim = 1e12
        mres, tnds = sim1.N_of_xi(snapshots, mlim, resol=resol, wpos=False)
        np.savetxt('counts_{}.txt'.format(sim_names[j]), mres[:-1,:]-mres[1:,:])
        np.savetxt('tot_halos_{}.txt'.format(sim_names[j]), tnds[1:, :])
        dmres = (mres[:-1, :] - mres[1:, :]) / tnds[1:, :]
        #ps = np.sqrt((mres[:-1, :] - mres[1:, :])*(1+(mres[:-1, :] - mres[1:, :])/tnds[1:, :]))/tnds[1:, :]
        #ps = np.sqrt(mres[:-1, :] - mres[1:, :])*(1-1/np.sqrt(tnds[1:, :]))/tnds[1:, :]
        ps = np.sqrt(mres[:-1, :] - mres[1:, :]) / tnds[1:, :]
        nmres = np.cumsum(dmres[::-1], axis=0)[::-1, :]
        nps = np.sqrt(np.cumsum(ps[::-1]**2, axis=0)[::-1, :])

        tot_integ = np.sum(nmres, axis=1)
        #tot_ps = np.sqrt(np.sum(nps**2, axis=1))
        tot_ps = np.sum(nps, axis=1)
        sim_res[i, :] = tot_integ
        pois_res[i, :] = tot_ps

    save = False
    zresol, omresol = 100, 50
    nreds = np.linspace(zmin, zmax, zresol)
    dz = (zmax - zmin) / zresol
    if c_param == '\Omega_m':
        a_omegas = np.linspace(0.15, 0.45, omresol)
    else:
        a_sigmas = np.linspace(0.6, 1, omresol)
    amlim = 1e13
    for n in range(len(ximins)):
        # ax = axs[n // 2, n % 2]

        i = ximins[n]
        #l = ximaxs[len(ximins) - 1]
        l = ximaxs[n]
        xmax = dexis[l-1]
        #xmax = max_fracs[len(ximins) - 1]

        if save:
            res = []
            for m in range(omresol):
                diff_nmerg = np.zeros(zresol)
                if c_param == '\Omega_m':
                    omg = a_omegas[m]
                    s8 = sim1.sig8
                else:
                    s8 = a_sigmas[m]
                    omg = sim1.om0
                for k in range(zresol):
                    diff_nmerg[k] = mr.integ_mrate(3 * amlim, nreds[k], xi_min=dexis[i], xi_max=xmax, om0=omg, sig8=s8)
                res.append(np.sum(diff_nmerg) * dz)
            if c_param == '\Omega_m':

                np.savetxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}'
                           '_ximax{:1.2f}_omegas_{}.txt'.format(zmin, zmax, amlim, dexis[i], xmax, s8), np.array(res))
            else:

                np.savetxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}'
                           '_ximax{:1.2f}_sigmas_{}.txt'.format(zmin, zmax, amlim, dexis[i], xmax, omg), np.array(res))

        else:
            if c_param == '\Omega_m':
                s8 = sim1.sig8
                res = np.loadtxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}_'
                                 'ximax{:1.2f}_omegas_{}.txt'.format(zmin, zmax, amlim, dexis[i], xmax, s8))
            else:
                omg = sim1.om0
                res = np.loadtxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_'
                                 'ximin{:1.2f}_ximax{:1.2f}_sigmas_{}.txt'.format(zmin, zmax, amlim, dexis[i], xmax, omg))
        if c_param == '\Omega_m':
            axs[p].plot(a_omegas, res, color='C{}'.format(n), label=r'{:1.2f} $<\xi<$ {:1.2f}'.format(dexis[i], xmax))
        else:
            axs[p].plot(a_sigmas, res, color='C{}'.format(n), label=r'{:1.2f} $<\xi<$ {:1.2f}'.format(dexis[i], xmax))
        if c_param == '\Omega_m':
            axs[p].errorbar(np.array(omegas)[sim_selec1[:3]], sim_res[:3, i] - sim_res[:3, l-2], pois_res[:3, i], fmt='o',
                        color='C{}'.format(n), label='MxSy'[:4 * (n == 0)])
            # ax.errorbar(np.array(omegas)[sim_selec1[4]], sim_res[-1, i] - sim_res[-1, l-2], pois_res[-1, i], fmt='s',
            #             label='Illustris'[:8 * (n == 0)])
            # ax.errorbar(np.array(omegas)[sim_selec1[3:-1]], sim_res[3:-1, i] - sim_res[3:-1, l-2], pois_res[3:-1, i], fmt='v',
            #             label='New HR sims'[:10 * (n == 0)])
        elif om1 == 0.3:
            axs[p].errorbar(np.array(sigmas)[sim_selec2[:3]], sim_res[:3, i] - sim_res[:3, l-2], pois_res[:3, i], fmt='o',
                        color='C{}'.format(n), label=r'MxSy'[:4 * (n == 0)])
            # ax.errorbar(np.array(sigmas)[sim_selec2[4]], sim_res[4, i] - sim_res[4, l-2], pois_res[4, i], fmt='s',
            #             label='Illustris'[:9 * (n == 0)])
            # ax.errorbar(np.array(sigmas)[sim_selec2[3]], sim_res[3, i] - sim_res[3, l-2], pois_res[3, i], fmt='v',
            #             label='New HR sim'[:10 * (n == 0)])
        else:
            axs[p].errorbar(np.array(sigmas)[sim_selec2[:3]], sim_res[:3, i] - sim_res[:3, l-2], pois_res[:3, i], fmt='o',
                        color='C{}'.format(n), label='MxSy'[:4 * (n == 0)])
            axs[p].errorbar(np.array(sigmas)[sim_selec2[3]], sim_res[3, i] - sim_res[3, l-2], pois_res[3, i], fmt='v',
                        color='C{}'.format(n), label='New HR sim'[:10 * (n == 0)])

    if p==0:
        axs[p].set_ylabel(r'dN($\xi_{min}$<$\xi$<$\xi_{max}$) [mergers/halo]', size=19)
        axs[p].legend(fontsize=10)
    axs[p].set_xlabel(r'${}$'.format(c_param), size=35)
    if p==1 and c_param=='\Omega_m':
        axs[p].set_title(r'{}<z<{}, log $M/M_\odot$>{:2.1f}'.format(zmin, zmax, np.log10(amlim)), size=25)
    if c_param == '\sigma_8':
        axs[p].annotate(r'$\Omega_m$ = {}'.format(omg), (0.6, 1.6), size=20)
    else:
        axs[p].annotate(r'$\sigma_8$ = {}'.format(s8), (0.14, 1.25), size=20)

if c_param == '\sigma_8':
    plt.savefig(r'n2integ_N_tot_{}_{:2.0f}_m{:2.1f}.pdf'.format(c_param[1:], 100 * omg, np.log10(amlim)),
            dpi=650, bbox_inches='tight', facecolor='white',
            transparent=False)
    plt.savefig(r'n2integ_N_tot_{}_{:2.0f}_m{:2.1f}.png'.format(c_param[1:], 100 * omg, np.log10(amlim)),
            dpi=650, bbox_inches='tight', facecolor='white',
            transparent=False)
else:
    plt.savefig(r'n2integ_N_tot_{}_{:2.0f}_s{:2.0f}.pdf'.format(c_param[1:], 100 * s8,  np.log10(amlim)),
                dpi=650, bbox_inches='tight', facecolor='white', transparent=False)
    plt.savefig(r'n2integ_N_tot_{}_{:2.0f}_s{:2.0f}.png'.format(c_param[1:], 100 * s8,  np.log10(amlim)),
                dpi=650, bbox_inches='tight', facecolor='white', transparent=False)
plt.show()
#




#
# fig = plt.figure(figsize=[7, 7])
#
# gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1])
# #fig, axs = plt.subplots(2, 2, figsize=[7, 7], dpi=500)
# save = False
# zresol, omresol = 100, 50
# nreds = np.linspace(zmin, zmax, zresol)
# dz = (zmax - zmin) / zresol
# if c_param == '\Omega_m':
#     a_omegas = np.linspace(0.15, 0.45, omresol)
# else:
#     a_sigmas = np.linspace(0.6, 1, omresol)
# amlim = 1e13
# for n in range(len(ximins)):
#     # ax = axs[n // 2, n % 2]
#     if n == 2:
#         ax = fig.add_subplot(gs[1:, :])
#         ax.set_box_aspect(1)
#     else:
#         ax = fig.add_subplot(gs[n // 2, n % 2])
#
#
#     i = ximins[n]
#     #l = ximaxs[len(ximins) - 1]
#     l = ximaxs[n]
#     xmax = dexis[l-1]
#     #xmax = max_fracs[len(ximins) - 1]
#
#     if save:
#         res = []
#         for m in range(omresol):
#             diff_nmerg = np.zeros(zresol)
#             if c_param == '\Omega_m':
#                 omg = a_omegas[m]
#                 s8 = 0.8
#             else:
#                 s8 = a_sigmas[m]
#                 omg = 0.3
#             for k in range(zresol):
#                 diff_nmerg[k] = integ_mrate(3 * amlim, nreds[k], xi_min=dexis[i], xi_max=xmax, om0=omg, sig8=s8)
#             res.append(np.sum(diff_nmerg) * dz)
#         if c_param == '\Omega_m':
#             np.savetxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}'
#                        '_ximax{:1.2f}_omegas.txt'.format(zmin, zmax, amlim, dexis[i], xmax), np.array(res))
#         else:
#             np.savetxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}'
#                        '_ximax{:1.2f}_sigmas.txt'.format(zmin, zmax, amlim, dexis[i], xmax), np.array(res))
#
#     else:
#         if c_param == '\Omega_m':
#             res = np.loadtxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}_'
#                              'ximax{:1.2f}_omegas.txt'.format(zmin, zmax, amlim, dexis[i], xmax))
#         else:
#             res = np.loadtxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_'
#                              'ximin{:1.2f}_ximax{:1.2f}_sigmas.txt'.format(zmin, zmax, amlim, dexis[i], xmax))
#     if c_param == '\Omega_m':
#         ax.plot(a_omegas, res, label=r'{:1.2f} $<\xi<$ {:1.2f}'.format(dexis[i], xmax))
#     else:
#         ax.plot(a_sigmas, res, label=r'{:1.2f} $<\xi<$ {:1.2f}'.format(dexis[i], xmax))
#     if c_param == '\Omega_m':
#         ax.errorbar(np.array(omegas)[sim_selec1[:3]], sim_res[:3, i] - sim_res[:3, l-2], pois_res[:3, i], fmt='o',
#                     label='MxSy'[:4 * (n == 0)])
#         # ax.errorbar(np.array(omegas)[sim_selec1[4]], sim_res[-1, i] - sim_res[-1, l-2], pois_res[-1, i], fmt='s',
#         #             label='Illustris'[:8 * (n == 0)])
#         # ax.errorbar(np.array(omegas)[sim_selec1[3:-1]], sim_res[3:-1, i] - sim_res[3:-1, l-2], pois_res[3:-1, i], fmt='v',
#         #             label='New HR sims'[:10 * (n == 0)])
#     elif om1 == 0.3:
#         ax.errorbar(np.array(sigmas)[sim_selec2[:3]], sim_res[:3, i] - sim_res[:3, l-2], pois_res[:3, i], fmt='o',
#                     label='MxSy'[:4 * (n == 0)])
#         # ax.errorbar(np.array(sigmas)[sim_selec2[4]], sim_res[4, i] - sim_res[4, l-2], pois_res[4, i], fmt='s',
#         #             label='Illustris'[:9 * (n == 0)])
#         # ax.errorbar(np.array(sigmas)[sim_selec2[3]], sim_res[3, i] - sim_res[3, l-2], pois_res[3, i], fmt='v',
#         #             label='New HR sim'[:10 * (n == 0)])
#     else:
#         ax.errorbar(np.array(sigmas)[sim_selec2[:3]], sim_res[:3, i] - sim_res[:3, l-2], pois_res[:3, i], fmt='o',
#                     label='MxSy'[:4 * (n == 0)])
#         ax.errorbar(np.array(sigmas)[sim_selec2[3]], sim_res[3, i] - sim_res[3, l-2], pois_res[3, i], fmt='v',
#                     label='New HR sim'[:10 * (n == 0)])
#
#     if n % 2 == 0:
#         ax.set_ylabel(r'dN(>$\xi$) [mergers/halo]')
#     if n // 2 == 1:
#         ax.set_xlabel(r'${}$'.format(c_param), size=15)
#     ax.legend()
# fig.suptitle(r'{}<z<{}, log $M/M_\odot$>{:2.1f}'.format(zmin, zmax, np.log10(amlim)), size=15)
# #plt.savefig(r'integ_N_tot_${}$_m{:2.1f}.pdf'.format(c_param, np.log10(amlim)), dpi=650, bbox_inches='tight', facecolor='white',
#             #transparent=False)
# plt.show()
# #
#


############################----------------------- SAME IN DIFFERENT PLOTS----------#################

# save = False
# zresol, omresol = 100, 50
# nreds = np.linspace(zmin, zmax, zresol)
# dz = (zmax - zmin) / zresol
# if c_param == '\Omega_m':
#     a_omegas = np.linspace(0.15, 0.45, omresol)
# else:
#     a_sigmas = np.linspace(0.6, 1, omresol)
# amlim = 1e13
# for n in range(len(ximins)):
#     #ax = axs[n // 2, n % 2]
#     i = ximins[n]
#     #l = ximaxs[len(ximins) - 1]
#     l = ximaxs[n]
#     xmax = dexis[l]
#     #xmax = max_fracs[len(ximins) - 1]
#     #plt.figure()
#     if save:
#         res = []
#         for m in range(omresol):
#             diff_nmerg = np.zeros(zresol)
#             if c_param == '\Omega_m':
#                 omg = a_omegas[m]
#                 s8 = 0.8
#             else:
#                 s8 = a_sigmas[m]
#                 omg = 0.3
#             for k in range(zresol):
#                 diff_nmerg[k] = integ_mrate(3 * amlim, nreds[k], xi_min=dexis[i], xi_max=xmax, om0=omg, sig8=s8)
#             res.append(np.sum(diff_nmerg) * dz)
#         if c_param == '\Omega_m':
#             np.savetxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}'
#                        '_ximax{:1.2f}_omegas.txt'.format(zmin, zmax, amlim, dexis[i], xmax), np.array(res))
#         else:
#             np.savetxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}'
#                        '_ximax{:1.2f}_sigmas.txt'.format(zmin, zmax, amlim, dexis[i], xmax), np.array(res))
#
#     else:
#         if c_param == '\Omega_m':
#             res = np.loadtxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}_'
#                              'ximax{:1.2f}_omegas.txt'.format(zmin, zmax, amlim, dexis[i], xmax))
#         else:
#             res = np.loadtxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_'
#                              'ximin{:1.2f}_ximax{:1.2f}_sigmas.txt'.format(zmin, zmax, amlim, dexis[i], xmax))
#     if c_param == '\Omega_m':
#         plt.plot(a_omegas, res, color='C{}'.format(n), label=r'{:1.2f} $<\xi<$ {:1.2f}'.format(dexis[i], xmax))
#     else:
#         plt.plot(a_sigmas, res, color='C{}'.format(n), label=r'{:1.2f} $<\xi<$ {:1.2f}'.format(dexis[i], xmax))
#     if c_param == '\Omega_m':
#         plt.errorbar(np.array(omegas)[sim_selec1[:3]], sim_res[:3, i] - sim_res[:3, l-1],
#                     pois_res[:3, i] - pois_res[:3, l-1], fmt='o',
#                     color='C{}'.format(n), label='MxSy'[:4 * (n == 0)])
#         plt.errorbar(np.array(omegas)[sim_selec1[-1]], sim_res[-1, i] - sim_res[-1, l-1],
#                      pois_res[-1, i] -pois_res[-1, l-1], fmt='s',
#                     color='C{}'.format(n), label='Illustris'[:8 * (n == 0)])
#         # plt.errorbar(np.array(omegas)[sim_selec1[-2]], sim_res[-2, i] - sim_res[-2, l-2], pois_res[-2, i], fmt='s',
#         #             color='C{}'.format(n), label='BolshoiP'[:7 * (n == 0)])
#
#         plt.errorbar(np.array(omegas)[sim_selec1[3]], sim_res[3, i] - sim_res[3, l-1],
#                      pois_res[3, i] - pois_res[3, l-1], fmt='v',
#                     color='C{}'.format(n), label='New HR sims'[:10 * (n == 0)])
#         print(sim_res[3, i], sim_res[3, l - 1])
#     elif om1 == 0.3:
#         plt.errorbar(np.array(sigmas)[sim_selec2[:3]], sim_res[:3, i] - sim_res[:3, l-1],
#                      pois_res[:3, i] - pois_res[:3, l-1], fmt='o',
#                     color='C{}'.format(n), label='MxSy'[:4 * (n == 0)])
#         plt.errorbar(np.array(sigmas)[sim_selec2[-1]], sim_res[-1, i] - sim_res[-1, l-1],
#                      pois_res[-1, i]-pois_res[-1, l-1], fmt='s',
#                     color='C{}'.format(n), label='Illustris'[:9 * (n == 0)])
#         # # plt.errorbar(np.array(omegas)[sim_selec1[-2]], sim_res[-2, i] - sim_res[-2, l-2], pois_res[-2, i], fmt='s',
#         # #             color='C{}'.format(n), label='BolshoiP'[:7 * (n == 0)])
#         # plt.errorbar(np.array(sigmas)[sim_selec2[3]], sim_res[3, i] - sim_res[3, l-1],
#         #              pois_res[3, i] - pois_res[3, l-1], fmt='v',
#         #             color='C{}'.format(n), label='New HR sim'[:10 * (n == 0)])
#         # print(sim_res[3, i], sim_res[3, l-1])
# plt.ylabel(r'dN(>$\xi$) [mergers/halo]')
# plt.xlabel(r'${}$'.format(c_param), size=15)
# plt.legend()
# plt.title(r'{}<z<{}, log $M/M_\odot$>{:2.1f}'.format(zmin, zmax, np.log10(amlim)), size=15)
# plt.savefig(r'new_integ_N_tot_combined_${}$_m{:2.1f}.pdf'.format(c_param, np.log10(amlim)), dpi=650, bbox_inches='tight', facecolor='white',
#             transparent=False)
#plt.show()
