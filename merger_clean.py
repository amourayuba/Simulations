from merger_rate import *
import matplotlib.pyplot as plt
import matplotlib as mpl

params = {'legend.fontsize': 7,
          'legend.handlelength': 2}
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)


################ TOTAL NUMBER OF MERGERS ################################################
#############--------------- MxSy Simulations --------------#############################

class Simulation:
    def __init__(self, name, om0, sig8):
        self.name = name
        self.om0 = om0
        self.sig8 = sig8

    def get_redshifts(self, mrate_path):
        if self.name[0] == 'M':
            return np.loadtxt(mrate_path + '/progs_desc/redshifts/M03S08.txt')
        else:
            return np.loadtxt(mrate_path + '/progs_desc/redshifts/{}.txt'.format(self.name))

    def get_desc_prog(self, mrate_path, snap, wpos=False):
        mds = np.load(mrate_path + '/progs_desc/{}_desc_mass_snap{}.npy'.format(self.name, snap), allow_pickle=True)
        mprg = np.load(mrate_path + '/progs_desc/{}_prog_mass_snap{}.npy'.format(self.name, snap), allow_pickle=True)
        if wpos:
            prog_pos = np.load(mrate_path + '/progs_desc/{}_prog_pos_snap{}.npy'.format(self.name, snap),
                               allow_pickle=True)
            return mds, mprg, prog_pos
        return mds, mprg

    def get_mrat(self, pgmasses, wpos=False, m=None, pos_s=None):
        if wpos:
            arg_mer = np.argmin(
                np.sum((pos_s[np.arange(len(pos_s)) != m, :] - pos_s[m, :]) ** 2, axis=1))
            if arg_mer >= m:
                arg_mer += 1
            return pgmasses[m] / pgmasses[arg_mer]
        else:
            pgratios = pgmasses / np.max(pgmasses)
            return pgratios[np.where(pgratios < 1)]

    def dndxi(self, mrate_path, snap, mlim, bins=20, ximin=1e-2, ximax=1.0, wpos=False):
        if wpos:
            mds, mprg, prog_pos = self.get_desc_prog(mrate_path, snap, wpos)
        else:
            mds, mprg = self.get_desc_prog(mrate_path, snap, wpos)
        mmin = np.min(mds)
        nmgs = np.zeros(bins)
        tnds = np.zeros(bins + 1)
        xis = np.logspace(np.log10(ximin), np.log10(ximax), bins + 1)

        for i in range(len(mds)):
            mass = mds[i]
            if mlim < mass < 1000 * mlim:
                if len(mprg[i]) > 0:
                    pgmasses = np.array(mprg[i])
                    if wpos:
                        pos_s = np.array(prog_pos[i])
                        if len(mprg[i]) == 1:
                            tnds += np.max(pgmasses) * xis > mmin
                    else:
                        tnds += np.max(pgmasses) * xis > mmin
                if len(mprg[i]) > 1:
                    if wpos:
                        for m in range(len(pgmasses)):
                            rat = self.get_mrat(pgmasses, wpos, m, pos_s)
                            if rat < 1:
                                for j in range(bins):
                                    if (rat > xis[j]) and (rat < xis[j + 1]):
                                        nmgs[j] += 1
                                        tnds += np.max(pgmasses) * xis > mmin
                    else:
                        rat = self.get_mrat(pgmasses, wpos)
                        for xs in rat:
                            for j in range(bins):
                                if (xs > xis[j]) and (xs < xis[j + 1]):
                                    nmgs[j] += 1
        return nmgs, tnds

    def N_of_xi(self, mrate_path, snaps, mlim, mlim_rat=1e3, ximin=0.01, resol=100, wpos=False):
        mres = np.zeros((resol, len(snaps)))
        tnds = np.zeros((resol, len(snaps)))
        dexis = np.logspace(np.log10(ximin), 0, resol)
        for k in range(len(snaps)):
            snap = snaps[k]
            if wpos:
                mds, mprg, prog_pos = self.get_desc_prog(mrate_path, snap, wpos)
            else:
                mds, mprg = self.get_desc_prog(mrate_path, snap, wpos)
            mmin = np.min(mds)
            for i in range(len(mds)):
                if mlim < mds[i] < mlim_rat * mlim:
                    if len(mprg[i]) > 0:
                        pgmasses = mprg[i]
                        xilim = mmin / np.max(pgmasses)
                        if wpos:
                            pos_s = np.array(prog_pos[i])
                            if len(mprg[i]) == 1:
                                tnds[:, k] += xilim < dexis
                        else:
                            tnds[:, k] += xilim < dexis
                    if len(mprg[i]) > 1:
                        if wpos:
                            for m in range(len(pgmasses)):
                                rat = self.get_mrat(pgmasses, wpos, m, pos_s)
                                if rat < 1:
                                    mres[:, k] += rat > dexis
                                    tnds[:, k] += xilim < dexis
                        else:
                            pgratios = pgmasses / np.max(pgmasses)
                            for rat in pgratios:
                                if rat < 1:
                                    mres[:, k] += rat > dexis

        return mres, tnds


# sim = 'M03S08'
# mrate_path = '/home/painchess/asus_fedora/merger_rate'
# sim1 = Simulation('M03S08', 0.3, 0.8)
# snapshots = [2, 6, 12, 20]
# mlim, bins, ximin, ximax = 5e13, 20, 1e-2, 1
# xis = np.logspace(np.log10(ximin), np.log10(ximax), bins + 1)
# dxis = xis[1:] - xis[:-1]
# m_reds = np.loadtxt('/home/painchess/asus_fedora/simulation_reading/simu_redshifts.txt')[::-1]
# colors = ['blue', 'red', 'green', 'orange']
# for i in range(len(snapshots)):
#     snap = snapshots[i]
#     nmgs, tds = sim1.dndxi(mrate_path, 4, mlim, bins, ximin, ximax, wpos=True)
#     dz = m_reds[snap + 1] - m_reds[snap]
#     y = nmgs / dz / dxis / tds[1:]
#     poisson = np.sqrt(nmgs) / dz / tds[1:] / dxis
#     plt.plot(xis[1:-1], ell_mrate_per_n(5 * mlim, m_reds[snap], xis), '--', color=colors[i], linewidth=2)
#     plt.scatter(xis[1:], y, color=colors[i], label='z={:1.1f}'.format(m_reds[snap]))
#     plt.fill_between(xis[1:], y - poisson, y + poisson, color=colors[i], alpha=0.2)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$\xi =M_1/M_2$', size=15)
# plt.ylabel(r'dN/dz/d$\xi$ [mergers/halo/dz]')
# plt.ylim(1e-2, 1e3)
# plt.title(r'$M_0$ = {:2.1e}'.format(mlim))
# plt.legend()
# plt.show()

ys, poisson = [], []
sim_names = ['M25S08', 'M03S08', 'M35S08', 'M03S07', 'M03S09', 'Illustris', 'bolshoiP', 'bolshoiW', 'm25s85',
             'm2s8', 'M25S07', 'M25S09']
omegas = [0.25, 0.3, 0.35, 0.3, 0.3, 0.309, 0.307, 0.27, 0.25, 0.2, 0.25, 0.25]
sigmas = [0.8, 0.8, 0.8, 0.7, 0.9, 0.816, 0.82, 0.82, 0.85, 0.8, 0.7, 0.9]

colors = ['blue', 'red', 'green', 'orange']
lss = ['-', '--', '-.']
markers = ['s', 'v', 'o', 'x']
resol, ximin, ximax = 100, 1e-2, 1
illustris_path = '/home/painchess/mounted/TNG300-625/output'
log_fracs = [0, 0.25, 0.5]  # log space fraction between ximin and ximax
max_fracs = [0.25, 0.5, 0.75]
ximins = (resol * np.array(log_fracs)).astype(int)
ximaxs = (resol * np.array(max_fracs)).astype(int)
dexis = np.logspace(np.log10(ximin), np.log10(ximax), resol)
sim_selec1 = np.array([0, 1, 2, 9, 5])
sim_selec2 = np.array([1, 3, 4, 9, 5])
sim_selec3 = np.array([0, -1, -2, -4])
zmin, zmax = 0.01, 0.45
sim_res, pois_res = np.zeros((len(sim_selec1), resol - 1)), np.zeros((len(sim_selec1), resol - 1))
c_param = '\sigma_8'
om1 = 0.3
for i in range(len(sim_selec1)):
    if c_param == '\Omega_m':
        j = sim_selec1[i]
    elif om1 == 0.3:
        j = sim_selec2[i]
    else:
        j = sim_selec3[i]
    sim1 = Simulation(sim_names[j], omegas[j], sigmas[j])
    mrate_path = '/home/painchess/asus_fedora/merger_rate'
    reds = sim1.get_redshifts(mrate_path)
    snap_min, snap_max = np.min(np.where(reds >= zmin)) + 1, np.min(np.where(reds >= zmax)) + 1
    snapshots = np.arange(snap_min, snap_max, 1)
    if sim1.name[0] == 'M':
        mlim = 5e13
    elif sim1.name[0] == 'I':
        mlim = 1e13
        mrate_path = illustris_path
    elif sim1.name[0] == 'm':
        mlim = 1e14
    else:
        mlim = 1e12
    mres, tnds = sim1.N_of_xi(mrate_path, snapshots, mlim, resol=resol, wpos=False)
    dmres = (mres[:-1, :] - mres[1:, :]) / tnds[1:, :]
    ps = np.sqrt((mres[:-1, :] - mres[1:, :])*(1+(mres[:-1, :] - mres[1:, :])/tnds[1:, :]))/tnds[1:, :]
    #ps = np.sqrt(mres[:-1, :] - mres[1:, :])*(1-1/np.sqrt(tnds[1:, :]))/tnds[1:, :]
    #ps = np.sqrt(mres[:-1, :] - mres[1:, :]) / tnds[1:, :]
    nmres = np.cumsum(dmres[::-1], axis=0)[::-1, :]
    nps = np.sqrt(np.cumsum(ps[::-1]**2, axis=0))[::-1, :]

    tot_integ = np.sum(nmres, axis=1)
    #tot_ps = np.sqrt(np.sum(nps**2, axis=1))
    tot_ps = np.sum(nps, axis=1)
    sim_res[i, :] = tot_integ
    pois_res[i, :] = tot_ps

# fig, axs = plt.subplots(2, 2, figsize=[7, 7], dpi=500)
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
#     ax = axs[n // 2, n % 2]
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
#         ax.errorbar(np.array(omegas)[sim_selec1[4]], sim_res[-1, i] - sim_res[-1, l-2], pois_res[-1, i], fmt='s',
#                     label='Illustris'[:8 * (n == 0)])
#         ax.errorbar(np.array(omegas)[sim_selec1[3:-1]], sim_res[3:-1, i] - sim_res[3:-1, l-2], pois_res[3:-1, i], fmt='v',
#                     label='New HR sims'[:10 * (n == 0)])
#     elif om1 == 0.3:
#         ax.errorbar(np.array(sigmas)[sim_selec2[:3]], sim_res[:3, i] - sim_res[:3, l-2], pois_res[:3, i], fmt='o',
#                     label='MxSy'[:4 * (n == 0)])
#         ax.errorbar(np.array(sigmas)[sim_selec2[4]], sim_res[4, i] - sim_res[4, l-2], pois_res[4, i], fmt='s',
#                     label='Illustris'[:9 * (n == 0)])
#         ax.errorbar(np.array(sigmas)[sim_selec2[3]], sim_res[3, i] - sim_res[3, l-2], pois_res[3, i], fmt='v',
#                     label='New HR sim'[:10 * (n == 0)])
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


#fig, axs = plt.subplots(2, 2, figsize=[7, 7], dpi=500)
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
    #ax = axs[n // 2, n % 2]
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
                s8 = 0.8
            else:
                s8 = a_sigmas[m]
                omg = 0.3
            for k in range(zresol):
                diff_nmerg[k] = integ_mrate(3 * amlim, nreds[k], xi_min=dexis[i], xi_max=xmax, om0=omg, sig8=s8)
            res.append(np.sum(diff_nmerg) * dz)
        if c_param == '\Omega_m':
            np.savetxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}'
                       '_ximax{:1.2f}_omegas.txt'.format(zmin, zmax, amlim, dexis[i], xmax), np.array(res))
        else:
            np.savetxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}'
                       '_ximax{:1.2f}_sigmas.txt'.format(zmin, zmax, amlim, dexis[i], xmax), np.array(res))

    else:
        if c_param == '\Omega_m':
            res = np.loadtxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_ximin{:1.2f}_'
                             'ximax{:1.2f}_omegas.txt'.format(zmin, zmax, amlim, dexis[i], xmax))
        else:
            res = np.loadtxt('anal_ntotmerg_zmin{}_zmax{:1.1f}_mlim{:2.1e}_'
                             'ximin{:1.2f}_ximax{:1.2f}_sigmas.txt'.format(zmin, zmax, amlim, dexis[i], xmax))
    if c_param == '\Omega_m':
        plt.plot(a_omegas, res, color='C{}'.format(n), label=r'{:1.2f} $<\xi<$ {:1.2f}'.format(dexis[i], xmax))
    else:
        plt.plot(a_sigmas, res, color='C{}'.format(n), label=r'{:1.2f} $<\xi<$ {:1.2f}'.format(dexis[i], xmax))
    if c_param == '\Omega_m':
        plt.errorbar(np.array(omegas)[sim_selec1[:3]], sim_res[:3, i] - sim_res[:3, l-2], pois_res[:3, i], fmt='o',
                    color='C{}'.format(n), label='MxSy'[:4 * (n == 0)])
        plt.errorbar(np.array(omegas)[sim_selec1[-1]], sim_res[-1, i] - sim_res[-1, l-2], pois_res[-1, i], fmt='s',
                    color='C{}'.format(n), label='Illustris'[:8 * (n == 0)])
        # plt.errorbar(np.array(omegas)[sim_selec1[-2]], sim_res[-2, i] - sim_res[-2, l-2], pois_res[-2, i], fmt='s',
        #             color='C{}'.format(n), label='BolshoiP'[:7 * (n == 0)])

        plt.errorbar(np.array(omegas)[sim_selec1[3]], sim_res[3, i] - sim_res[3, l-2], pois_res[3, i], fmt='v',
                    color='C{}'.format(n), label='New HR sims'[:10 * (n == 0)])
    elif om1 == 0.3:
        plt.errorbar(np.array(sigmas)[sim_selec2[:3]], sim_res[:3, i] - sim_res[:3, l-2], pois_res[:3, i], fmt='o',
                    color='C{}'.format(n), label='MxSy'[:4 * (n == 0)])
        plt.errorbar(np.array(sigmas)[sim_selec2[-1]], sim_res[-1, i] - sim_res[-1, l-2], pois_res[-1, i], fmt='s',
                    color='C{}'.format(n), label='Illustris'[:9 * (n == 0)])
        # plt.errorbar(np.array(omegas)[sim_selec1[-2]], sim_res[-2, i] - sim_res[-2, l-2], pois_res[-2, i], fmt='s',
        #             color='C{}'.format(n), label='BolshoiP'[:7 * (n == 0)])
        plt.errorbar(np.array(sigmas)[sim_selec2[3]], sim_res[3, i] - sim_res[3, l-2], pois_res[3, i], fmt='v',
                    color='C{}'.format(n), label='New HR sim'[:10 * (n == 0)])

plt.ylabel(r'dN(>$\xi$) [mergers/halo]')
plt.xlabel(r'${}$'.format(c_param), size=15)
plt.legend()
plt.title(r'{}<z<{}, log $M/M_\odot$>{:2.1f}'.format(zmin, zmax, np.log10(amlim)), size=15)
#plt.savefig(r'integ_N_tot_combined_${}$_m{:2.1f}.pdf'.format(c_param, np.log10(amlim)), dpi=650, bbox_inches='tight', facecolor='white',
            #transparent=False)
plt.show()


# fig, axs = plt.subplots(2, 2, figsize=[7, 7], dpi=500)
#
# for n in range(len(ximins)):
#     ax = axs[n // 2, n % 2]
#     i = ximins[n]
#     for j in range(len(ys)):
#         y = ys[j]
#         ps = poisson[j]
#         om = omegas[j]
#         res = []
#         for k in range(len(snapshots)):
#             res.append(integ_mrate(3*mlim, m_reds[snapshots[k]], xi_min=dexis[i], xi_max=1, om0=om, sig8=s8))
#         if j == 0 and n == 0:
#             ax.plot(1 + m_reds[snapshots], res, color=colors[j], ls=lss[j], label=r'EC $\Omega_m$={:1.2f}'.format(om))
#             ax.scatter(1 + m_reds[snapshots], y[i], color=colors[j], marker=markers[i],
#                        label=r'$\xi>$ {:1.2f}, $\Omega_m$={:1.2f}'.format(dexis[i], om))
#         elif j == 0:
#             ax.scatter(1 + m_reds[snapshots], y[i], color=colors[j], marker=markers[j],
#                        label=r'$\xi>$ {:1.2f}'.format(dexis[i]))
#             ax.plot(1 + m_reds[snapshots], res, color=colors[j], ls=lss[j])
#         elif n == 0:
#             ax.plot(1 + m_reds[snapshots], res, color=colors[j], ls=lss[j], label=r'EC $\Omega_m$={:1.2f}'.format(om))
#             ax.scatter(1 + m_reds[snapshots], y[i], color=colors[j], marker=markers[j],
#                        label=r'$\Omega_m$={:1.2f}'.format(om))
#         else:
#             ax.scatter(1 + m_reds[snapshots], y[i], color=colors[j], marker=markers[j])
#             ax.plot(1 + m_reds[snapshots], res, color=colors[j], ls=lss[j])
#
#         ax.fill_between(1 + m_reds[snapshots], y[i] - ps[i], y[i] + ps[i], color=colors[j], alpha=0.2)
#
#     ax.set_xlabel('1+z', size=15)
#     ax.set_ylabel(r'dN(>$\xi$/dz [mergers/halo/dz]')
#
#     ax.legend()
# plt.show()
