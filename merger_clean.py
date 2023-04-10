from merger_rate import *
from scipy.optimize import curve_fit
from astropy.cosmology import LambdaCDM, z_at_value
from astropy import units as u
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

params = {'legend.fontsize': 7,
          'legend.handlelength': 2}
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)


################ TOTAL NUMBER OF MERGERS ################################################
#############--------------- MxSy Simulations --------------#############################

def mcbride_mah(z, gamma, beta):
    return (1 + z) ** beta * np.exp(-gamma * z)


def get_mrat(pgmasses, wpos=False, m=None, pos_s=None):
    if wpos:
        arg_mer = np.argmin(
            np.sum((pos_s[np.arange(len(pos_s)) != m, :] - pos_s[m, :]) ** 2, axis=1))
        if arg_mer >= m:
            arg_mer += 1
        return pgmasses[m] / pgmasses[arg_mer]
    else:
        pgratios = pgmasses / np.max(pgmasses)
        return pgratios[np.where(pgratios < 1)]


def large_growth_analytical(Mass, zs, s8, om, nxibins=10000):
    res_ana = []
    dz = (zs[0] - zs[-1]) / len(zs)
    if type(Mass) == np.ndarray or type(Mass) == list:
        for M0 in Mass:
            res = []
            for red in zs:
                res.append(integ_mrate(M0, red, 1 / 3, 1, nxibins=nxibins, mass=False, sig8=s8, om0=om, ol0=1 - om))
            res_ana.append(np.sum(np.array(res)) * dz)
        return np.array(res_ana)
    else:
        res = []
        for red in zs:
            res.append(integ_mrate(Mass, red, 1 / 3, 1, nxibins=nxibins, mass=False, sig8=s8, om0=om, ol0=1 - om))
        return np.sum(np.array(res)) * dz


def average_growth_analytical(Mass, zs, s8, om, nxibins=10000):
    res_ana = []
    dz = (zs[0] - zs[-1]) / len(zs)
    if type(Mass) == np.ndarray or type(Mass) == list:
        for M0 in Mass:
            res = []
            for red in zs:
                res.append(integ_mrate(M0, red, 1e-14, 1, nxibins=nxibins, mass=True, sig8=s8, om0=om, ol0=1 - om) / M0)
            res_ana.append(np.sum(np.array(res)) * dz)
        return np.array(res_ana)
    else:
        res = []
        for red in zs:
            res.append(integ_mrate(Mass, red, 1e-14, 1, nxibins=nxibins, mass=True, sig8=s8, om0=om, ol0=1 - om) / Mass)
        return np.sum(np.array(res)) * dz


def get_zlastdyn(zf, h, om, zbins=20):
    cosmol = LambdaCDM(H0=100 * h, Om0=om, Ode0=1 - om)
    infall = 1.44 / hubble_ratio(zf, omega_m0=om, omega_l0=1 - om)  # infall time in gyr
    ages = cosmol.age(zf).value
    last_tdyn = ages - np.sqrt(2) * infall
    zi = z_at_value(cosmol.age, last_tdyn * u.Gyr)
    zs = np.linspace(zi, zf, zbins)
    dz = (zi - zf) / zbins
    return zs.value, dz.value


class Simulation:
    """A class of objects referring to a Dark Matter only simulation"""

    def __init__(self, name, om0, sig8, path):
        self.name = name  # name of the simulation
        self.om0 = om0  # Value of Omega_m for the simulation
        self.sig8 = sig8  # Value of sigma_8
        self.path = path  # The base (absolute) path of that simulation. All simulation paths have to be structured the same.
        self.simpath = self.path + self.name
    #################----------------GENERIC INFORMATION OF THE SIMULATION---------------------#########################

    def get_redshifts(self):
        """Gets the list of redshifts of each snapshot"""
        if self.name[0] == 'M':
            return np.loadtxt(self.path + 'redshifts/M03S08.txt')
        elif os.path.exists(self.path + 'redshifts/{}.txt'.format(self.name)):
            return np.loadtxt(self.path + 'redshifts/{}.txt'.format(self.name))
        else:
            return np.loadtxt(self.path + 'redshifts/mxsy_reds.txt')

    def get_prefs(self):
        """Gets the list of prefixes of each file for all available snapshots"""
        with open(self.path + self.name + '/' + self.name + '_prefixes.txt') as file:
            prefs = file.read().splitlines()  # Names of each snapshot prefix.
        return prefs

    def read_halos(self, snapshot=0):
        """Read the halo file associated with this simulation and snapshot"""
        prefs = self.get_prefs()
        return pd.read_table(self.path + self.name + '/halos/' + prefs[snapshot] + '.AHF_halos', delim_whitespace=True,
                             header=0)

    def get_subfrac(self, snapshot=0):
        """Gives the substructure fraction of all halos at the given snapshot
        :returns array (nhalo, 4) with (halo_idx, halo_mass, sub fraction, n subs)"""
        prefs = self.get_prefs()
        with open(self.path + self.name + '/substructure/' + prefs[snapshot] + '.AHF_substructure') as file:
            lines = file.read().splitlines()
        for i in range(len(lines)):
            lines[i] = lines[i].split()
        halo_nsubs = lines[::2]
        subs_list = lines[1::2]

        halos = self.read_halos(snapshot).set_index('#ID(1)')
        nh = len(halo_nsubs)
        subfrac = np.zeros((nh, 4))
        for i in range(nh):
            idx = int(halo_nsubs[i][0])
            mhalo = halos.loc[idx]['Mhalo(4)']
            subfrac[i, 0] = idx
            subfrac[i, 1] = mhalo
            msubs = 0

            for sub in subs_list[i]:
                try:  # sometimes the halo hasn't been recorded in _halos file, these cases are very rare
                    msubs += halos.loc[int(sub)]['Mhalo(4)']
                except KeyError:
                    msubs += 1e12  # this is the minimum halo mass
            subfrac[i, 2] = msubs / mhalo
            subfrac[i, 3] = int(int(halo_nsubs[i][1]))
        return subfrac

    ###################-----------------MASS ACCRETION HISTORY RELATED QUANTITIES----------------#######################
    def make_mah(self, save=False, fsizelim=1500):
        mahs, ids, emptyfiles = [], [], []
        filenames = os.listdir(self.path + self.name + '/mahs')
        filenames.sort()
        for file in filenames:
            if os.path.getsize(
                    self.path + self.name + '/mahs/' + file) > fsizelim:  # check that file is has actual halos
                #mahs.append(np.loadtxt(self.path + self.name + '/mahs/' + file)[:, 4])
                #ids.append(np.loadtxt(self.path + self.name + '/mahs/' + file)[:, 1][0])
                mahs.append(np.loadtxt(self.path + self.name + '/mahs/' + file)[:, 4])
                with open(self.path + self.name + '/mahs/' + file, 'r') as f:
                    for _ in range(3):
                        l = f.readline()
                    ids.append((int(l.split()[1])))
            else:
                # print('file '+file+' is empty and has size ',os.path.getsize(self.path+self.name+'/mahs/'+file)/1024,'Kb')
                emptyfiles.append(file)
        if save:
            np.save(self.path + self.name + '/mahs_{}.npy'.format(self.name), np.array(mahs, dtype=object))
            np.save(self.path + self.name + '/ids_{}.npy'.format(self.name), np.array(ids, dtype=int))
        return mahs, ids, emptyfiles

    def get_mah(self):
        """Gets the Mass Accretion History in the form of an .npy file with Nhalos elements, each one being a list of
        masses starting from z=0 """
        return np.load(self.path + '/{}/mahs_{}.npy'.format(self.name, self.name), allow_pickle=True)

    def get_mah_ids(self):
        """Gets the associated halo ids of the Mass Accretion History in the form of an .npy file with Nhalos
        elements, each one being a list of masses starting from z=0 """
        return np.load(self.path + '/{}/ids_{}.npy'.format(self.name, self.name))

    ###########################---------------CALCULATIONS WITH MAHS-------------------#################################

    def make_zxs(self, z, save=True, minsnapdist=5, jump_frac=[30, 25, 20, 10], snapnums=[11, 27, 59, 79, 89],
                 fracs=[90, 75, 50, 10, 1]):

        mahs = self.get_mah()
        idxs = self.get_mah_ids()
        reds = self.get_redshifts()
        nhalos = len(mahs)
        if nhalos != len(idxs):
            raise ValueError("Problem with MAHs and Ids not the same length")
        snapi = np.min(np.where(reds >= z))

        mofz_snaps = np.where(np.array(snapnums) > snapi)[0]

        if len(mofz_snaps) > 0:
            mofz = True
            numi = np.min(mofz_snaps)
            rsnaps = snapnums[numi:]
            mofzs = np.zeros((nhalos, len(rsnaps)))
        else:
            mofz = False

        indices_i, gamma, beta, mass_i = [], [], [], []

        zx_i = dict(zip(fracs, [[] for _ in range(len(fracs))]))
        zmm_i = dict(zip(jump_frac, [[] for _ in range(len(jump_frac))]))

        for i in range(len(mahs)):
            mah = mahs[i][snapi:]
            zeds = reds[snapi:len(mah) + snapi]
            if len(mah) > minsnapdist:
                m = mah[0]
                if mofz:
                    for k in range(len(rsnaps)):
                        if len(mah) > rsnaps[k]:
                            mofzs[i, k] = mah[rsnaps[k]] / m

                immfracs = mah[:-minsnapdist] / mah[minsnapdist:] - 1
                for p in range(len(jump_frac)):
                    frc = jump_frac[p] / 100
                    jump_idx = np.where(immfracs >= frc)[0]
                    if len(jump_idx) > 0:
                        res = np.sqrt(zeds[np.min(jump_idx)]*zeds[np.min(jump_idx)+minsnapdist])
                        zmm_i[jump_frac[p]].append(res)
                    else:
                        zmm_i[jump_frac[p]].append(123)

                for j in range(len(fracs)):
                    frac = fracs[j] / 100
                    ids = np.where(mah / m < frac)
                    if len(ids[0]) > 0:
                        idx = np.min(np.where(mah / m < frac))
                        zx = np.sqrt(zeds[idx] * zeds[idx - 1])
                        zx_i[fracs[j]].append(zx)
                    else:
                        zx_i[fracs[j]].append(np.inf)
                try:
                    param, cov = curve_fit(mcbride_mah, zeds, mah / m)
                    gamma.append(param[0])
                    beta.append(param[1])
                except RuntimeError:
                    gamma.append(-1e50)
                    beta.append(-1e50)

                mass_i.append(m)
                indices_i.append(idxs[i])

        mass_i, indices_i, gamma, beta = np.array(mass_i), np.array(indices_i, dtype=int), np.array(gamma), np.array(beta)

        nameszx = ['z{}'.format(frac) for frac in fracs]
        nameszmm = ['zmm{}'.format(jfrac) for jfrac in jump_frac]
        namesmofz = ['M/M0(z={:1.1f})'.format(reds[rsnp]) for rsnp in rsnaps]
        names = ['McBride_gamma', 'McBride_beta', 'Mass', 'Halo_index']

        zxt = pd.DataFrame(zx_i)
        zxt.columns = nameszx

        zmmt = pd.DataFrame(zmm_i)
        zmmt.columns = nameszmm

        mofzt = pd.DataFrame(mofzs, columns=namesmofz)

        tosave = dict(zip(names, [gamma, beta, mass_i, indices_i]))
        otht = pd.DataFrame(tosave)

        if save:
            zxt.to_csv(self.path + '/{}/zxt_{}_z{}.dat'.format(self.name, self.name, z))
            zmmt.to_csv(self.path + '/{}/zmmt_{}_z{}.dat'.format(self.name, self.name, z))
            mofzt.to_csv(self.path + '/{}/mofzt_{}_z{}.dat'.format(self.name, self.name, z))
            otht.to_csv(self.path + '/{}/otht_{}_z{}.dat'.format(self.name, self.name, z))
        return zxt, zmmt, mofzt, otht

    def get_agedata(self, z, atype='oth'):
        return pd.read_csv(self.path + '/{}/{}t_{}_z{}.dat'.format(self.name, atype, self.name, z))

    def average_growth(self, zf, mmin=1e12, mmax=1e14, mbins=20, zbins=20, save=False):
        """Calculates the average growth over the last dynamical timescale of halos between mmin and mmax at zf"""
        mahs = self.get_mah()
        zs, dz = get_zlastdyn(zf, h=0.7, om=self.om0, zbins=zbins)
        zi = np.max(zs)
        reds = self.get_redshifts()
        masses = np.logspace(np.log10(mmin), np.log10(mmax), mbins)
        res_sim = np.zeros(mbins)
        ntot_sim = np.zeros(mbins)
        for j in range(len(mahs)):
            mah = mahs[j]
            zeds = reds[:len(mah)]
            if zeds[-1] > zi:
                zbin_min, zbin_max = np.min(np.where(zeds > zf)), np.min(
                    np.where(zeds > zi))  # find the snapshot bin corresponding to zf
                if mah[zbin_min] < masses[-1]:
                    mbin = np.min(np.where(masses > mah[zbin_min]))
                    if mah[zbin_min] > mah[zbin_max] and mah[zbin_min] / mah[zbin_max] < 5:
                        res_sim[mbin] += mah[zbin_min] / mah[zbin_max] - 1
                        ntot_sim[mbin] += 1
        ps_sim = np.sqrt(res_sim * (1 + res_sim / ntot_sim))
        if save:
            np.savetxt(self.path + '/{}/mah_mergers/av_growth_sim_{:2.1f}_sim_{}_nbins{}'
                       .format(self.name, zf, self.name, mbins), np.array(res_sim))
            np.savetxt(self.path + '/{}/mah_mergers/av_growth_ntot_sim_{:2.1f}_sim_{}_nbins{}'
                       .format(self.name, zf, self.name, mbins), np.array(ntot_sim))
            np.savetxt(self.path + '/{}/mah_mergers/av_growth_ps_sim_{:2.1f}_sim_{}_nbins{}'
                       .format(self.name, zf, self.name, mbins), np.array(ps_sim))
        return res_sim, ntot_sim, ps_sim

    def large_growth(self, zf, mmin=1e12, mmax=1e14, mbins=20, zbins=20, save=False):
        """Calculates the fraction of halos at zf that had a major merger during the last dynamical timescale"""
        mahs = self.get_mah()
        zs, dz = get_zlastdyn(zf, h=0.7, om=self.om0, zbins=zbins)
        zi = np.max(zs)
        reds = self.get_redshifts()
        masses = np.logspace(np.log10(mmin), np.log10(mmax), mbins)
        res_sim = np.zeros(mbins)
        ntot_sim = np.zeros(mbins)
        for j in range(len(mahs)):
            mah = mahs[j]
            zeds = reds[:len(mah)]
            if zeds[-1] > zi:
                zbin_min, zbin_max = np.min(np.where(zeds > zf)), np.min(
                    np.where(zeds > zi))  # find the snapshot bin corresponding to zf
                if mah[zbin_min] < masses[-1]:
                    mbin = np.min(np.where(masses > mah[zbin_min]))
                    ntot_sim[mbin] += 1
                    if 1.333 < mah[zbin_min] / mah[zbin_max] < 2:
                        res_sim[mbin] += 1
        ps_sim = np.sqrt(res_sim * (1 + res_sim / ntot_sim))
        if save:
            np.savetxt(self.path + '/{}/mah_mergers/large_growth_sim_{:2.1f}_sim_{}_nbins{}'
                       .format(self.name, zf, self.name, mbins), np.array(res_sim))
            np.savetxt(self.path + '/{}/mah_mergers/large_growth_ntot_sim_{:2.1f}_sim_{}_nbins{}'
                       .format(self.name, zf, self.name, mbins), np.array(ntot_sim))
            np.savetxt(self.path + '/{}/mah_mergers/large_growth_ps_sim_{:2.1f}_sim_{}_nbins{}'
                       .format(self.name, zf, self.name, mbins), np.array(ps_sim))
        return res_sim, ntot_sim, ps_sim

    def load_large_growth(self, mbins, zf):
        res_sim = np.loadtxt(self.path + '/{}/mah_mergers/large_growth_sim_{:2.1f}_sim_{}_nbins{}'
                             .format(self.name, zf, self.name, mbins))
        ntot_sim = np.loadtxt(self.path + '/{}/mah_mergers/large_growth_ntot_sim_{:2.1f}_sim_{}_nbins{}'
                              .format(self.name, zf, self.name, mbins))
        ps_sim = np.loadtxt(self.path + '/{}/mah_mergers/large_growth_ps_sim_{:2.1f}_sim_{}_nbins{}'
                            .format(self.name, zf, self.name, mbins))
        return res_sim, ntot_sim, ps_sim

    def load_av_growth(self, mbins, zf):
        res_sim = np.loadtxt(self.path + '/{}/mah_mergers/av_growth_sim_{:2.1f}_sim_{}_nbins{}'
                             .format(self.name, zf, self.name, mbins))
        ntot_sim = np.loadtxt(self.path + '/{}/mah_mergers/av_growth_ntot_sim_{:2.1f}_sim_{}_nbins{}'
                              .format(self.name, zf, self.name, mbins))
        ps_sim = np.loadtxt(self.path + '/{}/mah_mergers/av_growth_ps_sim_{:2.1f}_sim_{}_nbins{}'
                            .format(self.name, zf, self.name, mbins))
        return res_sim, ntot_sim, ps_sim

    ###############-----------------MERGERS AND MERGER RATES SECTION--------------------################################
    def get_desc_prog(self, snap, wpos=False):
        """ Gets the descendant and progenitor masses if they are already saved"""
        mds = np.load(self.path + '/progs_desc/{}_desc_mass_snap{}.npy'.format(self.name, snap), allow_pickle=True)
        mprg = np.load(self.path + '/progs_desc/{}_prog_mass_snap{}.npy'.format(self.name, snap), allow_pickle=True)
        if wpos:
            prog_pos = np.load(self.path + '/progs_desc/{}_prog_pos_snap{}.npy'.format(self.name, snap),
                               allow_pickle=True)  # Gets the prog positions if needed.
            return mds, mprg, prog_pos
        return mds, mprg

    def make_prog_desc(self, snapshot, usepos=False, save=True):
        """Makes a list of descendant masses and each corresponding prog mass"""
        f_halos = self.path + self.name + '/halos/'  # Typically where the halos are stored
        f_mtrees = self.path + self.name + '/mtrees/'  # Typically where the merger trees are stored
        with open(self.path + self.name + '/' + self.name + '_prefixes.txt') as file:
            prefs = file.read().splitlines()  # Names of each snapshot prefix.

        d_halos = pd.read_table(f_halos + prefs[snapshot] + '.AHF_halos', delim_whitespace=True, header=0)
        p_halos = pd.read_table(f_halos + prefs[snapshot + 1] + '.AHF_halos', delim_whitespace=True, header=0)

        desc_mass = np.array(d_halos['Mhalo(4)'])  # halo masses at snap
        prog_mass = np.array(p_halos['Mhalo(4)'])  # halo masses at snap+1

        # desc_mass = np.loadtxt(f_halos + prefs[snapshot] + '.AHF_halos')[:, 3]
        # prog_mass = np.loadtxt(f_halos + prefs[snapshot + 1] + '.AHF_halos')[:, 3]
        id4_desc = np.loadtxt(f_halos + prefs[snapshot] + '.AHF_halos')[4, 0]  # random halo id at snap
        id4_prog = np.loadtxt(f_halos + prefs[snapshot + 1] + '.AHF_halos')[4, 0]  # random halo id at snap+1

        if id4_desc > len(desc_mass):  # test whether ids are from 0 tp len(desc_mass) or not
            dic_desc = dict(zip(np.array(d_halos['#ID(1)']).astype(int),
                                desc_mass))  # If ids are not ranked, need to use a dictionnary for quick access

        if id4_prog > len(prog_mass):
            dic_prog = dict(zip(np.array(p_halos['#ID(1)']).astype(int), prog_mass))

        if usepos:
            Xcs, Ycs, Zcs = np.loadtxt(f_halos + prefs[snapshot + 1] + '.AHF_halos')[:, 5], np.loadtxt(
                f_halos + prefs[snapshot + 1] + '_halos')[:, 6], np.loadtxt(f_halos + prefs[snapshot + 1] + '_halos')[:,
                                                                 7]
        mmin = np.min(desc_mass)
        desc, progs = [], []
        with open(f_mtrees + prefs[snapshot] + '.AHF_mtree') as file:
            lines = csv.reader(file, delimiter=' ', skipinitialspace=True)
            next(lines)  # skips the first line
            for j in range(len(desc_mass)):  # loop over desc
                try:
                    desc_id, npr = next(lines)
                    desc.append(int(desc_id))
                    hprog = []
                    for i in range(int(npr)):  # for each desc halo, save ids of all its progenitors
                        hprog.append(int(next(lines)[0]))
                    progs.append(hprog)
                except StopIteration:
                    val = 5
        mds, mprg, pos_prg = [], [], []
        for i in range(len(desc)):
            if id4_desc > len(desc):
                mds.append(dic_desc[desc[i]])
            else:
                mds.append(desc_mass[desc[i]])
            if id4_prog > len(progs):
                prg = []
                for prid in progs[i]:
                    prg.append(dic_prog[prid])
                mprg.append(prg)
            else:
                mprg.append(prog_mass[progs[i]])
            if usepos:
                pos_prg.append(np.array([Xcs[progs[i]], Ycs[progs[i]], Zcs[progs[i]]]).transpose())
        del (desc_mass, prog_mass, desc, progs)

        if save:
            np.save(self.path + '/progs_desc/{}_desc_mass_snap{}.npy'.format(self.name, snapshot),
                    np.array(mds, dtype=object))
            np.save(self.path + '/progs_desc/{}_prog_mass_snap{}.npy'.format(self.name, snapshot),
                    np.array(mprg, dtype=object))
            if usepos:
                np.save(self.path + '/progs_desc/{}_prog_pos_snap{}.npy'.format(self.name, snapshot),
                        np.array(pos_prg, dtype=object))
        if usepos:
            return np.array(mds, dtype=object), np.array(mprg, dtype=object), np.array(pos_prg, dtype=object)
        return np.array(mds, dtype=object), np.array(mprg, dtype=object)

    def dndxi(self, snap, mlim, bins=20, ximin=1e-2, ximax=1.0, wpos=False):
        if wpos:
            mds, mprg, prog_pos = self.get_desc_prog(snap, wpos)
        else:
            mds, mprg = self.get_desc_prog(snap, wpos)
        mmin = np.min(mds)
        nmgs = np.zeros(bins)
        tnds = np.zeros(bins + 1)
        xis = np.logspace(np.log10(ximin), np.log10(ximax), bins + 1)

        for i in range(len(mds)):
            mass = mds[i]
            if mlim < mass < 1000 * mlim:
                # if len(mprg[i]) > 0:
                #     pgmasses = np.array(mprg[i])
                #     if wpos:
                #         pos_s = np.array(prog_pos[i])
                #         if len(mprg[i]) == 1:
                #             tnds += np.max(pgmasses) * xis > mmin
                #     else:
                #         tnds += np.max(pgmasses) * xis > mmin

                tnds += xis > mmin / (mass - mmin)
                if len(mprg[i]) > 1:
                    pgmasses = np.array(mprg[i])
                    if wpos:
                        for m in range(len(pgmasses)):
                            pos_s = np.array(prog_pos[i])
                            rat = get_mrat(pgmasses, wpos, m, pos_s)
                            if rat < 1:
                                for j in range(bins):
                                    if (rat > xis[j]) and (rat < xis[j + 1]):
                                        nmgs[j] += 1
                                        tnds += np.max(pgmasses) * xis > mmin
                    else:
                        rat = get_mrat(pgmasses, wpos)
                        for xs in rat:
                            for j in range(bins):
                                if (xs > xis[j]) and (xs < xis[j + 1]):
                                    nmgs[j] += 1
        return nmgs, tnds

    def N_of_xi(self, snaps, mlim, mlim_rat=1e3, ximin=0.01, resol=100, wpos=False):
        mres = np.zeros((resol, len(snaps)))
        tnds = np.zeros((resol, len(snaps)))
        dexis = np.logspace(np.log10(ximin), 0, resol)
        for k in range(len(snaps)):
            snap = snaps[k]
            if wpos:
                mds, mprg, prog_pos = self.get_desc_prog(snap, wpos)
            else:
                mds, mprg = self.get_desc_prog(snap, wpos)
            mmin = np.min(mds)
            for i in range(len(mds)):
                if mlim < mds[i] < mlim_rat * mlim:
                    # if len(mprg[i]) > 0:
                    #     pgmasses = mprg[i]
                    #     xilim = mmin / np.max(pgmasses)
                    #     if wpos:
                    #         pos_s = np.array(prog_pos[i])
                    #         if len(mprg[i]) == 1:
                    #             tnds[:, k] += xilim < dexis
                    #     else:
                    #         tnds[:, k] += xilim < dexis

                    xilim = 1 / (mds[i] / mmin - 1)
                    # print(xilim)
                    tnds[:, k] += xilim < dexis
                    if len(mprg[i]) > 1:
                        pgmasses = mprg[i]
                        if wpos:
                            pos_s = np.array(prog_pos[i])
                            for m in range(len(pgmasses)):
                                rat = get_mrat(pgmasses, wpos, m, pos_s)
                                if rat < 1:
                                    mres[:, k] += rat > dexis
                                    tnds[:, k] += xilim < dexis
                        else:
                            pgratios = pgmasses / np.max(pgmasses)
                            for rat in pgratios:
                                if rat < 1:
                                    mres[:, k] += rat > dexis

        return mres, tnds

    ############################-------------VARIOUS PLOTTING CHECKS, HMF, CONC-----------##############################

    def plot_tests(self, test, snaps, vol=500, h=0.7, conctype=1):
        f_halos = self.path + self.name + '/halos/'
        with open(self.path + self.name + '/' + self.name + '_prefixes.txt') as file:
            prefs = file.read().splitlines()
        reds = self.get_redshifts()
        if test == 'hmf':
            plt.figure()
            plt.xlabel('Mass', size=20)
            plt.ylabel('N halos/mass dex/Mpc^3')
            plt.xscale('log')
            plt.yscale('log')
            for j in range(len(snaps)):
                snapshot = snaps[j]
                zn = reds[snapshot]
                halos = pd.read_table(f_halos + prefs[snapshot] + '.AHF_halos', delim_whitespace=True, header=0)
                print(prefs[snapshot], zn)
                mass_ahf = np.array(halos['Mhalo(4)'])
                bins = np.logspace(np.log10(np.min(mass_ahf)), np.log10(np.max(mass_ahf)), 80)
                unit = np.log(bins[1:] / bins[:-1])
                hist = np.histogram(mass_ahf, bins=bins)
                hist_dens = hist[0] / vol ** 3
                my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': self.om0, 'Ode0': 1 - self.om0, 'Ob0': 0.0482,
                            'sigma8': self.sig8,
                            'ns': 0.965}
                cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
                # truebins = np.sqrt(bins[:-1] * bins[1:])
                truebins = bins[1:]
                mfunc = mass_function.massFunction(truebins, zn, mdef='200c', model='tinker08', q_out='dndlnM')

                plt.plot(truebins, hist_dens / unit[0], '-.', color='C{}'.format(j),
                         label='1024'[:3 * (j == 0)] + ' {} snapshot={}'.format(self.name, snapshot))
                plt.plot(truebins, mfunc, color='C{}'.format(j), label='Colossus'[:8 * (j == 0)])
            plt.legend()
            plt.show()

        if test == 'conc':
            for j in range(len(snaps)):
                snapshot = snaps[j]
                halos = pd.read_table(f_halos + prefs[snapshot] + '.AHF_halos', delim_whitespace=True, header=0)
                if conctype == 1:
                    conc = np.array(halos['cNFW(43)'])
                if conctype == 2:
                    Rvir = np.array(halos['Rhalo(12)'])
                    Vvir = np.sqrt(G * mass_ahf / (0.001 * Rvir))
                    Vmax = np.array(halos['Vmax(17)'])
                    conc = Vmax / Vvir

                mass_ahf = np.array(halos['Mhalo(4)'])
                plt.figure()
                h, xedges, yedges, im = plt.hist2d(np.log10(mass_ahf), conc,
                                                   range=[[np.log10(3 * np.min(mass_ahf)), 14], [2.3, 14]], bins=80)
                plt.xlabel(r'log Mass [$M_\odot$/h]', size=15)
                plt.ylabel('c', size=20)
                plt.show()

                plt.figure()
                for k in range(1, 7):
                    plt.plot(yedges[1:], h[10 * k, :], label='log M = {:2.1f}'.format(xedges[10 * k]))
                plt.legend()
                plt.xlabel('c', size=20)
                plt.ylabel('N', size=20)
                plt.show()

                plt.figure()
                av = np.matmul(h, yedges[1:]) / np.sum(h, axis=1)
                plt.plot(xedges[1:], av)
                plt.xlabel(r'log Mass [$M_\odot$/h]', size=15)
                plt.ylabel('<C>', size=20)
                plt.show()



if __name__ == "__main__":
    ##############---------------- Getting the merger rates from sims -----####################

    sim_names = ['M25S07', 'M25S08', 'M25S09', 'M03S07','M03S08', 'M03S09', 'M35S07', 'M35S08', 'M35S09',
                 'Illustris', 'bolshoiP', 'bolshoiW', 'M03S08b', 'm25s85', 'm2s8', 'm4s7', 'm4s8', 'm2s9',
                 'm3s8_50', 'm3s8', 'm35s75', 'm4s9', 'm3s9', 'm25s75', 'm2s1', 'm3s7', 'm3s85', 'm2s7', 'm25s8',
                 'm35s8', 'm25s9', 'm35s85', 'm3s75', 'm35s9', 'm35s7']
    omegas = [0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.35, 0.35, 0.35, 0.309, 0.307, 0.27, 0.3, 0.25, 0.2, 0.4, 0.4, 0.2,  0.3
              ,0.3, 0.35, 0.4, 0.3, 0.25, 0.2, 0.3, 0.3, 0.2, 0.25, 0.35, 0.25, 0.35, 0.3, 0.35, 0.35]
    sigmas = [0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.816, 0.82, 0.82, 0.8, 0.85, 0.8, 0.7, 0.8, 0.9, 0.8
              ,0.8, 0.75, 0.9, 0.9, 0.75, 1.0, 0.7, 0.85, 0.7, 0.8, 0.8, 0.9, 0.85, 0.75, 0.9, 0.7]

    # sims = dict(zip(sim_names, list(zip(omegas, sigmas))))
    #
    # mrate_path = '/home/painchess/asus_fedora/merger_rate'
    # old_path = '/home/painchess/disq2/ahf-v1.0-101/'
    # localpath = '/home/painchess/sims/'
    # externalpath = '/home/painchess/mounted/HR_sims/niagara/'
    # illustris_path = '/home/painchess/mounted/TNG300-625/output'
    #
    # #
    # # s = -1
    # # sim38 = Simulation(sim_names[s], omegas[s], sigmas[s], localpath)
    # # reds = sim38.get_redshifts()
    #
    #
    # # snapshots = [3, 5, 8, 15]
    # # for snap in snapshots:
    # #     nmgs, tnds = sim38.dndxi(snap, 1e13)
    # #     red = reds[snap]
    # #     dz = reds[snap+1]-reds[snap]
    # #     ximin, ximax, resol = 1e-2, 1, 20
    # #     xis = np.logspace(np.log10(ximin), np.log10(ximax), resol+1)
    # #     dxis, ximeans = xis[1:]-xis[:-1], np.sqrt(xis[1:]*xis[:-1])
    # #
    # #     y = nmgs/dz/dxis/tnds[:-1]
    # #     plt.loglog(ximeans,  y, 'o', color='C{}'.format(snap))
    # #     plt.loglog(xis[1:-1], ell_mrate_per_n(5e13, red, xis, om0=omegas[s], sig8=sigmas[s]), color='C{}'.format(snap), linewidth=2)
    # # plt.show()
    #
    # # snaps = np.arange(0, 80, 40)
    # snaps = [0, 80]
    # # simselec = [-10, -5, -6, -12, -1]
    # simselec = ['m3s8', 'm2s7', 'm2s9', 'm4s9']
    # for s in simselec:
    #     sim38 = Simulation(s, sims[s][0], sims[s][1], localpath)
    #     reds = sim38.get_redshifts()
    #     # snaps = np.arange(0, 24, 8)
    #     # sim38.plot_tests('conc', snaps, vol=500)
    #     sim38.plot_tests('hmf', snaps, vol=500)
    # # snaps = np.arange(80)
    # # sim = 'm4s9'
    # # sim1 = Simulation(sim, sims[sim][0], sims[sim][1], localpath)
    # # for snap in snaps:
    # #      sim1.make_prog_desc(snap)
