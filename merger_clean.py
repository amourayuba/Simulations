from merger_rate import *
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

class Simulation:
    '''A class of objects referring to a Dark Matter only simulation'''
    def __init__(self, name, om0, sig8, path):
        self.name = name #name of the simulation
        self.om0 = om0 #Value of Omega_m for the simulation
        self.sig8 = sig8 #Value of sigma_8
        self.path = path #The base (absolute) path of that simulation. All simulation paths have to be structured the same.
    def get_redshifts(self):
        '''Gets the list of redshifts of each snapshot'''
        if self.name[0] == 'M':
            return np.loadtxt(self.path + '/redshifts/M03S08.txt')
        elif os.path.exists(self.path + '/redshifts/{}.txt'.format(self.name)):
            return np.loadtxt(self.path + '/redshifts/{}.txt'.format(self.name))
        else:
            return np.loadtxt(self.path + '/redshifts/mxsy_reds.txt')[::-1]

    def get_desc_prog(self, snap, wpos=False):
        ''' Gets the descendant and progenitor masses if they are already saved'''
        mds = np.load(self.path + '/progs_desc/{}_desc_mass_snap{}.npy'.format(self.name, snap), allow_pickle=True)
        mprg = np.load(self.path + '/progs_desc/{}_prog_mass_snap{}.npy'.format(self.name, snap), allow_pickle=True)
        if wpos:
            prog_pos = np.load(self.path + '/progs_desc/{}_prog_pos_snap{}.npy'.format(self.name, snap),
                               allow_pickle=True) #Gets the prog positions if needed.
            return mds, mprg, prog_pos
        return mds, mprg

    def make_prog_desc(self, snapshot, usepos=False, save=True):
        '''Makes a list of descendant masses and each corresponding prog mass'''
        f_halos = self.path + self.name +'/halos/'   #Typically where the halos are stored
        f_mtrees = self.path + self.name + '/mtrees/' #Typically where the merger trees are stored
        with open(self.path + self.name + '/'+self.name+'_prefixes.txt') as file:
            prefs = file.read().splitlines() #Names of each snapshot prefix.

        d_halos = pd.read_table(f_halos + prefs[snapshot] + '.AHF_halos', delim_whitespace=True, header=0)
        p_halos = pd.read_table(f_halos + prefs[snapshot + 1] + '.AHF_halos', delim_whitespace=True, header=0)

        desc_mass = np.array(d_halos['Mhalo(4)']) #halo masses at snap
        prog_mass = np.array(p_halos['Mhalo(4)']) #halo masses at snap+1

        # desc_mass = np.loadtxt(f_halos + prefs[snapshot] + '.AHF_halos')[:, 3]
        # prog_mass = np.loadtxt(f_halos + prefs[snapshot + 1] + '.AHF_halos')[:, 3]
        id4_desc = np.loadtxt(f_halos + prefs[snapshot] + '.AHF_halos')[4, 0] #random halo id at snap
        id4_prog = np.loadtxt(f_halos + prefs[snapshot + 1] + '.AHF_halos')[4, 0] #random halo id at snap+1

        if id4_desc > len(desc_mass): #test whether ids are from 0 tp len(desc_mass) or not
            dic_desc = dict(zip(np.array(d_halos['#ID(1)']).astype(int), desc_mass)) #If ids are not ranked, need to use a dictionnary for quick access

        if id4_prog > len(prog_mass):
            dic_prog = dict(zip(np.array(p_halos['#ID(1)']).astype(int), prog_mass))

        if usepos:
            Xcs, Ycs, Zcs = np.loadtxt(f_halos + prefs[snapshot + 1] + '.AHF_halos')[:, 5], np.loadtxt(
                f_halos + prefs[snapshot + 1] + '_halos')[:, 6], np.loadtxt(f_halos + prefs[snapshot + 1] + '_halos')[:,
                                                                 7]
        mmin = np.min(desc_mass)
        desc, progs = [], []
        with open(f_mtrees + prefs[snapshot][:-7] + '_mtree') as file:
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
            np.save(self.path + '/progs_desc/{}_desc_mass_snap{}.npy'.format(self.name, snapshot), np.array(mds, dtype=object))
            np.save(self.path + '/progs_desc/{}_prog_mass_snap{}.npy'.format(self.name, snapshot), np.array(mprg, dtype=object))
            if usepos:
                np.save(self.path + '/progs_desc/{}_prog_pos_snap{}.npy'.format(self.name, snapshot), np.array(pos_prg, dtype=object))
        if usepos:
            return np.array(mds, dtype=object), np.array(mprg, dtype=object), np.array(pos_prg, dtype=object)
        return np.array(mds, dtype=object), np.array(mprg, dtype=object)

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


    def plot_tests(self, test, snaps, vol=500, h=0.7, conctype=1):
        f_halos = self.path + self.name +'/halos/'
        with open(self.path + self.name + '/'+self.name+'_prefixes.txt') as file:
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
                hist_dens = hist[0] / vol**3
                my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': self.om0, 'Ode0':1-self.om0, 'Ob0': 0.0482, 'sigma8': self.sig8,
                            'ns': 0.965}
                cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
                #truebins = np.sqrt(bins[:-1] * bins[1:])
                truebins = bins[1:]
                mfunc = mass_function.massFunction(truebins, zn, mdef='200c', model='tinker08', q_out='dndlnM')

                plt.plot(truebins, hist_dens / unit[0], '-.', color='C{}'.format(j),
                         label='1024'[:3*(j==0)] +' {} snapshot={}'.format(self.name, snapshot))
                plt.plot(truebins, mfunc, color='C{}'.format(j), label='Colossus'[:8*(j==0)])
            plt.legend()
            plt.show()

        if test == 'conc':
            for j in range(len(snaps)):
                snapshot = snaps[j]
                halos = pd.read_table(f_halos + prefs[snapshot] + '.AHF_halos', delim_whitespace=True, header=0)
                if conctype ==1:
                    conc = np.array(halos['cNFW(43)'])
                if conctype == 2:
                    Rvir = np.array(halos['Rhalo(12)'])
                    Vvir = np.sqrt(G * mass_ahf / (0.001 * Rvir))
                    Vmax = np.array(halos['Vmax(17)'])
                    conc = Vmax / Vvir

                mass_ahf = np.array(halos['Mhalo(4)'])
                plt.figure()
                h, xedges, yedges, im = plt.hist2d(np.log10(mass_ahf), conc, range=[[np.log10(3*np.min(mass_ahf)), 14], [2.3, 14]], bins=80)
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

##############---------------- Getting the merger rates from sims -----####################


sim_names = ['M25S07', 'M25S08', 'M25S09', 'M03S07','M03S08', 'M03S09', 'M35S07', 'M35S08', 'M35S09',
             'Illustris', 'bolshoiP', 'bolshoiW', 'm25s85', 'm2s8', 'm4s7', 'm4s8', 'm2s9', 'M03S08b',
             'm3s8_50', 'm3s8', 'm35s75', 'm4s9', 'm3s9']
omegas = [0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.35, 0.35, 0.35, 0.309, 0.307, 0.27, 0.25, 0.2, 0.4, 0.4, 0.2, 0.3, 0.3
          ,0.3, 0.35, 0.4, 0.3]
sigmas = [0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.816, 0.82, 0.82, 0.85, 0.8, 0.7, 0.8, 0.9, 0.8, 0.8
          ,0.8, 0.75, 0.9, 0.9]
mrate_path = '/home/painchess/asus_fedora/merger_rate'
old_path = '/home/painchess/disq2/ahf-v1.0-101/'
localpath = '/home/painchess/sims/'
externalpath = '/home/painchess/mounted/HR_sims/niagara/'
illustris_path = '/home/painchess/mounted/TNG300-625/output'


#
# s = -1
# sim38 = Simulation(sim_names[s], omegas[s], sigmas[s], old_path)
# reds = sim38.get_redshifts()

#snaps = np.arange(21, 41, 4)

# snapshots = [3, 5, 8, 15]
# for snap in snapshots:
#     nmgs, tnds = sim38.dndxi(snap, 1e13)
#     red = reds[snap]
#     dz = reds[snap+1]-reds[snap]
#     ximin, ximax, resol = 1e-2, 1, 20
#     xis = np.logspace(np.log10(ximin), np.log10(ximax), resol+1)
#     dxis, ximeans = xis[1:]-xis[:-1], np.sqrt(xis[1:]*xis[:-1])
#
#     y = nmgs/dz/dxis/tnds[:-1]
#     plt.loglog(ximeans,  y, 'o', color='C{}'.format(snap))
#     plt.loglog(xis[1:-1], ell_mrate_per_n(5e13, red, xis, om0=omegas[s], sig8=sigmas[s]), color='C{}'.format(snap), linewidth=2)
# plt.show()

s = -3
sim38 = Simulation(sim_names[s], omegas[s], sigmas[s], localpath)
reds = sim38.get_redshifts()
snaps = np.arange(1, 20, 4)
sim38.plot_tests('conc', snaps, vol=500)

# for snap in snaps:
#     sim28b.make_prog_desc(snap)



