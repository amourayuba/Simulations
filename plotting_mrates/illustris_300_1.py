import base
import csv
import os
import numpy as np
import h5py
from autograd import grad
from scipy.special import gamma
import groupcat
import lhalotree
import sublink
import util
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from scipy.optimize import curve_fit
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from astropy.cosmology import LambdaCDM, z_at_value
from astropy import units as u

###################---------- COSMOLOGY-------###############################
### Fundamental set of parameters
G = 4.30091e-9  # Units Mpc/Msun x (km/s)^2
rho_c = 3 * 100 ** 2 / (8 * np.pi * G)  # h^2xMsun/Mpc**3

### New set of cosmo parameters used here using
# Eiseinstein and Hu conventions and Planck 2018 values

h = 0.6766
c = 3e5  # speed of light km/s
ombh2 = 0.02242  # Density of baryons in units of h2
omb = ombh2 / h ** 2  # Density of baryons
omch2 = 0.122  # Density of CDM in units h2
omc = omch2 / h ** 2  # Density of CDM
omnuh2 = 0.0006451439  # Density of neutrinos in units h²
omnu = omnuh2 / h ** 2  # density of neutrinos
omh2 = ombh2 + omch2 + omnuh2  # Density of matter in units h²
om = omb + omc  # Density of matter
omr = 1e-4  # Upper limit estimation of radiation density

sigma8 = 0.8  # fluctuation rms normalisation at 8Mpc
ns = 0.9626  # spectral index for initial power spectrum
oml = 0.685  # density of Lambda
om0 = oml + omb + omc + omr  # Total density including lambda

Tcmb = 2.7255  # Temperature of CMB
theta_cmb = Tcmb / 2.7  # Normalized CMB temperature
Nnu = 1  # Number of massive neutrino species
mnu = 91.5 * omnuh2 / Nnu  # Mass of neutrino part
zeq = 3387  # z at matter-radiation equality
zdrag = 1060.01  # z at compton drag
yd = (1 + zeq) / (1 + zdrag)
s = 147.21  # Sound horizon at recombination in Mpc

basepath = '/home/ayuba/scratch/Illustris/output'
a_reds = 1 / np.linspace(1 / 21.05, 1, 100) - 1


def get_mdesc_mprogs(Nhalos, GroupFirstSub, snap, ptype=1):
    fields = ['SubhaloCM', 'SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 'FirstProgenitorID',
              'SubhaloMassType', 'SubhaloMassType']
    mprogs = []
    cm_progs = []
    mdesc = []
    numMergers = []
    for j in range(Nhalos):
        if np.random.uniform() < 0.1:
            tree = sublink.loadTree(basepath, 99, GroupFirstSub[j],
                                    fields=fields)  # Load the tree corresponding to the subhalo groupfirstsub[j]
            if len(tree['SubhaloID']) > snap + 2:  # make sure the tree is long enough
                numMergers.append(sublink.numMergers(tree, minMassRatio=0.3, massPartType='dm', index=snap))
                rootID = tree['SubhaloID'][snap]  # The ID of the main branch subhalo of the tree at snapshot snap
                fpID = tree['FirstProgenitorID'][snap]  # The ID of the main branch progenitor
                if fpID != -1:  # condition that the subhalo has at least one progenitor
                    prg = []  # this will store the masses of all progenitors of this subhalo
                    prg_cm = []
                    mdesc.append(tree['SubhaloMassType'][
                                     snap, ptype] * 1e10)  # The mass of the main branch subhalo of the tree at snapshot snap
                    fpIndex = snap + (fpID - rootID)  # The index on the tree of the  main branch progenitor
                    prg.append(sublink.maxPastMass(tree, fpIndex, ptype) * 1e10)
                    prg_cm.append(tree['SubhaloCM'][fpIndex])
                    # prg.append(tree['SubhaloMassType'][fpIndex, ptype]*1e10)

                    npID = tree['NextProgenitorID'][fpIndex]  # exploring siblings of the FIRST PROGENITOR
                    while npID != -1:  # while there is a smaller sibling
                        npIndex = snap + (npID - rootID)
                        prg.append(sublink.maxPastMass(tree, npIndex, ptype) * 1e10)
                        prg_cm.append(tree['SubhaloCM'][npIndex])
                        # prg.append(tree['SubhaloMassType'][npIndex, ptype]*1e10) #storing the mass
                        npID = tree['NextProgenitorID'][npIndex]  # looking at the smaller mass sibling
                    mprogs.append(prg)  # we are done with the the progenitors of the main branch subhalo
                    cm_progs.append(prg_cm)

                    dnpID = tree['NextProgenitorID'][snap]  # exploring siblings of the main subhalo
                    while dnpID != -1:
                        dnpIndex = snap + (dnpID - rootID)
                        fpID = tree['FirstProgenitorID'][dnpIndex]  # The ID of the main branch progenitor
                        if fpID != -1:  # condition that the subhalo has at least one progenitor
                            prg = []  # this will store the masses of all progenitors of this subhalo
                            prg_cm = []
                            mdesc.append(tree['SubhaloMassType'][
                                             dnpIndex, ptype] * 1e10)  # The mass of the main branch subhalo of the tree at snapshot snap
                            fpIndex = snap + (fpID - rootID)  # The index on the tree of the  main branch progenitor
                            prg.append(sublink.maxPastMass(tree, fpIndex, ptype) * 1e10)
                            prg_cm.append(tree['SubhaloCM'][fpIndex])
                            # prg.append(tree['SubhaloMassType'][fpIndex, ptype]*1e10)
                            npID = tree['NextProgenitorID'][fpIndex]  # exploring siblings of the FIRST PROGENITOR
                            while npID != -1:  # while there is a smaller sibling
                                npIndex = snap + (npID - rootID)
                                # prg.append(tree['SubhaloMassType'][npIndex, ptype]*1e10) #storing the mass
                                prg.append(sublink.maxPastMass(tree, npIndex, ptype) * 1e10)
                                prg_cm.append(tree['SubhaloCM'][npIndex])
                                npID = tree['NextProgenitorID'][npIndex]  # looking at the smaller mass sibling
                            mprogs.append(prg)  # we are done with the the progenitors of this sibling
                            cm_progs.append(prg_cm)
                        dnpID = tree['NextProgenitorID'][dnpIndex]

    return mdesc, mprogs, cm_progs, numMergers


####################---------------------------- TEST 1 --------------------------##########################################################################

mb = []
snapshots = [50]
# colows = ['blue','red','green','orange']
# snapshots = [27, 50, 66, 74]
colows = ['blue', 'magenta', 'black', 'red']
totmerg = 0
bins = 40
xis = np.logspace(-4, 0, bins + 1)
cosmo = LambdaCDM(H0=100 * h, Om0=.3, Ode0=.7, Ob0=4e-2)
ages = cosmo.age(a_reds).value
# xis = np.linspace(0.01, 1, bins+1)
dxis = xis[1:] - xis[:-1]
ximeans = np.sqrt(xis[1:] * xis[:-1])
p0 = [0.01, 0.1, -1, 0.1]
bounds = ([1e-5, 0.01, -3, 0.001], [1, 1, 0, 1])
GroupFirstSub = groupcat.loadHalos(basepath, 99, fields=['GroupFirstSub'])
snap = 3
Nhalos = len(GroupFirstSub)
save = True
mlims = [5e11, 2e12, 1e13]
usepos = False
for l in range(len(mlims)):
    mlim = mlims[l]
    for k in range(len(snapshots)):
        snapshot = snapshots[k]
        mbins = []
        mratios = []

        nds = 0
        tnds = np.zeros(bins + 1)
        print(snapshot)
        if save:
            if os.path.exists(basepath + 'cm_progs_mp{}.npy'.format(snapshot)):
                mdesc = np.load(basepath + 'mdesc_mp{}.npy'.format(snapshot), allow_pickle=True)
                mprogs = np.load(basepath + 'mprogs_mp{}.npy'.format(snapshot), allow_pickle=True)
                cm_progs = np.load(basepath + 'cm_progs_mp{}.npy'.format(snapshot), allow_pickle=True)
            else:
                mdesc, mprogs, cm_progs, nmergers = get_mdesc_mprogs(Nhalos, GroupFirstSub, snapshot)
                np.save(basepath + 'mdesc_mp{}.npy'.format(snapshot), np.array(mdesc))
                np.save(basepath + 'mprogs_mp{}.npy'.format(snapshot), np.array(mprogs, dtype=object))
                np.save(basepath + 'cm_progs_mp{}.npy'.format(snapshot), np.array(cm_progs, dtype=object))
                print('we saved')
        else:
            mdesc = np.load(basepath + 'mdesc_mp{}.npy'.format(snapshot), allow_pickle=True)
            mprogs = np.load(basepath + 'mprogs_mp{}.npy'.format(snapshot), allow_pickle=True)
            cm_progs = np.load(basepath + 'cm_progs_mp{}.npy'.format(snapshot), allow_pickle=True)
        mmin = np.min(mdesc)
        nmgs = np.zeros(bins)
        for i in range(len(mdesc)):
            if (mdesc[i] > mlim) and (mdesc[i] < 10 * mlim):
                if len(mprogs[i]) > 0:
                    pgmasses = mprogs[i]
                    nds += 1
                    tnds += np.max(pgmasses) * xis > mmin
                if len(mprogs[i]) > 1:
                    if usepos:
                        cmprg = np.array(cm_progs[i])
                        for m in range(len(pgmasses)):
                            arg_mer = np.argmin(
                                np.sum((cmprg[np.arange(len(pgmasses)) != m, :] - cmprg[m, :]) ** 2, axis=1))
                            # rat = min(pgmasses[m]/pgmasses[arg_mer], pgmasses[arg_mer]/pgmasses[m])
                            rat = pgmasses[m] / pgmasses[arg_mer]
                            if rat < 1:
                                for j in range(bins):
                                    if (rat > xis[j]) and (rat < xis[j + 1]):
                                        nmgs[j] += 1
                    else:
                        pgratios = pgmasses / np.max(pgmasses)
                        for rat in pgratios:
                            if rat < 1:
                                for j in range(bins):
                                    if (rat > xis[j]) and (rat < xis[j + 1]):
                                        nmgs[j] += 1
        dz = a_reds[snapshot + 1] - a_reds[snapshot]
        ax = plt.gca()
        dt = ages[snapshot] - ages[snapshot + 1]
        y = nmgs / dz / tnds[1:] / dxis
        poisson = np.sqrt(nmgs) / dz / tnds[1:] / dxis
        nximeans, ny, npoisson = ximeans[y != 0], y[y != 0], poisson[y != 0]


        def tofit(x, A, xi_mean, beta, gamma):
            return np.log(FM_mrate(x, 5 * mlim, a_reds[snapshot], A, xi_mean, beta, gamma, 0.133, 0.1))


        # ximeans, y = np.loadtxt('m3s8_m5e13_z{:1.1f}.txt'.format(a_reds[snapshot]))

        par, _ = curve_fit(tofit, nximeans, np.log10(ny), p0=p0, bounds=bounds)

        # mrate = nmergers[1:]/np.histogram(desc_mass, bins=masses)[0]/dt
        plt.scatter(nximeans, np.log10(ny), color=colows[l], s=8,
                    label='z={:1.1f} M={:1.2}'.format(a_reds[snapshot], mlim))
        plt.fill_between(nximeans, np.log10(ny - npoisson), np.log10(ny + npoisson), color=colows[l], alpha=0.2)
        plt.plot(nximeans, np.log(FM_mrate(nximeans, 5 * mlim, a_reds[snapshot], par[0], par[1], par[2], par[3])), '-.',
                 color=colows[l], linewidth=2)

    # plt.plot(nximeans, np.log(FM_mrate(nximeans, 5*mlim, a_reds[snapshot], 0.0104, 0.00972, -1.995, 0.263)),'-.', color = colows[k], linewidth=2)
plt.title('Illustris ')

plt.xscale('log')

plt.xlabel(r'$\xi =M_1/M_2$', size=15)
plt.ylabel(r'log dN/dz/d$\xi$ [mergers/halo/dz]')
plt.legend()
plt.savefig('N_of_xi_corrected_ill_vs_m.pdf', dpi=650, bbox_inches='tight', facecolor='white', transparent=False)

#######################--------------------------------TEST 2 -------------------------------------##############################################

snapshots = [27, 50, 66, 74]
colows = ['blue', 'magenta', 'black', 'red']
totmerg = 0
bins = 40
xis = np.logspace(-4, 0, bins + 1)
cosmo = LambdaCDM(H0=100 * h, Om0=.3, Ode0=.7, Ob0=4e-2)
ages = cosmo.age(a_reds).value
p0 = [0.01, 0.1, -1, 0.1]
bounds = ([1e-5, 0.01, -3, 0.001], [1, 1, 0, 1])
dxis = xis[1:] - xis[:-1]
ximeans = np.sqrt(xis[1:] * xis[:-1])
GroupFirstSub = groupcat.loadHalos(basepath, 99, fields=['GroupFirstSub'])

Nhalos = len(GroupFirstSub)
save = True
mlim = 1e13
for k in range(len(snapshots)):
    snapshot = snapshots[k]
    mbins = []
    mratios = []

    nds = 0
    tnds = np.zeros(bins + 1)
    if save:
        mdesc, mprogs, cm_progs, nmergers = get_mdesc_mprogs(Nhalos, GroupFirstSub, snapshot)
        np.save(basepath + 'mdesc_mp{}.npy'.format(snapshot), np.array(mdesc))
        np.save(basepath + 'mprogs_mp{}.npy'.format(snapshot), np.array(mprogs, dtype=object))
        np.save(basepath + 'cm_progs_mp{}.npy'.format(snapshot), np.array(cm_progs, dtype=object))
    else:
        mdesc = np.load(basepath + 'mdesc_mp{}.npy'.format(snapshot), allow_pickle=True)
        mprogs = np.load(basepath + 'mprogs_mp{}.npy'.format(snapshot), allow_pickle=True)
        cm_progs = np.load(basepath + 'cm_progs_mp{}.npy'.format(snapshot), allow_pickle=True)
    mmin = np.min(mdesc)
    nmgs = np.zeros(bins)
    for i in range(len(mdesc)):
        if (mdesc[i] > mlim) and (mdesc[i] < 10 * mlim):
            if len(mprogs[i]) > 0:
                pgmasses = mprogs[i]
                nds += 1
                tnds += np.max(pgmasses) * xis > mmin
            if len(mprogs[i]) > 1:
                if usepos:
                    cmprg = np.array(cm_progs[i])
                    for m in range(len(pgmasses)):
                        arg_mer = np.argmin(
                            np.sum((cmprg[np.arange(len(pgmasses)) != m, :] - cmprg[m, :]) ** 2, axis=1))
                        rat = min(pgmasses[m] / pgmasses[arg_mer], pgmasses[arg_mer] / pgmasses[m])
                        for j in range(bins):
                            if (rat > xis[j]) and (rat < xis[j + 1]):
                                nmgs[j] += 1
                else:
                    pgratios = pgmasses / np.max(pgmasses)
                    for rat in pgratios:
                        if rat < 1:
                            for j in range(bins):
                                if (rat > xis[j]) and (rat < xis[j + 1]):
                                    nmgs[j] += 1

    dz = a_reds[snapshot + 1] - a_reds[snapshot]
    ax = plt.gca()
    dt = ages[snapshot] - ages[snapshot + 1]
    y = nmgs / dz / tnds[1:] / dxis
    idx = np.nonzero(y)
    nximeans, ny = ximeans[y != 0], y[y != 0]


    def tofit(x, A, xi_mean, beta, gamma):
        return np.log(FM_mrate(x, 5 * mlim, a_reds[snapshot], A, xi_mean, beta, gamma, 0.133, 0.1))


    # ximeans, y = np.loadtxt('m3s8_m5e13_z{:1.1f}.txt'.format(a_reds[snapshot]))

    par, _ = curve_fit(tofit, nximeans, np.log(ny), p0=p0, bounds=bounds)

    plt.scatter(nximeans, np.log(ny), color=colows[k], s=8, label='z={:1.1f}'.format(a_reds[snapshot]))
    if k != 0:
        plt.plot(nximeans, np.log(FM_mrate(nximeans, 5 * mlim, a_reds[snapshot], 0.0104, 0.00972, -1.995, 0.263)), '-.',
                 color=colows[k], linewidth=2)
    else:
        plt.plot(nximeans, np.log(FM_mrate(nximeans, 5 * mlim, a_reds[snapshot], 0.0104, 0.00972, -1.995, 0.263)),
                 label='FM10', color=colows[k], linewidth=2)

plt.xscale('log')

plt.xlabel(r'$\xi =M_1/M_2$', size=15)
plt.ylabel(r'dN/dz/d$\xi$ [mergers/halo/dz]')
plt.legend()
plt.savefig('N_of_xi_corrected_ill_dz.pdf', dpi=650, bbox_inches='tight', facecolor='white', transparent=False)

############## --------------------------------- TEST 3 ------------------#########################################################


snapshots = [27, 50, 66, 74]
colows = ['blue', 'magenta', 'black', 'red']
totmerg = 0
bins = 20
xis = np.logspace(-2.2, 0, bins + 1)
cosmo = LambdaCDM(H0=100 * h, Om0=.3, Ode0=.7, Ob0=4e-2)
ages = cosmo.age(a_reds).value
p0 = [0.01, 0.1, -1, 0.1]
bounds = ([1e-5, 0.01, -3, 0.001], [1, 1, 0, 1])
dxis = xis[1:] - xis[:-1]
ximeans = np.sqrt(xis[1:] * xis[:-1])
GroupFirstSub = groupcat.loadHalos(basepath, 99, fields=['GroupFirstSub'])
snap = 3
Nhalos = len(GroupFirstSub)
save = False
mlim = 1e12
for k in range(len(snapshots)):
    snapshot = snapshots[k]
    mbins = []
    mratios = []

    nds = 0
    tnds = np.zeros(bins + 1)
    if save:
        mdesc, mprogs, cm_progs, nmergers = get_mdesc_mprogs(Nhalos, GroupFirstSub, snapshot)
        np.save(basepath + 'mdesc_mp{}.npy'.format(snapshot), np.array(mdesc))
        np.save(basepath + 'mprogs_mp{}.npy'.format(snapshot), np.array(mprogs, dtype=object))
        np.save(basepath + 'cm_progs_mp{}.npy'.format(snapshot), np.array(cm_progs, dtype=object))
    else:
        mdesc = np.load(basepath + 'mdesc_mp{}.npy'.format(snapshot), allow_pickle=True)
        mprogs = np.load(basepath + 'mprogs_mp{}.npy'.format(snapshot), allow_pickle=True)
        cm_progs = np.load(basepath + 'cm_progs_mp{}.npy'.format(snapshot), allow_pickle=True)
    mmin = np.min(mdesc)
    nmgs = np.zeros(bins)
    for i in range(len(mdesc)):
        if (mdesc[i] > mlim) and (mdesc[i] < 1000 * mlim):
            if len(mprogs[i]) > 0:
                pgmasses = mprogs[i]
                nds += 1
                tnds += np.max(pgmasses) * xis > mmin
            if len(mprogs[i]) > 1:
                if usepose:
                    cmprg = np.array(cm_progs[i])
                    for m in range(len(pgmasses)):
                        arg_mer = np.argmin(
                            np.sum((cmprg[np.arange(len(pgmasses)) != m, :] - cmprg[m, :]) ** 2, axis=1))
                        rat = min(pgmasses[m] / pgmasses[arg_mer], pgmasses[arg_mer] / pgmasses[m])
                        for j in range(bins):
                            if (rat > xis[j]) and (rat < xis[j + 1]):
                                nmgs[j] += 1
                else:
                    pgratios = pgmasses / np.max(pgmasses)
                    for rat in pgratios:
                        if rat < 1:
                            for j in range(bins):
                                if (rat > xis[j]) and (rat < xis[j + 1]):
                                    nmgs[j] += 1

    dz = a_reds[snapshot + 1] - a_reds[snapshot]
    ax = plt.gca()
    da = 1 / (1 + a_reds[snapshot]) - 1 / (a_reds[snapshot + 1] + 1)
    dt = ages[snapshot] - ages[snapshot + 1]
    y = nmgs / dt / tnds[1:] / dxis
    poisson = np.sqrt(nmgs) / dt / tnds[1:] / dxis
    idx = np.nonzero(y)
    nximeans, ny, npoisson = ximeans[y != 0], y[y != 0], poisson[y != 0]


    def tofit(x, A, xi_mean, beta, gamma):
        return np.log(FM_mrate(x, 10 * mlim, a_reds[snapshot], A, xi_mean, beta, gamma, 0.133, 0.1))


    # ximeans, y = np.loadtxt('m3s8_m5e13_z{:1.1f}.txt'.format(a_reds[snapshot]))

    par, _ = curve_fit(tofit, nximeans, np.log(ny), p0=p0, bounds=bounds)

    plt.scatter(nximeans, np.log10(ny), color=colows[k], s=8, label='z={:1.1f}'.format(a_reds[snapshot]))

plt.xscale('log')
plt.title('Illustris all Mass')
plt.xlabel(r'$\xi =M_1/M_2$', size=15)
plt.ylabel(r'dN/dz/d$\xi$ [mergers/halo/da]')
plt.legend()
plt.savefig('N_of_xi_ill_all_mass_dt.pdf', dpi=650, bbox_inches='tight', facecolor='white', transparent=False)

################# -------------------------- TEST 4 -------------------------########################################################################

import matplotlib.pyplot as plt

params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
from matplotlib import ticker

params = {'legend.fontsize': 8, 'legend.handlelength': 1}
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "serif"

cosmo = LambdaCDM(H0=100 * h, Om0=.3, Ode0=.7, Ob0=4e-2)
infall = 1.44 / hubble_ratio(a_reds, omega_m0=0.3, omega_l0=0.7)  # infall time in gyr
ages = cosmo.age(a_reds).value

mb = []
snapshots = np.arange(2, 62, 4)
# snapshots = [12]
colows = ['blue', 'red', 'green', 'orange']
totmerg = 0
bins = 15
# ximins = np.array([0.01, 0.03, 0.1, 0.3])
# ximins = np.array([0.01, 0.03, 0.1, 0.3])
save = False
mlim = 1e12
ys, poisson = [], []
resol = 100
ximins = np.array([0, 250, 500, 750]) // 10
mres = np.zeros((resol, len(snapshots)))
nds = np.zeros(len(snapshots))
tnds = np.zeros((resol, len(snapshots)))
dndxi = np.zeros((len(snapshots), resol))
dexis = np.logspace(-2, 0, resol)

markers = ['s', 'v', 'o', 'x']
s8 = 0.82
om = 0.31
GroupFirstSub = groupcat.loadHalos(basepath, 99, fields=['GroupFirstSub'])
Nhalos = len(GroupFirstSub)

for k in range(len(snapshots)):
    snapshot = snapshots[k]
    mbins = []
    mratios = []
    if save:
        if os.path.exists(basepath + '/cm_progs_mp{}.npy'.format(snapshot)):
            mdesc = np.load(basepath + '/mdesc_mp{}.npy'.format(snapshot), allow_pickle=True)
            mprogs = np.load(basepath + '/mprogs_mp{}.npy'.format(snapshot), allow_pickle=True)
            cm_progs = np.load(basepath + '/cm_progs_mp{}.npy'.format(snapshot), allow_pickle=True)
        else:
            mdesc, mprogs, cm_progs, nmergers = get_mdesc_mprogs(Nhalos, GroupFirstSub, snapshot)
            np.save(basepath + '/mdesc_mp{}.npy'.format(snapshot), np.array(mdesc))
            np.save(basepath + '/mprogs_mp{}.npy'.format(snapshot), np.array(mprogs, dtype=object))
            np.save(basepath + '/cm_progs_mp{}.npy'.format(snapshot), np.array(cm_progs, dtype=object))
    else:
        mdesc = np.load(basepath + '/mdesc_mp{}.npy'.format(snapshot), allow_pickle=True)
        mprogs = np.load(basepath + '/mprogs_mp{}.npy'.format(snapshot), allow_pickle=True)
        cm_progs = np.load(basepath + '/cm_progs_mp{}.npy'.format(snapshot), allow_pickle=True)
    mmin = np.min(mdesc)
    for i in range(len(mdesc)):
        if (mdesc[i] > mlim) and (mdesc[i] < 1000 * mlim):
            if len(mprogs[i]) > 0:
                nds[k] += 1
                pgmasses = mprogs[i]
                xilim = mmin / np.max(pgmasses)
                tnds[:, k] += xilim < dexis
            if len(mprogs[i]) > 1:
                if usepos:
                    cmprg = np.array(cm_progs[i])
                    for m in range(len(pgmasses)):
                        arg_mer = np.argmin(
                            np.sum((cmprg[np.arange(len(pgmasses)) != m, :] - cmprg[m, :]) ** 2, axis=1))
                        rat = min(pgmasses[m] / pgmasses[arg_mer], pgmasses[arg_mer] / pgmasses[m])
                        if rat < 1:
                            mres[:, k] += rat > dexis
                        # print(rat>dexis)
                else:
                    pgratios = pgmasses / np.max(pgmasses)
                    for rat in pgratios:
                        if rat < 1:
                            mres[:, k] += rat > dexis
                mbins.append(mdesc[i])

# dt = ages[snapshot]-ages[snapshot+1]
dzs = np.array([a_reds[snapshots + 1] - a_reds[snapshots]] * (resol - 1))
dmres = (mres[:-1, :] - mres[1:, :]) / tnds[1:, :]
ps = np.sqrt(mres[:-1, :] - mres[1:, :]) / tnds[1:, :]
nmres = np.cumsum(dmres[::-1], axis=0)[::-1, :]
nps = np.cumsum(ps[::-1], axis=0)[::-1, :]

ys.append(nmres / dzs)
poisson.append(nps / dzs)

lss = ['-', '--', '-.']
fig, axs = plt.subplots(2, 2, figsize=[14, 14], dpi=500)

for n in range(len(ximins)):
    ax = axs[n // 2, n % 2]
    i = ximins[n]
    for j in range(len(ys)):
        y = ys[j]
        ps = poisson[j]

        res = []
        for k in range(len(snapshots)):
            res.append(integ_mrate(mlim, a_reds[snapshots[k]], xi_min=dexis[i], xi_max=1, om0=om, sig8=s8))
        if j == 0 and n == 0:
            ax.plot(1 + a_reds[snapshots], res, color=colows[j], ls=lss[j], label=r'EC $\Omega_m$={:1.2f}'.format(om))
            ax.scatter(1 + a_reds[snapshots], y[i], color=colows[j], marker=markers[i],
                       label=r'$\xi>$ {:1.2f}, $\Omega_m$={:1.2f}'.format(dexis[i], om))
        elif j == 0:
            ax.scatter(1 + a_reds[snapshots], y[i], color=colows[j], marker=markers[j],
                       label=r'$\xi>$ {:1.2f}'.format(dexis[i]))
            ax.plot(1 + a_reds[snapshots], res, color=colows[j], ls=lss[j])
        elif n == 0:
            ax.plot(1 + a_reds[snapshots], res, color=colows[j], ls=lss[j], label=r'EC $\Omega_m$={:1.2f}'.format(om))
            ax.scatter(1 + a_reds[snapshots], y[i], color=colows[j], marker=markers[j],
                       label=r'$\Omega_m$={:1.2f}'.format(om))
        else:
            ax.scatter(1 + a_reds[snapshots], y[i], color=colows[j], marker=markers[j])
            ax.plot(1 + a_reds[snapshots], res, color=colows[j], ls=lss[j])

        ax.fill_between(1 + a_reds[snapshots], y[i] - ps[i], y[i] + ps[i], color=colows[j], alpha=0.2)

    ax.set_xlabel('1+z', size=15)
    ax.set_ylabel(r'dN(>$\xi$/dz [mergers/halo/dz]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
plt.savefig('Cumulative_N_xi.png', dpi=650, bbox_inches='tight', facecolor='white', transparent=False)
