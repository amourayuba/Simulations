import sys
sys.path.append('/home/painchess/projects_clean/Halo_Analytical_Calculations')
import merger_rate as mr
import cosmo_parameters as cp
from astropy.table import Table
from scipy.optimize import curve_fit
from astropy.cosmology import LambdaCDM, z_at_value
from astropy import units as u
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv


################ TOTAL NUMBER OF MERGERS ################################################
#############--------------- MxSy Simulations --------------#############################

def mcbride_mah(z, gamma, beta):
    """
    Fitting formula for a mass accretion history from McBride et al. (2009)
    :param z: float, redshift
    :param gamma: float, first fitting parameter
    :param beta: float, second fitting parameter
    :return: float, mass fraction at z
    """
    return (1 + z) ** beta * np.exp(-gamma * z)


def get_mratio(progenitor_masses, wpos=False, m=None, pos_s=None):
    """
    Getting the merger mass ratios given a set of progenitor masses
    :param progenitor_masses: array, progenitor masses
    :param wpos: bool, if use positions to determine the order of mergers
    :param m: int, index of considered halo mass
    :param pos_s: positions of progenitors
    :return:
    """
    if wpos:
        arg_mer = np.argmin(
            np.sum((pos_s[np.arange(len(pos_s)) != m, :] - pos_s[m, :]) ** 2, axis=1))
        if arg_mer >= m:
            arg_mer += 1
        return progenitor_masses[m] / progenitor_masses[arg_mer]
    else:
        pgratios = progenitor_masses / np.max(progenitor_masses)
        return pgratios[np.where(pgratios < 1)]


def large_growth_analytical(mass, zs, s8, om, nxibins=10000):
    """
    Calculates analytically the fraction of halos that experience a growth of more than 30% for a given
    Omega_m and sigma_m
    :param mass: float, or array. Mass of the Halo(s) considered.
    :param zs: array, redshifts over which to calculate the fraction that experienced large growth
    :param s8: float, sigma_8 of the universe considered
    :param om: float, Omega_M of the universe considered
    :param nxibins: int, number of numerical integration steps
    :return: float, or array if mass is array. Large growth fraction
    """
    res_integral = []
    dz = (zs[0] - zs[-1]) / len(zs)
    if type(mass) == np.ndarray or type(mass) == list:
        for M0 in mass:
            res_per_z = []
            for red in zs:
                res_per_z.append(mr.integ_mrate(M0, red, 1 / 3, 1, nxibins=nxibins, mass=False, sig8=s8, om0=om, ol0=1 - om))
            res_integral.append(np.sum(np.array(res_per_z)) * dz)
        return np.array(res_integral)
    else:
        res_per_z = []
        for red in zs:
            res_per_z.append(mr.integ_mrate(mass, red, 1 / 3, 1, nxibins=nxibins, mass=False, sig8=s8, om0=om, ol0=1 - om))
        return np.sum(np.array(res_per_z)) * dz


def average_growth_analytical(mass, zs, s8, om, nxibins=10000):
    """
    Calculates analytically the average growth of halos over redshifts zs for a given
    Omega_m and sigma_m
    :param mass: float, or array. Mass of the Halo(s) considered.
    :param zs: array, redshifts over which to calculate the average growth
    :param s8: float, sigma_8 of the universe considered
    :param om: float, Omega_M of the universe considered
    :param nxibins: int, number of numerical integration steps
    :return: float, or array if mass is array. Average growth
    """
    res_integral = []
    dz = (zs[0] - zs[-1]) / len(zs)
    if type(mass) == np.ndarray or type(mass) == list:
        for M0 in mass:
            res_per_z = []
            for red in zs:
                res_per_z.append(mr.integ_mrate(M0, red, 1e-14, 1, nxibins=nxibins, mass=True, sig8=s8, om0=om, ol0=1 - om) / M0)
            res_integral.append(np.sum(np.array(res_per_z)) * dz)
        return np.array(res_integral)
    else:
        res_per_z = []
        for red in zs:
            res_per_z.append(mr.integ_mrate(mass, red, 1e-14, 1, nxibins=nxibins, mass=True, sig8=s8, om0=om, ol0=1 - om) / mass)
        return np.sum(np.array(res_per_z)) * dz


def get_zlastdyn(zf, h, om, nzbins=20):
    """
    Gives the redshifts until one dynamical time before zf for a given hubble factor and Omega_m
    :param zf: float, final redshift of the considered object.
    :param h: float, hubble parameter h=H0/100
    :param om: float, Omega_m
    :param nzbins: number of redshift bins between last dynamical time and final redshift
    :return: zs: array of redshifts
            dz: float, redshift bin size
    """
    cosmol = LambdaCDM(H0=100 * h, Om0=om, Ode0=1 - om)
    infall = 1.44 / cp.hubble_ratio(zf, omega_m0=om, omega_l0=1 - om)  # infall time in gyr
    ages = cosmol.age(zf).value
    last_tdyn = ages - np.sqrt(2) * infall
    zi = z_at_value(cosmol.age, last_tdyn * u.Gyr).value
    zs = np.linspace(zi, zf, nzbins)
    dz = (zi - zf) / nzbins
    return zs, dz


class Simulation:
    """A class of objects referring to a Dark Matter only simulation
    The simulations that this class was written for have specific formats, with halos, merger trees and
     mass accretion histories found and measured through the Amiga Halo Finder, and are supposed to have specific files
     located in appropriate locations from 'simpath'. Unless you are using the specific simulations and
      files this simulation is meant for, I'd be cautious on how to use the methods below, and try
      to adapt the codes."""
    def __init__(self, name, om0, sig8, path):
        """
        :param name: str, name of the simulation. data files will be saved and reference this name
        :param om0: float, the value of Omega_m of the simulation.
        :param sig8: float, the value of s8 of the simulation.
        :param path: str, base path of the simulation, other data have to be in specific folder from the base path
        """
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
        """Gets the list of AHF prefixes of each file for all available snapshots"""
        with open(self.path + self.name + '/' + self.name + '_prefixes.txt') as file:
            prefs = file.read().splitlines()  # Names of each snapshot prefix.
        return prefs

    def read_halos(self, snapshot=0):
        """Read the AHF_halo file associated with this simulation and snapshot"""
        prefs = self.get_prefs()
        return pd.read_table(self.path + self.name + '/halos/' + prefs[snapshot] + '.AHF_halos', delim_whitespace=True,
                             header=0)

    def get_2dprop(self, prop, snapshot=0):
        """
        Gets the specific 2D property needed from the 2D property data finle
        :param prop: str, 2D property of concert
        :param snapshot: int, snapshot considered
        :return: array, value of the property for each halo in the snapshot
        """
        props = {'Conc': 0, r'$\chi^2_M$': 1, r'$\chi^2_\rho$': 2, 'axis_ratio': 3, 'axis_angle': -3, 'mbp_offset': -2,
                 'com_offset': -1}
        props2d = np.load(self.path + self.name + '/data/nprops2D_{}_snap{}.npy'.format(self.name, 118 - snapshot),
                          allow_pickle=True)
        prop_array = []
        for el in props2d:
            prop_array.append(el[props[prop]])
        return prop_array

    def get_3dprop(self, prop, snapshot=0):
        """
        Gets the specific 3D property needed from the 3D property data finle
        :param prop: str, 3D property of concert
        :param snapshot: int, snapshot considered
        :return: array, value of the property for each halo in the snapshot
        """
        props = {'Conc': 0, r'$\chi^2_M$': 1, r'$\chi^2_\rho$': 2}
        props2d = np.load(self.path + self.name + '/data/nprops3D_{}_snap{}.npy'.format(self.name, 118 - snapshot),
                          allow_pickle=True)
        prop_array = []
        for el in props2d:
            prop_array.append(el[props[prop]])
        return prop_array

    def make_structure_data(self, mmin, mmax, propage='z50', ztype='zx', snap=0, maxval=50, save=True):
        """
        Makes and saves two tables containing 2D and 3D structural properties with a given age property "propage"
        :param mmin: float, minimum halo mass limit
        :param mmax: float, maximum halo mass limit
        :param propage: str, specific age property to consider
        :param ztype: str, type of age property "zx", "zmm" or "mofz"
        :param snap: int, snapshot considered
        :param maxval: int, limit on the value of age property to not consider cases where the value hasn't been
        calculated properly
        :param save: bool, whether to save the data or not
        :return: data3D, data2D r: astropy table with the relevent 3D/2D sturctural propeties and the age property
        """
        reds = self.get_redshifts()
        agedata = self.get_agedata(z=reds[snap], atype=ztype) # The relevent age data for each halo
        mass_indx = self.get_agedata(z=reds[snap], atype='oth') # The indexes associated with each halo

        # Loading structural properties
        props2d = np.load(self.path + self.name + '/data/nprops2D_{}_snap{}.npy'.format(self.name, 118 - snap),
                          allow_pickle=True)
        props3d = np.load(self.path + self.name + '/data/nprops3D_{}_snap{}.npy'.format(self.name, 118 - snap),
                          allow_pickle=True)

        # These lines link the halo_id to an index going from 0 to Nhalos
        halids = np.array(self.read_halos(snapshot=snap)['#ID(1)'], dtype=int)
        idkeys = dict(zip(halids, np.arange(len(halids))))
        del halids

        # These lines remove the halos where the age property hasn't been measured correctly
        cl_data = agedata[agedata[propage] < maxval].reset_index(drop=True)
        clm_indx = mass_indx[agedata[propage] < maxval].reset_index(drop=True)
        del agedata, mass_indx

        comb_data2d, comb_data3d = [], []
        for i in range(len(cl_data)):
            if mmin < clm_indx.loc[i]['Mass'] < mmax: # Check if halo mass in mass range considered
                hid = clm_indx['Halo_index'].loc[i] # Get the halo id
                halidx = idkeys[hid] # Get its associated index to find the halo in the 2d/3d structural property tables
                # Save all properties for each considered halos to create a table
                comb_data3d.append([halidx, cl_data.loc[i][propage], props3d[halidx][0],
                                    props3d[halidx][1][0], props3d[halidx][1][1], props3d[halidx][2][0],
                                    props3d[halidx][2][1]])
                comb_data2d.append([halidx, cl_data.loc[i][propage], props2d[halidx][0],
                                    props2d[halidx][1][0], props2d[halidx][1][1], props2d[halidx][2][0],
                                    props2d[halidx][2][1], props2d[halidx][3], props2d[halidx][5], props2d[halidx][6]])
        # Create the two tables with the appropriate titles
        data3d = Table(np.array(comb_data3d),
                       names=['Halo_index', propage, 'Conc', r'$\chi^2_\rho$', 'log_Chi_rho', r'$\chi^2_M$',
                              'log_Chi_M'])
        data2d = Table(np.array(comb_data2d),
                       names=['Halo_index', propage, 'Conc', r'$\chi^2_\rho$', 'log_Chi_rho', r'$\chi^2_M$',
                              'log_Chi_M',
                              'axis_ratio', 'mbp_off', 'com_off'])

        if save:
            ascii.write(data3d,
                        self.path + self.name + '/data/{}_{}_dat3d_{:1.2e}.dat'.format(propage, self.name, mmin),
                        overwrite=True)
            ascii.write(data2d,
                        self.path + self.name + '/data/{}_{}_dat2d_{:1.2e}.dat'.format(propage, self.name, mmax),
                        overwrite=True)
            return data3d, data2d
        else:
            return data3d, data2d

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
        """
        Making mass accretion histories from AHF MAH files located in a folder mahs
        :param save: bool, whether to save them.
        :param fsizelim: int, minimum fileze limit in octets to avoid halos with no progenitors, or only one.
        :return: mahs: array arrays, mass accretion histories for each halo
                ids: array of int, halo id of each mahs
                emptyfiles: list of empty files that are not considered
        """
        mahs, ids, emptyfiles = [], [], []
        filenames = os.listdir(self.path + self.name + '/mahs')
        filenames.sort()
        for file in filenames:
            if os.path.getsize(
                    self.path + self.name + '/mahs/' + file) > fsizelim:  # check that file is has actual halos
                mahs.append(np.loadtxt(self.path + self.name + '/mahs/' + file)[:, 4])
                with open(self.path + self.name + '/mahs/' + file, 'r') as f:
                    for _ in range(3):
                        l = f.readline()
                    ids.append((int(l.split()[1])))
            else:
                emptyfiles.append(file)
        if save:
            np.save(self.path + self.name + '/data/mahs_{}.npy'.format(self.name), np.array(mahs, dtype=object))
            np.save(self.path + self.name + '/data/ids_{}.npy'.format(self.name), np.array(ids, dtype=int))
        return mahs, ids, emptyfiles

    def get_mah(self):
        """Gets the Mass Accretion History in the form of an .npy file with Nhalos elements, each one being a list of
        masses starting from z=0 """
        return np.load(self.path + '/{}/data/mahs_{}.npy'.format(self.name, self.name), allow_pickle=True)

    def get_mah_ids(self):
        """Gets the associated halo ids of the Mass Accretion History in the form of an .npy file with Nhalos
        elements, each one being a list of masses starting from z=0 """
        if os.path.exists(self.path + '/{}/data/ids_{}.npy'.format(self.name, self.name)):
            return np.load(self.path + '/{}/data/ids_{}.npy'.format(self.name, self.name))
        else:
            # if Ids file don't exist, assign them increasing ids
            mahs = self.get_mah()
            return np.arange(len(mahs))

    ###########################---------------CALCULATIONS WITH MAHS-------------------#################################

    def make_zxs(self, z, save=True, min_snap_dist=5, jump_fraction=(30, 25, 20, 10), snapnums=(11, 27, 59, 79, 89),
                 fractions=(90, 75, 50, 10, 1)):
        """
        Measures and saves various halo age properties from halos and mass accretio histories. Requires MAHs to be saved.
        :param z: float, Redshift of the halos
        :param save: bool, whether to save the age data after measuring them
        :param min_snap_dist: int, default=5.  minimum number of snapshots to where to consider calculating last major mass jump
        :param jump_fraction: tuple of int or floats. Mass jumps fractions to measure. 30 means Look at last
        time halo mass increased by at least 30%.
        :param snapnums: tuple of int. Snapshots where to store M/M_0 (z)
        :param fractions: tuple of int/float. fraction of the mass at z0 to at which to record the redshift.
        if fraction = 50, then record redshift at which the mass has been 50/100 of its z0 mass.
        :return:
        zxt: table containing z where mass was fractions of its z0 mass
        zmmt: table containing z of last mass jump of at least jump_fraction
        mofzt: table containing mass fraction at each of the given snapnums
        otht: table containing halo id of each halo and McBribe 2009 fitting parameters
        """
        # Loading the Mass Accretion Histories, and the Halo IDs of each.
        mahs = self.get_mah()
        idxs = self.get_mah_ids()
        reds = self.get_redshifts()

        nhalos = len(mahs)
        if nhalos != len(idxs):
            raise ValueError("Problem with MAHs and Ids not the same length")
        # Index of the first snapshot to consider
        snapi = np.min(np.where(reds >= z))

        # Snapshots to consider
        mofz_snaps = np.where(np.array(snapnums) > snapi)[0]

        # Data for the mofz table
        if len(mofz_snaps) > 0: # If there are any snapshots to consider saving
            mofz = True
            numi = np.min(mofz_snaps)
            rsnaps = snapnums[numi:]
            mofzs = np.zeros((nhalos, len(rsnaps)))
        else:
            mofz = False

        # Data for the oth table
        indices_i, gamma, beta, mass_i = [], [], [], []

        # Data for the zx table
        zx_i = dict(zip(fractions, [[] for _ in range(len(fractions))]))

        # Data for the zmm table
        zmm_i = dict(zip(jump_fraction, [[] for _ in range(len(jump_fraction))]))

        # Going through each halo and its Mass Accretion History
        for i in range(len(mahs)):
            mah = mahs[i][snapi:] # Mass accretion history to consider
            zeds = reds[snapi:len(mah) + snapi] # Redshifts of the considered MAH


            if len(mah) > min_snap_dist: # Only consider the halo if the MAH is longer than the minimum length chosen
                m = mah[0] # Mass at z0

                # Measurement of mofz
                if mofz: # If recording mofz
                    for k in range(len(rsnaps)):
                        if len(mah) > rsnaps[k]:
                            mofzs[i, k] = mah[rsnaps[k]] / m

                # Measurement of zmms
                immfractions = mah[:-min_snap_dist] / mah[min_snap_dist:] - 1 # Mass fractions along the halo history

                for p in range(len(jump_fraction)): # For each considered jump fraction
                    frc = jump_fraction[p] / 100
                    jump_idx = np.where(immfractions >= frc)[0] # Find the first halo index at which it made that jump

                    if len(jump_idx) > 0: # If that jump happened, store it
                        res = np.sqrt(zeds[np.min(jump_idx)] * zeds[np.min(jump_idx) + min_snap_dist])
                        zmm_i[jump_fraction[p]].append(res)
                    else: # If it didn't, store an arbitrary high value. Here equivalent z = 123, start of the simulation
                        zmm_i[jump_fraction[p]].append(123)

                # Measurement of zxs
                for j in range(len(fractions)): # For each considered mass fraction
                    fraction = fractions[j] / 100
                    ids = np.where(mah / m < fraction) # Find the index where the mass was below the considered fraction of z0 mass
                    if len(ids[0]) > 0: # If it exists, save the redshift
                        idx = np.min(np.where(mah / m < fraction))
                        zx = np.sqrt(zeds[idx] * zeds[idx - 1])
                        zx_i[fractions[j]].append(zx)
                    else: # If it doesn't save infinity as value.
                        zx_i[fractions[j]].append(np.inf)
                # Measurement of McBribe 2009 fitting parameters
                try: # If fitting succeeds, save them
                    param, cov = curve_fit(mcbride_mah, zeds, mah / m)
                    gamma.append(param[0])
                    beta.append(param[1])
                except RuntimeError: # If it doesn't, save an arbitrarily small value
                    gamma.append(-1e50)
                    beta.append(-1e50)

                mass_i.append(m) # Save the halo mass at z0
                indices_i.append(idxs[i]) # Save the halo Id

        mass_i, indices_i, gamma, beta = np.array(mass_i), np.array(indices_i, dtype=int), np.array(gamma), np.array(
            beta)

        nameszx = ['z{}'.format(fraction) for fraction in fractions]
        nameszmm = ['zmm{}'.format(jfraction) for jfraction in jump_fraction]
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
            zxt.to_csv(self.path + '/{}/data/zxt_{}_z{}.dat'.format(self.name, self.name, z))
            zmmt.to_csv(self.path + '/{}/data/zmmt_{}_z{}.dat'.format(self.name, self.name, z))
            mofzt.to_csv(self.path + '/{}/data/mofzt_{}_z{}.dat'.format(self.name, self.name, z))
            otht.to_csv(self.path + '/{}/data/otht_{}_z{}.dat'.format(self.name, self.name, z))
        return zxt, zmmt, mofzt, otht

    def get_agedata(self, z, atype='oth'):
        """
        Load specific age property from file located in the right directory.
        :param z: float, redshift
        :param atype: str, type of age data, either zx, zmm, mofz or oth
        :return: pandas dataframe, the specific age data for all halos
        """
        return pd.read_csv(self.path + '/{}/data/{}t_{}_z{}.dat'.format(self.name, atype, self.name, z))

    def average_growth(self, zf, mmin=1e12, mmax=1e14, mbins=20, save=False, h=0.7, subsample_fraction=1):
        """Calculates the average growth over the last dynamical timescale of halos between mmin and mmax at zf
        :param zf: float, redshift at final state of halo.
        :param mmin: float, lower bound of the considered mass range
        :param mmax: float, higher bound of the considered mass range
        :param mbins: int, number of mass bins between mmin and mmax
        :param save: bool, whether to save the data.
        :param h, float, hubble parameter
        :param subsample_fraction: float between 0 and 1.
        :return:
        res_sim: array, Total growth for each mass bin
        ntot_sim: array, Number of halos in each mass bin
        ps_sim: array, poissonian errors in each mass bin
        """
        mahs = self.get_mah() # Loads the MAH. MAH file need to be there.
        zs, dz = get_zlastdyn(zf, h=h, om=self.om0, nzbins=20) # Finds z at last dynamical time
        zi = np.max(zs)
        reds = self.get_redshifts()
        masses = np.logspace(np.log10(mmin), np.log10(mmax), mbins)
        res_sim, ntot_sim = np.zeros(mbins), np.zeros(mbins)

        # Go over each halo
        for j in range(len(mahs)):
            if np.random.random() < subsample_fraction: # If want a random subsample of halos
                mah = mahs[j]
                zeds = reds[:len(mah)]
                if zeds[-1] > zi: # Check if halo's MAH goes at least until last dynamical time
                    zbin_min, zbin_max = np.min(np.where(zeds > zf)), np.min(
                        np.where(zeds > zi))  # find the snapshot bin corresponding to zf
                    if mah[zbin_min] < masses[-1]: # If the mass in the mass range considered
                        mbin = np.min(np.where(masses > mah[zbin_min])) # Find the mass bin
                        if mah[zbin_min] > mah[zbin_max] and mah[zbin_min] / mah[zbin_max] < 5: # Avoid big sudden jumps, and decreasing masses
                            res_sim[mbin] += mah[zbin_min] / mah[zbin_max] - 1  # Save the growth fraction
                            ntot_sim[mbin] += 1 # Add to the number of halos in the bin
        ps_sim = np.sqrt(res_sim * (1 + res_sim / ntot_sim))
        if save:
            np.savetxt(self.path + '/{}/mah_mergers/av_growth_sim_{:2.1f}_sim_{}_nbins{}'
                       .format(self.name, zf, self.name, mbins), np.array(res_sim))
            np.savetxt(self.path + '/{}/mah_mergers/av_growth_ntot_sim_{:2.1f}_sim_{}_nbins{}'
                       .format(self.name, zf, self.name, mbins), np.array(ntot_sim))
            np.savetxt(self.path + '/{}/mah_mergers/av_growth_ps_sim_{:2.1f}_sim_{}_nbins{}'
                       .format(self.name, zf, self.name, mbins), np.array(ps_sim))
        return res_sim, ntot_sim, ps_sim

    def large_growth(self, zf, mmin=1e12,  mmax=1e14, mbins=20, h=0.7, save=False, subsample_fraction=1):
        """Calculates the fraction of halos at zf that had a major merger during the last dynamical timescale
        :param zf: float, redshift at final state of halo.
        :param mmin: float, lower bound of the considered mass range
        :param mmax: float, higher bound of the considered mass range
        :param mbins: int, number of mass bins between mmin and mmax
        :param save: bool, whether to save the data.
        :param subsample_fraction: float between 0 and 1.
        :return:
        res_sim: array, Number of halos that had a large fraction for each mass bin
        ntot_sim: array, Number of halos in each mass bin
        ps_sim: array, poissonian errors in each mass bin
        """
        mahs = self.get_mah() # Loads the MAH. MAH file need to be there.
        zs, dz = get_zlastdyn(zf, h=h, om=self.om0, nzbins=20) # Finds z at last dynamical time
        zi = np.max(zs)
        reds = self.get_redshifts()
        masses = np.logspace(np.log10(mmin), np.log10(mmax), mbins)
        res_sim, ntot_sim = np.zeros(mbins), np.zeros(mbins)

        # Go over each halo
        for j in range(len(mahs)):
            if np.random.random() < subsample_fraction: # If want a random subsample of halos
                mah = mahs[j]
                zeds = reds[:len(mah)]
                if zeds[-1] > zi: # Check if halo's MAH goes at least until last dynamical time
                    zbin_min, zbin_max = np.min(np.where(zeds > zf)), np.min(
                        np.where(zeds > zi))  # find the snapshot bin corresponding to zf and zi
                    if mah[zbin_min] < masses[-1]: # If the mass in the mass range considered
                        mbin = np.min(np.where(masses > mah[zbin_min])) # Find the mass bin
                        ntot_sim[mbin] += 1 # Add to the total of the mass bin
                        if 1.333 < mah[zbin_min] / mah[zbin_max] < 2: # If there has been large growth, add to the count
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
        """
        Loads saved large growth fraction files saved through large_growth() method
        :param mbins: int, number of mass bins
        :param zf: float, redshift.
        :return:
        res_sim: array, Number of halos that had a large fraction for each mass bin
        ntot_sim: array, Number of halos in each mass bin
        ps_sim: array, poissonian errors in each mass bin
        """
        res_sim = np.loadtxt(self.path + '/{}/mah_mergers/large_growth_sim_{:2.1f}_sim_{}_nbins{}'
                             .format(self.name, zf, self.name, mbins))
        ntot_sim = np.loadtxt(self.path + '/{}/mah_mergers/large_growth_ntot_sim_{:2.1f}_sim_{}_nbins{}'
                              .format(self.name, zf, self.name, mbins))
        ps_sim = np.loadtxt(self.path + '/{}/mah_mergers/large_growth_ps_sim_{:2.1f}_sim_{}_nbins{}'
                            .format(self.name, zf, self.name, mbins))
        return res_sim, ntot_sim, ps_sim

    def load_av_growth(self, mbins, zf):
        """
        Loads saved average growth files saved through large_growth() method
        :param mbins: int, number of mass bins
        :param zf: float, redshift.
        :return:
        res_sim: array, Total growth for each mass bin
        ntot_sim: array, Number of halos in each mass bin
        ps_sim: array, poissonian errors in each mass bin
        """
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
                    pass
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
        """
        Measures the merger rate per halo per dz per merger ratio xi
        :param snap: int, snapshot number
        :param mlim: float, minimum mass range
        :param bins: int, number of merger mass ratio bins
        :param ximin: float, minimum merger ratio
        :param ximax: float, maximum merger ratio
        :param wpos: bool, default False (DO NOT CHANGE. FEATURE NOT READY). Whether to use progenitor positions to
        find sequence of mergers
        :return: nmgs: array, number of mergers
                tnds: array, number of halos in each bin. Corrected for selection effect.
        """
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
                tnds += xis > mmin / (mass - mmin)
                if len(mprg[i]) > 1:
                    progenitor_masses = np.array(mprg[i])
                    if wpos:
                        for m in range(len(progenitor_masses)):
                            pos_s = np.array(prog_pos[i])
                            rat = get_mratio(progenitor_masses, wpos, m, pos_s)
                            if rat < 1:
                                for j in range(bins):
                                    if (rat > xis[j]) and (rat < xis[j + 1]):
                                        nmgs[j] += 1
                                        tnds += np.max(progenitor_masses) * xis > mmin
                    else:
                        rat = get_mratio(progenitor_masses, wpos)
                        for xs in rat:
                            for j in range(bins):
                                if (xs > xis[j]) and (xs < xis[j + 1]):
                                    nmgs[j] += 1
        return nmgs, tnds

    def N_of_xi(self, snaps, mlim, mlim_rat=1e3, ximin=0.01, resol=100, wpos=False):
        """
        Measures the merger rate per halo per dz integrated between ximin and 1
        :param snaps: array, snapshots to consider
        :param mlim: float, minimum mass range
        :param mlim_rat: float, mlim_rat*mlim is the maximum mass range
        :param ximin: float, minimum merger mass ratio
        :param resol: int, number of integration steps
        :param wpos: bool, default False (DO NOT CHANGE. FEATURE NOT READY). Whether to use progenitor positions to
        find sequence of mergers
        :return: mres: array, cummulative number of mergers
                tnds: array, number of halos in each bin. Corrected for selection effect.
        """
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
                    xilim = 1 / (mds[i] / mmin - 1)
                    # print(xilim)
                    tnds[:, k] += xilim < dexis
                    if len(mprg[i]) > 1:
                        progenitor_masses = mprg[i]
                        if wpos:
                            pos_s = np.array(prog_pos[i])
                            for m in range(len(progenitor_masses)):
                                rat = get_mratio(progenitor_masses, wpos, m, pos_s)
                                if rat < 1:
                                    mres[:, k] += rat > dexis
                                    tnds[:, k] += xilim < dexis
                        else:
                            pgratios = progenitor_masses / np.max(progenitor_masses)
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

    sim_names = ['M25S07', 'M25S08', 'M25S09', 'M03S07', 'M03S08', 'M03S09', 'M35S07', 'M35S08', 'M35S09',
                 'Illustris', 'bolshoiP', 'bolshoiW', 'M03S08b', 'm25s85', 'm2s8', 'm4s7', 'm4s8', 'm2s9',
                 'm3s8_50', 'm3s8', 'm35s75', 'm4s9', 'm3s9', 'm25s75', 'm2s1', 'm3s7', 'm3s85', 'm2s7', 'm25s8',
                 'm35s8', 'm25s9', 'm35s85', 'm3s75', 'm35s9', 'm35s7']
    omegas = [0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.35, 0.35, 0.35, 0.309, 0.307, 0.27, 0.3, 0.25, 0.2, 0.4, 0.4, 0.2, 0.3
        , 0.3, 0.35, 0.4, 0.3, 0.25, 0.2, 0.3, 0.3, 0.2, 0.25, 0.35, 0.25, 0.35, 0.3, 0.35, 0.35]
    sigmas = [0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.816, 0.82, 0.82, 0.8, 0.85, 0.8, 0.7, 0.8, 0.9, 0.8
        , 0.8, 0.75, 0.9, 0.9, 0.75, 1.0, 0.7, 0.85, 0.7, 0.8, 0.8, 0.9, 0.85, 0.75, 0.9, 0.7]

    sims = dict(zip(sim_names, list(zip(omegas, sigmas))))
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
