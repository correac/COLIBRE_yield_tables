import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.integrate import quad, romberg
from scipy.integrate import cumtrapz, romb, simpson
from scipy.interpolate import interp1d
from sympy import Interval
from scipy.interpolate import RegularGridInterpolator
import warnings
import matplotlib.ticker as tck

warnings.filterwarnings("ignore")


mass_fraction = np.array([0.73738788833, #H
                          0.24924186942, #He
                          0.0023647215,  #C
                          0.0006928991,  #N
                          0.00573271036, #O
                          0.00125649278, #Ne
                          0.00070797838, #Mg
                          0.00066495154, #Si
                          0.00129199252]) #Fe

def imf_log(m):
    # Xi(log10m) (log)

    if m <= 1:
        log_m = np.log10(m)
        dm = log_m - np.log10(0.079)
        sigma = 0.69
        A = 0.852464
        xi = A * np.exp(-0.5 * dm**2 / sigma**2)

    else:
        A = 0.237912
        x = -1.3
        xi = A * m ** x

    return xi

def imf_lin(m):
    # Xi(m) (linear, not log!)

    if m <= 1:
        log_m = np.log10(m)
        dm = log_m - np.log10(0.079)
        sigma = 0.69
        A = 0.852464
        xi = A * np.exp(-0.5 * dm**2 / sigma**2) / m

    else:
        A = 0.237912
        x = -2.3
        xi = A * m ** x

    xi /= np.log(10)
    return xi

def integrate_IMF(m_min, m_max):

    # mass_range = np.arange(np.log10(m_min), np.log10(m_max), 0.2)
    # imf_array = np.zeros(len(mass_range))
    # for i in range(0,len(mass_range)):
    #     imf_array[i] = imf_log(10**mass_range[i])
    #
    # y_int = simpson(imf_array, dx=0.2)
    # print('simpson',y_int)
    #
    # I = quad(imf_lin, m_min, m_max)
    # print('quad',I[0])

    Masses = np.arange(m_min, m_max)
    imf_array = np.zeros(len(Masses))
    for i in range(0, len(Masses)):
        imf_array[i] = imf_lin(Masses[i])

    IMF_int = simpson(imf_array * Masses, x=Masses)
    return IMF_int

def lifetimes(m,Z):

    # Write data to HDF5
    with h5py.File('./data/EAGLE_yieldtables/Lifetimes.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        lifetimes = data_file["Lifetimes"][:][:]
        Metallicities = data_file["Metallicities"][:]

    num_metals = len(Metallicities)
    MZ = np.zeros(num_metals)
    for i in range(num_metals):
        f = interp1d(Masses, lifetimes[i,:])
        MZ[i] = f(m)

    f = interp1d(Metallicities, MZ)
    result = f(Z) / 1e9 # Gyr
    return result

def inverse_lifetime(Z):
    Hubble_time = 13.8 # Gyr
    mass_range = np.arange(0.7,100,0.01)
    mass_limit = []
    for i in range(len(mass_range)):
        t = lifetimes(mass_range[i],Z)
        if t <= Hubble_time:
            mass_limit = np.append(mass_limit,mass_range[i])

    return mass_limit

def read_SNII_COLIBRE(Z):

    total_mass_fraction = 0.0133714

    file = '../data/SNII_linear_extrapolation.hdf5'
    data_file = h5py.File(file, 'r')

    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])

    metallicity_range = data_file['Metallicities'][:]
    metallicity_flag = [k.decode() for k in data_file['Yield_names'][:]]

    Masses = data_file["Masses"][:]

    num_elements = len(mass_fraction)
    num_mass_bins = len(Masses)
    stellar_yields_ccsn = np.zeros((num_elements, num_mass_bins))
    stellar_yields_winds = np.zeros((num_elements, num_mass_bins))
    stellar_yields_ccsn_1 = np.zeros((num_elements, num_mass_bins))
    stellar_yields_ccsn_2 = np.zeros((num_elements, num_mass_bins))
    stellar_yields_winds_1 = np.zeros((num_elements, num_mass_bins))
    stellar_yields_winds_2 = np.zeros((num_elements, num_mass_bins))

    stellar_yields_total = np.zeros((num_elements, num_mass_bins))
    stellar_yields_total_1 = np.zeros((num_elements, num_mass_bins))
    stellar_yields_total_2 = np.zeros((num_elements, num_mass_bins))

    indx_nearest = np.abs(metallicity_range - Z).argmin()

    if indx_nearest == len(metallicity_range) - 1:

        Ejected_mass_winds = data_file["/Yields/"+metallicity_flag[indx_nearest]+"/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/"+metallicity_flag[indx_nearest]+"/Ejected_mass_in_ccsn"][:][:]

        factor = metallicity_range[indx_nearest] / total_mass_fraction

        for i in range(num_mass_bins):
            stellar_yields_ccsn[:, i] = Ejected_mass_ccsn[indx, i]
            stellar_yields_winds[:, i] = factor * mass_fraction * Ejected_mass_winds[i]
            stellar_yields_total[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]

        # stellar_yields = Ejected_mass_ccsn[indx, :] + factor * mass_fraction[indx] * Ejected_mass_winds
        # stellar_yields_ccsn *= (1. + (Z - metallicity_range[indx_nearest]) / metallicity_range[indx_nearest])
        # stellar_yields_winds *= (1. + (Z - metallicity_range[indx_nearest]) / metallicity_range[indx_nearest])
        # stellar_yields_total *= (1. + (Z - metallicity_range[indx_nearest]) / metallicity_range[indx_nearest])

    else:
        if metallicity_range[indx_nearest] > Z:
            indx_1 = indx_nearest - 1
            indx_2 = indx_nearest
        else:
            indx_1 = indx_nearest
            indx_2 = indx_nearest + 1

        Ejected_mass_winds = data_file["/Yields/"+metallicity_flag[indx_1]+"/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/"+metallicity_flag[indx_1]+"/Ejected_mass_in_ccsn"][:][:]
        factor = metallicity_range[indx_1] / total_mass_fraction
        # stellar_yields_1 = Ejected_mass_ccsn[indx, :] + factor * mass_fraction[indx] * Ejected_mass_winds

        for i in range(num_mass_bins):
            stellar_yields_ccsn_1[:, i] = Ejected_mass_ccsn[indx, i]
            stellar_yields_winds_1[:, i] = factor * mass_fraction * Ejected_mass_winds[i]
            stellar_yields_total_1[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]

        Ejected_mass_winds = data_file["/Yields/"+metallicity_flag[indx_2]+"/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/"+metallicity_flag[indx_2]+"/Ejected_mass_in_ccsn"][:][:]
        factor = metallicity_range[indx_2] / total_mass_fraction
        # stellar_yields_2 = Ejected_mass_ccsn[indx, :] + factor * mass_fraction[indx] * Ejected_mass_winds

        for i in range(num_mass_bins):
            stellar_yields_ccsn_2[:, i] = Ejected_mass_ccsn[indx, i]
            stellar_yields_winds_2[:, i] = factor * mass_fraction * Ejected_mass_winds[i]
            stellar_yields_total_2[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]

        # b = (stellar_yields_ccsn_2 - stellar_yields_ccsn_1) / (metallicity_range[indx_2] - metallicity_range[indx_1])
        # a = stellar_yields_ccsn_1 - b * metallicity_range[indx_1]
        #
        # stellar_yields_ccsn = a + b * Z
        #
        # b = (stellar_yields_winds_2 - stellar_yields_winds_1) / (metallicity_range[indx_2] - metallicity_range[indx_1])
        # a = stellar_yields_winds_1 - b * metallicity_range[indx_1]
        #
        # stellar_yields_winds = a + b * Z
        #
        # b = (stellar_yields_total_2 - stellar_yields_total_1) / (metallicity_range[indx_2] - metallicity_range[indx_1])
        # a = stellar_yields_total_1 - b * metallicity_range[indx_1]
        #
        # stellar_yields_total = a + b * Z
        dz = metallicity_range[indx_2] - metallicity_range[indx_1]
        stellar_yields_total = stellar_yields_total_1 + dz * (stellar_yields_total_2 - stellar_yields_total_1)


    # total_mass_fraction = 0.0133714
    #
    # #['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron'
    # indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
    #
    # factor = metallicity / total_mass_fraction
    #
    # # Write data to HDF5
    # with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
    #     Masses = data_file["Masses"][:]
    #     Ejected_mass_winds = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_winds"][:]
    #     Ejected_mass_ccsn = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_ccsn"][:][:]
    #     Total_metals = data_file["/Yields/Z_"+metallicity_flag+"/Total_Mass_ejected"][:]
    #
    # select = np.where(Masses >= 8)[0]
    # Masses = Masses[select]
    # num_mass_bins = len(Masses)
    # num_elements = len(mass_fraction)
    # stellar_yields_ccsn = np.zeros((num_elements, num_mass_bins))
    # stellar_yields_winds = np.zeros((num_elements, num_mass_bins))
    #
    # for i in range(num_mass_bins):
    #     stellar_yields_ccsn[:, i] = Ejected_mass_ccsn[indx, i]
    #     stellar_yields_winds[:, i] = factor * mass_fraction * Ejected_mass_winds[i]

    new_mass_range = np.arange(8, 40.1, 0.1)
    new_imf_range = np.zeros(len(new_mass_range))
    new_stellar_yields_ccsn_range = np.zeros(len(new_mass_range))
    new_stellar_yields_winds_range = np.zeros(len(new_mass_range))
    new_stellar_yields_total_range = np.zeros(len(new_mass_range))
    colibre_ccsn = np.zeros(9)
    colibre_winds = np.zeros(9)
    colibre_total = np.zeros(9)
    for i in range(0,9):
        f = interp1d(Masses, stellar_yields_ccsn[i, :])
        g = interp1d(Masses, stellar_yields_winds[i, :])
        h = interp1d(Masses, stellar_yields_total[i, :])
        for j in range(len(new_mass_range)):
            new_imf_range[j] = imf_lin(new_mass_range[j])
            if new_mass_range[j] < np.min(Masses):
                new_stellar_yields_ccsn_range[j] = stellar_yields_ccsn[i, 0]
                new_stellar_yields_winds_range[j] = stellar_yields_winds[i, 0]
                new_stellar_yields_total_range[j] = stellar_yields_total[i, 0]
            elif new_mass_range[j] > np.max(Masses):
                new_stellar_yields_ccsn_range[j] = stellar_yields_ccsn[i, -1]
                new_stellar_yields_winds_range[j] = stellar_yields_winds[i, -1]
                new_stellar_yields_total_range[j] = stellar_yields_total[i, -1]
            else:
                new_stellar_yields_ccsn_range[j] = f(new_mass_range[j])
                new_stellar_yields_winds_range[j] = g(new_mass_range[j])
                new_stellar_yields_total_range[j] = h(new_mass_range[j])

        colibre_ccsn[i] = simpson(new_imf_range * new_stellar_yields_ccsn_range, x=new_mass_range)
        colibre_winds[i] = simpson(new_imf_range * new_stellar_yields_winds_range, x=new_mass_range)
        colibre_total[i] = simpson(new_imf_range * new_stellar_yields_total_range, x=new_mass_range)

    return colibre_ccsn, colibre_winds, colibre_total


def read_SNII_EAGLE(Z,min_range,max_range):

    total_mass_fraction = 0.0133714

    file = '../data/EAGLE_yieldtables/SNII.hdf5'
    data_file = h5py.File(file, 'r')

    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])

    metallicity_range = data_file['Metallicities'][:]
    metallicity_flag = [k.decode() for k in data_file['Yield_names'][:]]

    Masses = data_file["Masses"][:]

    num_elements = len(mass_fraction)
    num_mass_bins = len(Masses)
    stellar_yields = np.zeros((num_elements, num_mass_bins))
    stellar_yields_1 = np.zeros((num_elements, num_mass_bins))
    stellar_yields_2 = np.zeros((num_elements, num_mass_bins))

    indx_nearest = np.abs(metallicity_range - Z).argmin()

    if indx_nearest == 0 or indx_nearest == len(metallicity_range) - 1:

        Yield = data_file["/Yields/" + metallicity_flag[indx_nearest] + "/Yield"][:][:]
        Ejected_mass = data_file["/Yields/" + metallicity_flag[indx_nearest] + "/Ejected_mass"][:]

        factor = metallicity_range[indx_nearest] / total_mass_fraction

        for i in range(num_mass_bins):
            stellar_yields[:, i] = Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]

        # stellar_yields = Yield[indx, :] + factor * mass_fraction[indx] * Ejected_mass
        # stellar_yields *= (1. + (Z - metallicity_range[indx_nearest]) / metallicity_range[indx_nearest])

    else:
        if metallicity_range[indx_nearest] > Z:
            indx_1 = indx_nearest - 1
            indx_2 = indx_nearest
        else:
            indx_1 = indx_nearest
            indx_2 = indx_nearest + 1

        Yield = data_file["/Yields/" + metallicity_flag[indx_1] + "/Yield"][:][:]
        Ejected_mass = data_file["/Yields/" + metallicity_flag[indx_1] + "/Ejected_mass"][:]
        factor = metallicity_range[indx_1] / total_mass_fraction

        for i in range(num_mass_bins):
            stellar_yields_1[:, i] = Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]

        # stellar_yields_1 = Yield[indx, :] + factor * mass_fraction * Ejected_mass

        Yield = data_file["/Yields/" + metallicity_flag[indx_2] + "/Yield"][:][:]
        Ejected_mass = data_file["/Yields/" + metallicity_flag[indx_2] + "/Ejected_mass"][:]
        factor = metallicity_range[indx_2] / total_mass_fraction

        for i in range(num_mass_bins):
            stellar_yields_2[:, i] = Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]

        # stellar_yields_2 = Yield[indx, :] + factor * mass_fraction * Ejected_mass

        dz = metallicity_range[indx_2] - metallicity_range[indx_1]

        # b = (stellar_yields_2 - stellar_yields_1) / (metallicity_range[indx_2] - metallicity_range[indx_1])
        # a = stellar_yields_1 - b * metallicity_range[indx_1]
        # stellar_yields = a + b * Z
        stellar_yields = stellar_yields_1 + dz * (stellar_yields_2 - stellar_yields_1)

    for i in range(num_mass_bins):
        for j in range(num_elements):
            if stellar_yields[j,i] < 0: stellar_yields[j,i]=0

    new_mass_range = np.arange(min_range, max_range, 0.1)
    new_imf_range = np.zeros(len(new_mass_range))
    new_stellar_yields_range = np.zeros(len(new_mass_range))
    eagle = np.zeros(9)
    for i in range(0,9):
        f = interp1d(Masses, stellar_yields[i, :])
        for j in range(len(new_mass_range)):
            new_imf_range[j] = imf_lin(new_mass_range[j])
            if new_mass_range[j] < np.min(Masses):
                new_stellar_yields_range[j] = stellar_yields[i, 0]
            elif new_mass_range[j] > np.max(Masses):
                new_stellar_yields_range[j] = stellar_yields[i, -1]
            else:
                new_stellar_yields_range[j] = f(new_mass_range[j])

        eagle[i] = simpson(new_imf_range * new_stellar_yields_range, x=new_mass_range)

    return eagle


if __name__ == "__main__":


    metallicity = np.arange(-5,0.2,0.2)
    metallicity = 10**metallicity
    num_metals = len(metallicity)
    CCSN_COLIBRE = np.zeros(num_metals)
    CCSN_EAGLE = np.zeros(num_metals)
    CCSN_EAGLE2 = np.zeros(num_metals)

    CCSN_COLIBRE_HHe = np.zeros(num_metals)
    CCSN_EAGLE_HHe = np.zeros(num_metals)
    CCSN_EAGLE2_HHe = np.zeros(num_metals)
    CCSN_COLIBRE_All = np.zeros((num_metals,9))
    CCSN_EAGLE_All = np.zeros((num_metals,9))
    CCSN_EAGLE2_All = np.zeros((num_metals,9))

    IMF_int = integrate_IMF(0.1, 100)

    for i in range(num_metals):
        _, _, CCSN_mej_total = read_SNII_COLIBRE(metallicity[i])
        CCSN_COLIBRE[i] = np.sum(CCSN_mej_total[2:]) / IMF_int
        CCSN_COLIBRE_HHe[i] = np.sum(CCSN_mej_total[0:2]) / IMF_int
        CCSN_COLIBRE_All[i,:] = CCSN_mej_total / IMF_int

        CCSN_Eagle_2 = read_SNII_EAGLE(metallicity[i],8,40)
        CCSN_EAGLE[i] = np.sum(CCSN_Eagle_2[2:]) / IMF_int
        CCSN_EAGLE_HHe[i] = np.sum(CCSN_Eagle_2[0:2]) / IMF_int
        CCSN_EAGLE_All[i,:] = CCSN_Eagle_2 / IMF_int

        # CCSN_Eagle_2 = read_SNII_EAGLE(metallicity[i],6,100)
        # CCSN_EAGLE2[i] = np.sum(CCSN_Eagle_2[2:]) / IMF_int
        # CCSN_EAGLE2_All[i] = np.sum(CCSN_Eagle_2[0:2]) / IMF_int


    ########################
    # Plot parameters
    params = {
        "font.size": 11,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (4, 2.8),
        "figure.subplot.left": 0.18,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.93,
        "figure.subplot.wspace": 0.25,
        "figure.subplot.hspace": 0.25,
        "lines.markersize": 3,
        "lines.linewidth": 1,
        "figure.max_open_warning": 0,
        "axes.axisbelow": True,
    }
    rcParams.update(params)

    ########################
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-', linewidth=0.3)
    # plt.grid(which='minor', linestyle=':', linewidth=0.3)

    plt.plot(metallicity,np.ones(len(metallicity)),'--',color='black')

    plt.plot(metallicity, CCSN_COLIBRE/CCSN_EAGLE, 'o-', color='tab:blue', label='Total Returned Metal Mass Fraction')
    # plt.plot(metallicity, CCSN_COLIBRE/CCSN_EAGLE2, '--', color='tab:blue')
    plt.plot(metallicity, CCSN_COLIBRE_HHe/CCSN_EAGLE_HHe, 'o-', color='tab:orange', label='Total Returned H+He Mass Fraction')
    # plt.plot(metallicity, CCSN_COLIBRE_All/CCSN_EAGLE2_All, '--', color='tab:orange')
    # plt.plot(metallicity, CCSN_EAGLE, 'o-', color='tab:orange', label='EAGLE')
    # plt.plot(metallicity, CCSN_EAGLE2, '--', color='tab:orange')

    plt.axis([1e-5, 1, 0, 13])
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('COLIBRE/EAGLE')
    plt.xlabel('Metallicity')
    plt.legend(labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=1, fontsize=10)

    # locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    # ax.yaxis.set_major_locator(locmaj)
    # locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    # ax.yaxis.set_minor_locator(locmin)
    # ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('Returned_mass_as_function_Z.png', dpi=300)

    ########################
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-', linewidth=0.3)

    elem = ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Iron']
    for i in range(9):
        plt.plot(metallicity, CCSN_COLIBRE_All[:,i], 'o-', label=elem[i])

    plt.axis([1e-5, 1, 1e-5, 10])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Total Returned Elements Mass Fraction')
    plt.xlabel('Metallicity')
    plt.legend(labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=3, fontsize=10)

    props = dict(boxstyle='round', fc='white', ec='white', alpha=0.2)
    ax.text(0.7, 0.1, 'COLIBRE', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('Returned_element_mass_as_function_Z_COLIBRE.png', dpi=300)

    ########################
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-', linewidth=0.3)

    elem = ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Iron']
    for i in range(9):
        plt.plot(metallicity, CCSN_EAGLE_All[:, i], 'o-', label=elem[i])

    plt.axis([1e-5, 1, 1e-5, 10])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Total Returned Elements Mass Fraction')
    plt.xlabel('Metallicity')
    plt.legend(labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=3, fontsize=10)

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    props = dict(boxstyle='round', fc='white', ec='white', alpha=0.2)
    ax.text(0.8, 0.1, 'EAGLE', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('Returned_element_mass_as_function_Z_EAGLE.png', dpi=300)

