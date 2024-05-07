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
    stellar_yields_total = np.zeros((num_elements, num_mass_bins))

    indx_nearest = np.abs(metallicity_range - Z).argmin()
    print(metallicity_flag[indx_nearest])
    Ejected_mass_winds = data_file["/Yields/"+metallicity_flag[indx_nearest]+"/Ejected_mass_in_winds"][:]
    Ejected_mass_ccsn = data_file["/Yields/"+metallicity_flag[indx_nearest]+"/Ejected_mass_in_ccsn"][:][:]

    factor = metallicity_range[indx_nearest] / total_mass_fraction

    for i in range(num_mass_bins):
        stellar_yields_total[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]

    return stellar_yields_total, Masses

def read_SNII_EAGLE(Z):

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

    indx_nearest = np.abs(metallicity_range - Z).argmin()
    print(metallicity_flag[indx_nearest])
    Yield = data_file["/Yields/" + metallicity_flag[indx_nearest] + "/Yield"][:][:]
    Ejected_mass = data_file["/Yields/" + metallicity_flag[indx_nearest] + "/Ejected_mass"][:]

    factor = metallicity_range[indx_nearest] / total_mass_fraction

    for i in range(num_mass_bins):
        stellar_yields[:, i] = Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]

    return stellar_yields, Masses


if __name__ == "__main__":


    CCSN_COLIBRE_1, Masses = read_SNII_COLIBRE(0)
    CCSN_COLIBRE_2, Masses = read_SNII_COLIBRE(0.001)
    CCSN_COLIBRE_3, Masses = read_SNII_COLIBRE(0.004)
    CCSN_COLIBRE_4, Masses = read_SNII_COLIBRE(0.05)

    CCSN_EAGLE_1, Masses_2 = read_SNII_EAGLE(0.0004)
    CCSN_EAGLE_2, Masses_2 = read_SNII_EAGLE(0.004)
    CCSN_EAGLE_3, Masses_2 = read_SNII_EAGLE(0.05)

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
    color = ['darkblue','tab:blue','tab:green','tab:orange','crimson']
    plt.plot(Masses, CCSN_COLIBRE_1[0,:], '-', color='black',label='EAGLE (Z=0.0004)')
    plt.plot(Masses, CCSN_COLIBRE_1[0,:], '--', color='black',label='COLIBRE (Z=0)')

    plt.plot(Masses, CCSN_COLIBRE_1[0,:], '--', color=color[0])#,label='Hydrogen')
    plt.plot(Masses, CCSN_COLIBRE_1[2,:], '--', color=color[1])#,label='Carbon')
    plt.plot(Masses, CCSN_COLIBRE_1[3,:], '--', color=color[2])#,label='Nitrogen')
    plt.plot(Masses, CCSN_COLIBRE_1[4,:], '--', color=color[3])#,label='Oxygen')
    plt.plot(Masses, CCSN_COLIBRE_1[8,:], '--', color=color[4])#,label='Iron')

    plt.plot(Masses_2, CCSN_EAGLE_1[0,:], '-', color=color[0],label='Hydrogen')
    plt.plot(Masses_2, CCSN_EAGLE_1[2,:], '-', color=color[1],label='Carbon')
    plt.plot(Masses_2, CCSN_EAGLE_1[3,:], '-', color=color[2],label='Nitrogen')
    plt.plot(Masses_2, CCSN_EAGLE_1[4,:], '-', color=color[3],label='Oxygen')
    plt.plot(Masses_2, CCSN_EAGLE_1[8,:], '-', color=color[4],label='Iron')

    # plt.axis([-0.5, 13.5, 1e-6, 1])
    plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('Yields [M$_{\odot}$]')
    plt.xlabel('Mass [M$_{\odot}$]')
    plt.legend(labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=3, fontsize=10)

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('COLIBRE_yield_tables_Z_0.png', dpi=300)

    ########################
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-', linewidth=0.3)
    color = ['darkblue','tab:blue','tab:green','tab:orange','crimson']
    plt.plot(Masses, CCSN_COLIBRE_2[0,:], '-', color='black',label='EAGLE (Z=0.0004)')
    plt.plot(Masses, CCSN_COLIBRE_2[0,:], '--', color='black',label='COLIBRE (Z=0.001)')

    plt.plot(Masses, CCSN_COLIBRE_2[0,:], '--', color=color[0])#,label='Hydrogen')
    plt.plot(Masses, CCSN_COLIBRE_2[2,:], '--', color=color[1])#,label='Carbon')
    plt.plot(Masses, CCSN_COLIBRE_2[3,:], '--', color=color[2])#,label='Nitrogen')
    plt.plot(Masses, CCSN_COLIBRE_2[4,:], '--', color=color[3])#,label='Oxygen')
    plt.plot(Masses, CCSN_COLIBRE_2[8,:], '--', color=color[4])#,label='Iron')

    plt.plot(Masses_2, CCSN_EAGLE_1[0,:], '-', color=color[0],label='Hydrogen')
    plt.plot(Masses_2, CCSN_EAGLE_1[2,:], '-', color=color[1],label='Carbon')
    plt.plot(Masses_2, CCSN_EAGLE_1[3,:], '-', color=color[2],label='Nitrogen')
    plt.plot(Masses_2, CCSN_EAGLE_1[4,:], '-', color=color[3],label='Oxygen')
    plt.plot(Masses_2, CCSN_EAGLE_1[8,:], '-', color=color[4],label='Iron')

    # plt.axis([-0.5, 13.5, 1e-6, 1])
    plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('Yields [M$_{\odot}$]')
    plt.xlabel('Mass [M$_{\odot}$]')
    plt.legend(labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=3, fontsize=10)

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('COLIBRE_yield_tables_Z_0p001.png', dpi=300)

    ########################
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-', linewidth=0.3)
    color = ['darkblue', 'tab:blue', 'tab:green', 'tab:orange', 'crimson']
    plt.plot(Masses, CCSN_COLIBRE_3[0, :], '-', color='black',label='EAGLE (Z=0.004)')
    plt.plot(Masses, CCSN_COLIBRE_3[0, :], '--', color='black',label='COLIBRE (Z=0.004)')

    plt.plot(Masses, CCSN_COLIBRE_3[0, :], '--', color=color[0])#, label='Hydrogen')
    plt.plot(Masses, CCSN_COLIBRE_3[2, :], '--', color=color[1])#, label='Carbon')
    plt.plot(Masses, CCSN_COLIBRE_3[3, :], '--', color=color[2])#, label='Nitrogen')
    plt.plot(Masses, CCSN_COLIBRE_3[4, :], '--', color=color[3])#, label='Oxygen')
    plt.plot(Masses, CCSN_COLIBRE_3[8, :], '--', color=color[4])#, label='Iron')

    plt.plot(Masses_2, CCSN_EAGLE_2[0, :], '-', color=color[0],label='Hydrogen')
    plt.plot(Masses_2, CCSN_EAGLE_2[2, :], '-', color=color[1],label='Carbon')
    plt.plot(Masses_2, CCSN_EAGLE_2[3, :], '-', color=color[2],label='Nitrogen')
    plt.plot(Masses_2, CCSN_EAGLE_2[4, :], '-', color=color[3],label='Oxygen')
    plt.plot(Masses_2, CCSN_EAGLE_2[8, :], '-', color=color[4],label='Iron')

    # plt.axis([-0.5, 13.5, 1e-6, 1])
    plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('Yields [M$_{\odot}$]')
    plt.xlabel('Mass [M$_{\odot}$]')
    plt.legend(labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=3, fontsize=10)

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('COLIBRE_yield_tables_Z_0p004.png', dpi=300)

    ########################
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-', linewidth=0.3)
    color = ['darkblue', 'tab:blue', 'tab:green', 'tab:orange', 'crimson']
    plt.plot(Masses_2, CCSN_EAGLE_3[2, :], '--', color='black', label='COLIBRE (Z=0.05)')
    plt.plot(Masses_2, CCSN_EAGLE_3[2, :], '-', color='black', label='EAGLE (Z=0.05)')

    # plt.plot(Masses, CCSN_COLIBRE_4[0, :], '--', color=color[0])  # , label='Hydrogen')
    plt.plot(Masses, CCSN_COLIBRE_4[2, :], '--', color=color[1])  # , label='Carbon')
    plt.plot(Masses, CCSN_COLIBRE_4[3, :], '--', color=color[2])  # , label='Nitrogen')
    plt.plot(Masses, CCSN_COLIBRE_4[4, :], '--', color=color[3])  # , label='Oxygen')
    plt.plot(Masses, CCSN_COLIBRE_4[8, :], '--', color=color[4])  # , label='Iron')

    # plt.plot(Masses_2, CCSN_EAGLE_3[0, :], '-', color=color[0], label='Hydrogen')
    plt.plot(Masses_2, CCSN_EAGLE_3[2, :], '-', color=color[1], label='Carbon')
    plt.plot(Masses_2, CCSN_EAGLE_3[3, :], '-', color=color[2], label='Nitrogen')
    plt.plot(Masses_2, CCSN_EAGLE_3[4, :], '-', color=color[3], label='Oxygen')
    plt.plot(Masses_2, CCSN_EAGLE_3[8, :], '-', color=color[4], label='Iron')

    # plt.axis([-0.5, 13.5, 1e-6, 1])
    plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('Yields [M$_{\odot}$]')
    plt.xlabel('Mass [M$_{\odot}$]')
    plt.legend(labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=3, fontsize=10)

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('COLIBRE_yield_tables_Z_0p05.png', dpi=300)