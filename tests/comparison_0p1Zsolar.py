import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.integrate import quad, romberg
from scipy.integrate import cumtrapz, romb, simpson
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as tck

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

    Masses = np.arange(m_min, m_max)
    imf_array = np.zeros(len(Masses))
    for i in range(0, len(Masses)):
        imf_array[i] = imf_lin(Masses[i])

    IMF_int = simpson(imf_array * Masses, x=Masses)
    return IMF_int

def lifetimes(m,Z):

    # Write data to HDF5
    with h5py.File('../data/EAGLE_yieldtables/Lifetimes.hdf5', 'r') as data_file:
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

def read_AGB_COLIBRE(metallicity_flag, metallicity, total_mass_fraction):

    #['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron']
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
    factor = metallicity / total_mass_fraction

    # Write data to HDF5
    with h5py.File('../data/AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Yield = data_file["/Yields/Z_"+metallicity_flag+"/Yield"][:][:]
        Ejected_mass = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass"][:]
        Total_metals = data_file["/Yields/Z_"+metallicity_flag+"/Total_Metals"][:]

    select = np.where((Masses <= 8) & (Masses >= 1))[0]
    Masses = Masses[select]
    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    stellar_yields = np.zeros((num_elements, num_mass_bins))
    total_yields = np.zeros(num_mass_bins)

    for i in range(num_mass_bins):
        stellar_yields[:, i] = Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]
        total_yields[i] = Total_metals[i] + factor * total_mass_fraction * Ejected_mass[i]

    new_mass_range = np.arange(1,8,0.1)
    new_imf_range = np.zeros(len(new_mass_range))
    new_stellar_yields_range = np.zeros(len(new_mass_range))
    colibre = np.zeros(9)
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

        colibre[i] = simpson(new_imf_range * new_stellar_yields_range, x=new_mass_range)

    return colibre

def read_AGB_EAGLE(metallicity_flag, metallicity, total_mass_fraction):

    factor = metallicity / total_mass_fraction

    #['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Iron']
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])

    # Write data to HDF5
    with h5py.File('../data/EAGLE_yieldtables/AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Yield = data_file["/Yields/Z_"+metallicity_flag+"/Yield"][:][:]
        Ejected_mass = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass"][:]
        Total_metals = data_file["/Yields/Z_"+metallicity_flag+"/Total_Metals"][:]

    select = np.where((Masses <= 6) & (Masses >= 1))[0]
    Masses = Masses[select]
    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    stellar_yields = np.zeros((num_elements, num_mass_bins))
    total_yields = np.zeros(num_mass_bins)

    for i in range(num_mass_bins):
        stellar_yields[:, i] = Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]
        total_yields[i] = Total_metals[i] + factor * total_mass_fraction * Ejected_mass[i]

    new_mass_range = np.arange(1,6,0.1)
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
                new_stellar_yields_range[j] = stellar_yields[i, -1] *  new_mass_range[j] / np.max(Masses)
            else:
                new_stellar_yields_range[j] = f(new_mass_range[j])

        eagle[i] = simpson(new_imf_range * new_stellar_yields_range, x=new_mass_range)

    # imf_array = np.zeros(num_mass_bins)
    # for i in range(0, num_mass_bins):
    #     imf_array[i] = imf_lin(Masses[i])
    # colibre = np.zeros(9)
    # for i in range(0,9):
    #     colibre[i] = simpson(imf_array * stellar_yields[i,:], x=Masses)

    return eagle


def read_SNIa_EAGLE(total_mass_fraction):

    # Write data to HDF5
    with h5py.File('./data/EAGLE_yieldtables/SNIa.hdf5', 'r') as data_file:
        stellar_yields = data_file["Yield"][:]

    indx = np.array([0,1,5,6,7,9,11,13,25])
    stellar_yields = stellar_yields[indx]
    mass_limit = inverse_lifetime(total_mass_fraction)
    IMF_int = integrate_IMF(0.1, 100)

    masses = np.arange(np.min(mass_limit),100,0.1)
    IMF = np.zeros(len(masses))
    NSNIa = np.zeros(len(masses))
    for i in range(len(masses)):
        IMF[i] = imf_lin(masses[i])
        NSNIa[i] = SNIa_rates(masses[i], total_mass_fraction)

    integral_mej = simpson(IMF * NSNIa * masses, x=masses)

    colibre = stellar_yields * integral_mej / IMF_int
    return colibre

def read_SNIa_COLIBRE(total_mass_fraction):

    # Write data to HDF5
    with h5py.File('./data/SNIa.hdf5', 'r') as data_file:
        stellar_yields = data_file["Yield"][:]

    mass_limit = inverse_lifetime(total_mass_fraction)
    IMF_int = integrate_IMF(0.1, 100)

    masses = np.arange(np.min(mass_limit),100,0.1)
    IMF = np.zeros(len(masses))
    NSNIa = np.zeros(len(masses))
    for i in range(len(masses)):
        IMF[i] = imf_lin(masses[i])
        NSNIa[i] = SNIa_rates(masses[i], total_mass_fraction)

    integral_mej = simpson(IMF * NSNIa * masses, x=masses)
    colibre = stellar_yields * integral_mej / IMF_int
    return colibre

def plot_AGB(total_mass_fraction):

    IMF_int = integrate_IMF(0.1, 100)

    colibre_Z0p14 = read_AGB_COLIBRE('0.004',0.004, total_mass_fraction)
    colibre_Z0p3 = read_AGB_COLIBRE('0.001',0.001, total_mass_fraction)
    Z = np.array([0.004, 0.001])
    colibre = colibre_Z0p3
    for i in range(len(colibre)):
        f = interp1d(Z, np.array([colibre_Z0p14[i], colibre_Z0p3[i]]))
        colibre[i] = f(total_mass_fraction) / IMF_int

    # eagle_Z0p02 = read_AGB_EAGLE('0.019',0.019, total_mass_fraction)
    eagle_Z0p008 = read_AGB_EAGLE('0.004',0.004, total_mass_fraction)

    # Z = np.array([0.008,0.019])
    eagle = eagle_Z0p008 * total_mass_fraction / 0.004
    eagle /= IMF_int
    # for i in range(len(eagle)):
    #     f = interp1d(Z, np.array([eagle_Z0p008[i], eagle_Z0p02[i]]))
    #     eagle[i] = f(total_mass_fraction) / IMF_int

    # tng = np.array([0.2, 0.07, 0.0012, 0.0008, 0.0015, 0.00045, 0.00015, 0.00018, 0.0003])
    # illustris = np.array([0.17, 6e-2, 0.0012, 6.5e-4, 1.3e-3, 4e-4, 1.2e-4, 1.6e-4, 2.5e-4])

    color_list = ['lightskyblue', 'steelblue', 'khaki', 'salmon']

    index = np.arange(9) * 1.5
    bar_width = 0.25
    opacity = 1
    plt.bar(index, colibre, bar_width, alpha=opacity,
            color=color_list[0], label='COLIBRE', edgecolor='black', linewidth=0.2)
    plt.bar(index + bar_width, eagle, bar_width, alpha=opacity,
            color=color_list[1], label='EAGLE', edgecolor='black', linewidth=0.2)
    # plt.bar(index + 2 * bar_width, tng, bar_width, alpha=opacity,
    #         color=color_list[2], label='TNG', edgecolor='black', linewidth=0.2)
    # plt.bar(index + 3 * bar_width, illustris, bar_width, alpha=opacity,
    #         color=color_list[3], label='Illustris', edgecolor='black', linewidth=0.2)

    # print('===AGN===')
    # elements = ['H','He','C','N','O','Ne','Mg','Si','Fe']
    # for i, elem in enumerate(elements):
    #     print(elem, eagle[i]/colibre[i],'(',colibre[i]/eagle[i],')',
    #           tng[i]/colibre[i], '(',colibre[i]/tng[i],')',
    #           illustris[i]/colibre[i], '(',colibre[i]/illustris[i],')')

def read_SNII_COLIBRE(metallicity_flag, metallicity, total_mass_fraction):

    #['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron'
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])

    factor = metallicity / total_mass_fraction

    # Write data to HDF5
    with h5py.File('../data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Ejected_mass_winds = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_ccsn"][:][:]

    select = np.where(Masses >= 8)[0]
    Masses = Masses[select]
    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    stellar_yields = np.zeros((num_elements, num_mass_bins))
    boost_factors = np.array([1,1,1.5,1,1,1,1.5,1,1])

    for i in range(num_mass_bins):
        stellar_yields[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]
        stellar_yields[:, i] *= boost_factors

    new_mass_range = np.arange(8, 40.1, 0.1)
    new_imf_range = np.zeros(len(new_mass_range))
    new_stellar_yields_range = np.zeros(len(new_mass_range))
    colibre = np.zeros(9)
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

        colibre[i] = simpson(new_imf_range * new_stellar_yields_range, x=new_mass_range)

    return colibre

def read_SNII_EAGLE(metallicity_flag, metallicity, total_mass_fraction):

    factor = metallicity / total_mass_fraction

    #['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Iron']
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])

    # Write data to HDF5
    with h5py.File('../data/EAGLE_yieldtables/SNII.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Yield = data_file["/Yields/Z_"+metallicity_flag+"/Yield"][:][:]
        Ejected_mass = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass"][:]

    select = np.where(Masses >= 6)[0]
    Masses = Masses[select]
    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    stellar_yields = np.zeros((num_elements, num_mass_bins))
    boost_factors = np.array([1,1,0.5,1,1,1.0,2.0,1.0,0.5])

    for i in range(num_mass_bins):
        stellar_yields[:, i] = Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]
        stellar_yields[:, i] *= boost_factors

    new_mass_range = np.arange(6, 100, 0.1)
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

def plot_SNII(total_mass_fraction):

    IMF_int = integrate_IMF(0.1, 100)
    colibre_Z0p02 = read_SNII_COLIBRE('0.004',0.004, total_mass_fraction)
    colibre_Z0p008 = read_SNII_COLIBRE('0.001',0.001, total_mass_fraction)

    eagle_Z0p02 = read_SNII_EAGLE('0.0004',0.0004, total_mass_fraction)
    eagle_Z0p008 = read_SNII_EAGLE('0.004',0.004, total_mass_fraction)

    num_bins = 9
    colibre = np.zeros(num_bins)
    eagle = np.zeros(num_bins)

    for i in range(num_bins):
        Z = np.array([0.0004, 0.004])
        f = interp1d(Z, np.array([eagle_Z0p008[i], eagle_Z0p02[i]]))
        eagle[i] = f(total_mass_fraction)

        Z = np.array([0.001, 0.004])
        f = interp1d(Z, np.array([colibre_Z0p008[i], colibre_Z0p02[i]]))
        colibre[i] = f(total_mass_fraction)

    eagle /= IMF_int
    colibre /= IMF_int

    # tng = np.array([8e-2, 6.5e-2, 2.5e-3, 5e-4, 1.5e-2, 5e-3, 1.5e-3, 1.1e-3, 8e-4])
    # illustris = np.array([1.1e-1, 8e-2, 8.5e-3, 9e-4, 1.7e-2, 2.1e-3, 5e-4, 1.8e-3, 1.5e-3])

    color_list = ['lightskyblue', 'steelblue', 'khaki', 'salmon']

    ####
    index = np.arange(9) * 1.5
    bar_width = 0.25
    opacity = 1
    plt.bar(index, colibre, bar_width, alpha=1, color=color_list[0],
            label='COLIBRE', edgecolor='black', linewidth=0.2)
    plt.bar(index + bar_width, eagle, bar_width, alpha=opacity,
            color=color_list[1], label='EAGLE', edgecolor='black', linewidth=0.2)
    # plt.bar(index + 2 * bar_width, tng, bar_width, alpha=opacity,
    #         color=color_list[2], label='TNG', edgecolor='black', linewidth=0.2)
    # plt.bar(index + 3 * bar_width, illustris, bar_width, alpha=opacity,
    #         color=color_list[3], label='Illustris', edgecolor='black', linewidth=0.2)

    # print('===CCSN===')
    # elements = ['H','He','C','N','O','Ne','Mg','Si','Fe']
    # for i, elem in enumerate(elements):
    #     print(elem, eagle[i]/colibre[i],'(',colibre[i]/eagle[i],')',
    #           tng[i]/colibre[i], '(',colibre[i]/tng[i],')',
    #           illustris[i]/colibre[i], '(',colibre[i]/illustris[i],')')


def read_SNIa_EAGLE(total_mass_fraction):

    # Write data to HDF5
    with h5py.File('../data/EAGLE_yieldtables/SNIa.hdf5', 'r') as data_file:
        stellar_yields = data_file["Yield"][:]

    indx = np.array([0,1,5,6,7,9,11,13,25])
    stellar_yields = stellar_yields[indx]
    mass_limit = inverse_lifetime(total_mass_fraction)

    IMF_int = integrate_IMF(0.1, 100)

    masses = np.arange(np.min(mass_limit),100,0.1)
    IMF = np.zeros(len(masses))
    NSNIa = 2e-3 # Msun^-1
    for i in range(len(masses)):
        IMF[i] = imf_lin(masses[i])

    integral_mej = simpson(IMF * NSNIa * masses, x=masses)

    colibre = stellar_yields * integral_mej / IMF_int
    return colibre

def read_SNIa_COLIBRE(total_mass_fraction):

    # Write data to HDF5
    with h5py.File('../data/LeungNomoto2018/SNIa_W7LeungNomoto2018.hdf5', 'r') as data_file:
        stellar_yields = data_file["Yield"][:]

    mass_limit = inverse_lifetime(total_mass_fraction)
    IMF_int = integrate_IMF(0.1, 100)

    masses = np.arange(np.min(mass_limit),100,0.1)
    IMF = np.zeros(len(masses))
    NSNIa = 2e-3 # Msun^-1
    for i in range(len(masses)):
        IMF[i] = imf_lin(masses[i])

    integral_mej = simpson(IMF * NSNIa * masses, x=masses)

    colibre = stellar_yields * integral_mej / IMF_int
    return colibre


def plot_SNIa(solar_metallicity):

    colibre = read_SNIa_COLIBRE(solar_metallicity)
    eagle = read_SNIa_EAGLE(solar_metallicity)

    # tng = np.array([1e-8, 1e-8, 6e-5, 1e-8, 2e-4, 6e-6, 1.5e-5, 2e-4, 1e-3])
    # illustris = np.array([1e-8, 1e-8, 3e-5, 1e-8, 8e-5, 2e-6, 8e-6, 9e-5, 4.5e-4])

    index = np.arange(9) * 1.5
    bar_width = 0.25
    width = 0.1
    opacity = 1

    color_list = ['lightskyblue', 'steelblue', 'khaki', 'salmon']

    plt.bar(index, colibre, bar_width, alpha=1, color=color_list[0],
            label='COLIBRE', edgecolor='black', linewidth=0.2)
    plt.bar(index + bar_width, eagle, bar_width, alpha=opacity,
            color=color_list[1], label='EAGLE', edgecolor='black', linewidth=0.2)
    # plt.bar(index + 2 * bar_width, tng, bar_width, alpha=opacity,
    #         color=color_list[2], label='TNG', edgecolor='black', linewidth=0.2)
    # plt.bar(index + 3 * bar_width, illustris, bar_width, alpha=opacity,
    #         color=color_list[3], label='Illustris', edgecolor='black', linewidth=0.2)

    # print('===SNIa===')
    # elements = ['H','He','C','N','O','Ne','Mg','Si','Fe']
    # for i, elem in enumerate(elements):
    #     print(elem, eagle[i]/colibre[i],'(',colibre[i]/eagle[i],')',
    #           tng[i]/colibre[i], '(',colibre[i]/tng[i],')',
    #           illustris[i]/colibre[i], '(',colibre[i]/illustris[i],')')


def make_comparison():

    # solar_metallicity = 0.0133714
    solar_metallicity = 0.1 * 0.014

    # Plot parameters
    params = {
        "font.size": 13,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (10, 3.8),
        "figure.subplot.left": 0.07,
        "figure.subplot.right": 0.99,
        "figure.subplot.bottom": 0.1,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.03,
        "figure.subplot.hspace": 0.03,
        "lines.markersize": 3,
        "lines.linewidth": 1,
        "figure.max_open_warning": 0,
        "axes.axisbelow": True,
    }
    rcParams.update(params)

    ######
    plt.figure()

    ax = plt.subplot(1, 3, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-',linewidth=0.3)
    plt.grid(which='minor', linestyle=':',linewidth=0.3)

    plot_AGB(solar_metallicity)
    index = np.arange(9) * 1.5
    bar_width = 0.25

    plt.ylabel('Returned Stellar Mass Fraction')
    plt.xticks(index + bar_width, ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe'))

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.35, 0.96, 'AGB yields', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    props = dict(boxstyle='square', fc='khaki', ec='darkblue', alpha=0.7)
    ax.text(0.71, 0.93, 'Z = 0.1Z$_{\odot}$', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props, color='darkblue')

    plt.axis([-0.5, 13.5, 1e-6, 0.3])
    plt.yscale('log')
    plt.legend(loc=[0.65, 0.72], labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=1, fontsize=12)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    ########################
    ########################
    ax = plt.subplot(1, 3, 2)
    ax.grid(True)
    plt.grid(which='major', linestyle='-',linewidth=0.3)
    plt.grid(which='minor', linestyle=':',linewidth=0.3)
    plot_SNII(solar_metallicity)

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.35, 0.96, 'CCSN yields', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.xticks(index + bar_width, ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe'))
    plt.axis([-0.5, 13.5, 1e-6, 0.3])
    plt.yscale('log')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_yaxis().set_ticklabels([])

    ########################
    ########################
    ax = plt.subplot(1, 3, 3)
    ax.grid(True)
    plt.grid(which='major', linestyle='-',linewidth=0.3)
    plt.grid(which='minor', linestyle=':',linewidth=0.3)
    plot_SNIa(solar_metallicity)

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.35, 0.96, 'SNIa yields', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.xticks(index + bar_width, ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe'))
    plt.axis([-0.5, 13.5, 1e-6, 0.3])
    plt.yscale('log')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_yaxis().set_ticklabels([])

    plt.savefig('./Comparison_0p1Zsolar.png', dpi=300)


if __name__ == "__main__":

    make_comparison()
