import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.integrate import quad, romberg
from scipy.integrate import cumtrapz, romb, simpson
from scipy.interpolate import interp1d
from sympy import Interval
import time
from scipy.interpolate import RegularGridInterpolator
import warnings

warnings.filterwarnings("ignore")

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


#def lifetimes(m,Z):
def inverse_lifetime(Z, t):

    mass_range = np.arange(np.log10(1), np.log10(100), 0.01)
    mass_range = 10**mass_range
    tm = np.zeros(len(mass_range))

    # Write data to HDF5
    with h5py.File('../data/EAGLE_yieldtables/Lifetimes.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Metallicities = data_file["Metallicities"][:]
        lifetimes = data_file["Lifetimes"][:][:]

    if Z < np.min(Metallicities):
        f = interp1d(Masses, lifetimes[0, :])
        # result = f(m) / 1e6 # Myr
        for i in range(len(mass_range)):
            # tm[i] = lifetimes(mass_range[i],Z) / 1e6 # Myr
            tm[i] = f(mass_range[i]) / 1e6 #Myr

    else:
        f = RegularGridInterpolator((Metallicities, Masses), lifetimes)
        # result = f(np.array([Z,m])) / 1e6 # Myr
        for i in range(len(mass_range)):
            # tm[i] = lifetimes(mass_range[i],Z) / 1e6 # Myr
            tm[i] = f(np.array([Z,mass_range[i]])) / 1e6 # Myr


    select = np.where(tm <= t)[0]
    mass_limit = mass_range[select]

    return mass_limit


    # num_metals = len(Metallicities)
    # MZ = np.zeros(num_metals)
    # for i in range(num_metals):
    #     f = interp1d(Masses, lifetimes[i,:])
    #     MZ[i] = f(m)
    #
    # if Z < np.min(Metallicities):
    #     result = MZ[0]
    # else:
    #     f = interp1d(Metallicities, MZ)
    #     result = f(Z) / 1e6 # Myr
    return result

# def inverse_lifetime(Z, t):
#
#     mass_range = np.arange(1, 100, 0.1)
#     tm = np.zeros(len(mass_range))
#
#     for i in range(len(mass_range)):
#         tm[i] = lifetimes(mass_range[i],Z) / 1e6 # Myr
#
#     select = np.where(tm <= t)[0]
#     mass_limit = mass_range[select]
#
#     return mass_limit


def read_CCSN_COLIBRE(Z, element):

    total_mass_fraction = 0.0133714

    file = '../data/SNII_linear_extrapolation.hdf5'
    data_file = h5py.File(file, 'r')

    elements_all = np.array([k.decode() for k in data_file['Species_names'][:]])
    indx = np.where(elements_all == element)[0]

    metallicity_range = data_file['Metallicities'][:]
    metallicity_flag = [k.decode() for k in data_file['Yield_names'][:]]

    Masses = data_file["Masses"][:]

    indx_nearest = np.abs(metallicity_range - Z).argmin()

    if indx_nearest == len(metallicity_range) - 1:

        Ejected_mass_winds = data_file["/Yields/"+metallicity_flag[indx_nearest]+"/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/"+metallicity_flag[indx_nearest]+"/Ejected_mass_in_ccsn"][:][:]

        factor = metallicity_range[indx_nearest] / total_mass_fraction

        stellar_yields = Ejected_mass_ccsn[indx, :] + factor * mass_fraction[indx] * Ejected_mass_winds
        stellar_yields *= (1. + (Z - metallicity_range[indx_nearest]) / metallicity_range[indx_nearest])

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
        stellar_yields_1 = Ejected_mass_ccsn[indx, :] + factor * mass_fraction[indx] * Ejected_mass_winds

        Ejected_mass_winds = data_file["/Yields/"+metallicity_flag[indx_2]+"/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/"+metallicity_flag[indx_2]+"/Ejected_mass_in_ccsn"][:][:]
        factor = metallicity_range[indx_2] / total_mass_fraction
        stellar_yields_2 = Ejected_mass_ccsn[indx, :] + factor * mass_fraction[indx] * Ejected_mass_winds

        b = (stellar_yields_2 - stellar_yields_1) / (metallicity_range[indx_2] - metallicity_range[indx_1])
        a = stellar_yields_1 - b * metallicity_range[indx_1]

        stellar_yields = a + b * Z

    return Masses, stellar_yields


def read_AGB_COLIBRE(Z, element):

    total_mass_fraction = 0.0133714

    file = '../data/AGB.hdf5'
    data_file = h5py.File(file, 'r')

    elements_all = np.array([k.decode() for k in data_file['Species_names'][:]])
    indx = np.where(elements_all == element)[0]

    metallicity_range = data_file['Metallicities'][:]
    metallicity_flag = [k.decode() for k in data_file['Yield_names'][:]]

    Masses = data_file["Masses"][:]

    indx_nearest = np.abs(metallicity_range - Z).argmin()

    if indx_nearest == 0 or indx_nearest == len(metallicity_range)-1:

        Yield = data_file["/Yields/" + metallicity_flag[indx_nearest] + "/Yield"][:][:]
        Ejected_mass = data_file["/Yields/" + metallicity_flag[indx_nearest] + "/Ejected_mass"][:]

        factor = metallicity_range[indx_nearest] / total_mass_fraction
        stellar_yields = Yield[indx, :] + factor * mass_fraction[indx] * Ejected_mass
        stellar_yields *= (1. + (Z - metallicity_range[indx_nearest]) / metallicity_range[indx_nearest])

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
        stellar_yields_1 = Yield[indx, :] + factor * mass_fraction[indx] * Ejected_mass

        Yield = data_file["/Yields/" + metallicity_flag[indx_2] + "/Yield"][:][:]
        Ejected_mass = data_file["/Yields/" + metallicity_flag[indx_2] + "/Ejected_mass"][:]
        factor = metallicity_range[indx_2] / total_mass_fraction
        stellar_yields_2 = Yield[indx, :] + factor * mass_fraction[indx] * Ejected_mass

        b = (stellar_yields_2 - stellar_yields_1 ) / (metallicity_range[indx_2] - metallicity_range[indx_1])
        a = stellar_yields_1 - b * metallicity_range[indx_1]

        stellar_yields = a + b * Z

    return Masses, stellar_yields


def returned_mass(Fe_H, element):

    IMF_integral = integrate_IMF(1, 100)

    Hydrogen_fraction = 0.73738788833  # H
    Iron_fraction = 0.00129199252      # Fe
    Z_0 = 0.0133714                    # Default metallicity
    Fe_H_0 = Iron_fraction / Hydrogen_fraction # default fraction

    Fe_H_Sun_Asplund = 7.5
    mp_in_cgs = 1.6737236e-24
    mH_in_cgs = 1.00784 * mp_in_cgs
    mFe_in_cgs = 55.845 * mp_in_cgs

    Fe_H_Sun = Fe_H_Sun_Asplund - 12.0 - np.log10(mH_in_cgs / mFe_in_cgs)

    Fe_H_1 = 10**(Fe_H + Fe_H_Sun)

    # If we assume the relative fraction between elements is unchanged
    # then Fe_1 = Fe_0 x Z_1 / Z_default,
    # with Z_1 the metallicity that corresponds to Fe_H = -2, say.
    # Additionally, X_1 + Y_1 + Z_1  = 1,
    # (1 - Z_1) / (1 _ Z_0) x (X_0 + Y_0) + Z_1  = 1
    # So H_1 = (1 - Z_1) / (1 _ Z_0) x H_0
    # This means Fe_1 / H_1 = (Z_1 / Z_default) / [(1 - Z_1) / (1 _ Z_default)] x Fe_0 / H_0

    metallicity = Z_0 * Fe_H_1 / (Fe_H_0 + Z_0 * (Fe_H_1 - Fe_H_0))

    AGB_masses, AGB_yields = read_AGB_COLIBRE(metallicity, element)
    CCSN_masses, CCSN_yields = read_CCSN_COLIBRE(metallicity, element)
    # print('CC masses & CC yields')
    # print('Element,',element)
    # print(CCSN_masses)
    # print(CCSN_yields)

    # Note different mass ranges!
    AGB_interval = Interval(1, 8)
    CCSN_interval = Interval(8, 40)

    # Calling interpolation function for the yields calculation
    AGB_fyields = interp1d(AGB_masses, AGB_yields)
    CCSN_fyields = interp1d(CCSN_masses,CCSN_yields)

    time_array = np.arange(-1, np.log10(500), 0.05)  # Myr
    time_array = 10**time_array
    Phi = np.zeros(len(time_array))
    Phi_AGB = np.zeros(len(time_array))
    Phi_CCSN = np.zeros(len(time_array))

    for j, ti in enumerate(time_array):

        mZ_ti = inverse_lifetime(metallicity, ti)

        if len(mZ_ti) <= 1: continue
        if np.min(mZ_ti) < 40:
            yields_AGB = np.zeros(len(mZ_ti))
            yields_CCSN = np.zeros(len(mZ_ti))
            imf = np.zeros(len(mZ_ti))

            for i, mZ in enumerate(mZ_ti):

                if CCSN_interval.contains(mZ):
                    yields_CCSN[i] = CCSN_fyields(mZ)

                if AGB_interval.contains(mZ):
                    yields_AGB[i] = AGB_fyields(mZ)

                imf[i] = imf_lin(mZ)

            Phi[j] = simpson(imf * yields_AGB + imf * yields_CCSN, x= mZ_ti)
            Phi_AGB[j] = simpson(imf * yields_AGB, x= mZ_ti)
            Phi_CCSN[j] = simpson(imf * yields_CCSN, x= mZ_ti)

    Phi /= IMF_integral
    Phi_AGB /= IMF_integral
    Phi_CCSN /= IMF_integral

    return Phi, time_array, Phi_AGB, Phi_CCSN

def calculate_hydrogen(Fe_H):

    IMF_integral = integrate_IMF(1, 100)

    Hydrogen_fraction = 0.73738788833  # H
    Oxygen_fraction = 0.00573271036    # O
    Iron_fraction = 0.00129199252      # Fe
    Z_0 = 0.0133714                    # Default metallicity
    Fe_H_0 = Iron_fraction / Hydrogen_fraction # default fraction

    Fe_H_Sun_Asplund = 7.5
    mp_in_cgs = 1.6737236e-24
    mH_in_cgs = 1.00784 * mp_in_cgs
    mFe_in_cgs = 55.845 * mp_in_cgs

    Fe_H_Sun = Fe_H_Sun_Asplund - 12.0 - np.log10(mH_in_cgs / mFe_in_cgs)

    Fe_H_1 = 10**(Fe_H + Fe_H_Sun)

    # If we assume the relative fraction between elements is unchanged
    # then Fe_1 = Fe_0 x Z_1 / Z_default,
    # with Z_1 the metallicity that corresponds to Fe_H = -2, say.
    # Additionally, X_1 + Y_1 + Z_1  = 1,
    # (1 - Z_1) / (1 _ Z_0) x (X_0 + Y_0) + Z_1  = 1
    # So H_1 = (1 - Z_1) / (1 _ Z_0) x H_0
    # This means Fe_1 / H_1 = (Z_1 / Z_default) / [(1 - Z_1) / (1 _ Z_default)] x Fe_0 / H_0

    metallicity = Z_0 * Fe_H_1 / (Fe_H_0 + Z_0 * (Fe_H_1 - Fe_H_0))

    b = (1. - metallicity) / (1. - Z_0)
    a = metallicity / Z_0
    O_H = a * Oxygen_fraction / (b * Hydrogen_fraction)
    O_H = np.log10(O_H) + 12.0 + np.log10(mH_in_cgs / mO_in_cgs)
    print(O_H)
    Hydrogen = b * Hydrogen_fraction

    # Note different mass ranges!
    AGB_masses = Interval(1, 8)
    CCSN_masses = Interval(8, 40)

    time_array = np.arange(-1, np.log10(500), 0.05)  # Myr
    time_array = 10**time_array
    Phi = np.zeros(len(time_array))
    Phi_AGB = np.zeros(len(time_array))
    Phi_CCSN = np.zeros(len(time_array))

    for j, ti in enumerate(time_array):

        mZ_ti = inverse_lifetime(metallicity, ti)

        if len(mZ_ti) <= 1: continue
        if np.min(mZ_ti) < 40:
            yields_AGB = np.zeros(len(mZ_ti))
            yields_CCSN = np.zeros(len(mZ_ti))
            imf = np.zeros(len(mZ_ti))

            for i, mZ in enumerate(mZ_ti):

                if CCSN_masses.contains(mZ):
                    yields_CCSN[i] = Hydrogen * mZ

                if AGB_masses.contains(mZ):
                    yields_AGB[i] = Hydrogen * mZ

                imf[i] = imf_lin(mZ)

            Phi[j] = simpson(imf * yields_AGB + imf * yields_CCSN, x= mZ_ti)
            Phi_AGB[j] = simpson(imf * yields_AGB, x= mZ_ti)
            Phi_CCSN[j] = simpson(imf * yields_CCSN, x= mZ_ti)

    Phi /= IMF_integral
    Phi_AGB /= IMF_integral
    Phi_CCSN /= IMF_integral

    return Phi, time_array, Phi_AGB, Phi_CCSN

if __name__ == "__main__":


    # print(Phi_N)
    # print(Phi_O)
    # print(time_array)

    # ####
    # plt.figure()
    # ax = plt.subplot(1, 1, 1)
    # ax.grid(True)
    # plt.grid(which='major', linestyle='-', linewidth=0.3)
    # plt.grid(which='minor', linestyle=':', linewidth=0.3)
    #
    # plt.plot(time_array, Phi_N, '-', color='tab:blue', label='N')
    # plt.plot(time_array, Phi_O, '-', color='tab:orange', label='O')
    #
    # plt.ylabel('Returned Mass')
    # plt.xlabel('Time [Myr]')
    #
    # # plt.axis([-0.5, 13.5, 1e-5, 1])
    # # plt.yscale('log')
    # plt.legend(loc=[0.5, 0.6], labelspacing=0.2, handlelength=0.8,
    #            handletextpad=0.3, frameon=False, columnspacing=0.4,
    #            ncol=1, fontsize=10)
    # ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.savefig('Mass_returned_CNO.png', dpi=300)
    #
    # ########################
    # plt.figure()
    # ax = plt.subplot(1, 1, 1)
    # ax.grid(True)
    # plt.grid(which='major', linestyle='-', linewidth=0.3)
    # plt.grid(which='minor', linestyle=':', linewidth=0.3)
    #
    # plt.plot(time_array, Phi_N, '-', color='tab:blue', label='Total')
    # plt.plot(time_array, Phi_N_AGB, '--', color='tab:blue', label='AGB')
    # plt.plot(time_array, Phi_N_CCSN, ':', color='tab:blue', label='CCSN')
    #
    # plt.ylabel('Returned Mass Nitrogen')
    # plt.xlabel('Time [Myr]')
    #
    # # plt.axis([-0.5, 13.5, 1e-5, 1])
    # # plt.yscale('log')
    # plt.legend(loc=[0.5, 0.6], labelspacing=0.2, handlelength=0.8,
    #            handletextpad=0.3, frameon=False, columnspacing=0.4,
    #            ncol=1, fontsize=10)
    # ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.savefig('Mass_returned_N.png', dpi=300)

    ########################
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-', linewidth=0.3)
    plt.grid(which='minor', linestyle=':', linewidth=0.3)

    Fe_H = -2
    Phi_N, time_array, Phi_N_AGB, Phi_N_CCSN = returned_mass(Fe_H, 'Nitrogen')
    Phi_O, time_array, Phi_O_AGB, Phi_O_CCSN = returned_mass(Fe_H, 'Oxygen')
    Phi_H, time_array, Phi_H_AGB, Phi_H_CCSN = returned_mass(Fe_H, 'Hydrogen')

    plt.plot(time_array, np.log10(Phi_N / Phi_O), '-', color='tab:blue', label='Total')
    plt.plot(time_array, np.log10(Phi_N_AGB / Phi_O), '--', color='tab:blue', label='AGB')
    plt.plot(time_array, np.log10(Phi_N_CCSN / Phi_O), ':', color='tab:blue', label='CCSN')

    plt.ylabel('$\log_{10}$(N/O)')
    plt.xlabel('Time [Myr]')

    # plt.axis([-0.5, 13.5, 1e-5, 1])
    # plt.yscale('log')
    plt.legend(loc=[0.6, 0.1], labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=1, fontsize=10)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('Mass_returned_NoverO_FeH-2.png', dpi=300)

    ########################
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-', linewidth=0.3)
    plt.grid(which='minor', linestyle=':', linewidth=0.3)

    Fe_H = -2
    Phi_N, time_array, Phi_N_AGB, Phi_N_CCSN = returned_mass(Fe_H, 'Nitrogen')
    Phi_O, time_array, Phi_O_AGB, Phi_O_CCSN = returned_mass(Fe_H, 'Oxygen')
    Phi_H, time_array, Phi_H_AGB, Phi_H_CCSN = returned_mass(Fe_H, 'Hydrogen')

    plt.plot(time_array, np.log10(Phi_N / Phi_O), '-', color='tab:blue', label='[Fe/H]=-2')
    plt.plot(time_array, np.log10(Phi_N_AGB / Phi_O), '--', color='tab:blue')#, label='AGB')
    plt.plot(time_array, np.log10(Phi_N_CCSN / Phi_O), ':', color='tab:blue')#, label='CCSN')

    Fe_H = -1
    Phi_N, time_array, Phi_N_AGB, Phi_N_CCSN = returned_mass(Fe_H, 'Nitrogen')
    Phi_O, time_array, Phi_O_AGB, Phi_O_CCSN = returned_mass(Fe_H, 'Oxygen')

    plt.plot(time_array, np.log10(Phi_N / Phi_O), '-', color='tab:orange', label='[Fe/H]=-1')
    plt.plot(time_array, np.log10(Phi_N_AGB / Phi_O), '--', color='tab:orange')#, label='AGB')
    plt.plot(time_array, np.log10(Phi_N_CCSN / Phi_O), ':', color='tab:orange')#, label='CCSN')

    Fe_H = 0
    Phi_N, time_array, Phi_N_AGB, Phi_N_CCSN = returned_mass(Fe_H, 'Nitrogen')
    Phi_O, time_array, Phi_O_AGB, Phi_O_CCSN = returned_mass(Fe_H, 'Oxygen')

    plt.plot(time_array, np.log10(Phi_N / Phi_O), '-', color='crimson', label='[Fe/H]=0')
    plt.plot(time_array, np.log10(Phi_N_AGB / Phi_O), '--', color='crimson')#, label='AGB')
    plt.plot(time_array, np.log10(Phi_N_CCSN / Phi_O), ':', color='crimson')#, label='CCSN')

    Fe_H = 1
    Phi_N, time_array, Phi_N_AGB, Phi_N_CCSN = returned_mass(Fe_H, 'Nitrogen')
    Phi_O, time_array, Phi_O_AGB, Phi_O_CCSN = returned_mass(Fe_H, 'Oxygen')

    plt.plot(time_array, np.log10(Phi_N / Phi_O), '-', color='darkblue', label='[Fe/H]=1')
    plt.plot(time_array, np.log10(Phi_N_AGB / Phi_O), '--', color='darkblue')#, label='AGB')
    plt.plot(time_array, np.log10(Phi_N_CCSN / Phi_O), ':', color='darkblue')#, label='CCSN')

    plt.ylabel('$\log_{10}$(N/O)')
    plt.xlabel('Time [Myr]')

    # plt.axis([-0.5, 13.5, 1e-5, 1])
    # plt.yscale('log')
    plt.legend(loc=[0.6, 0.1], labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=1, fontsize=10)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('Mass_returned_NoverO.png', dpi=300)


    # mp_in_cgs = 1.6737236e-24
    # mH_in_cgs = 1.00784 * mp_in_cgs
    # mFe_in_cgs = 55.845 * mp_in_cgs
    # mO_in_cgs = 15.999 * mp_in_cgs
    # mMg_in_cgs = 24.305 * mp_in_cgs
    # mC_in_cgs = 12.0107 * mp_in_cgs
    # mN_in_cgs = 14.0067 * mp_in_cgs
    #
    #
    # ########################
    # plt.figure()
    # ax = plt.subplot(1, 1, 1)
    # ax.grid(True)
    # plt.grid(which='major', linestyle='-', linewidth=0.3)
    # plt.grid(which='minor', linestyle=':', linewidth=0.3)
    #
    # Fe_H = -2
    # # calculate_hydrogen(Fe_H)
    # # Phi_H, time_array, Phi_H_AGB, Phi_H_CCSN = calculate_hydrogen(Fe_H)
    #
    # Phi_N, time_array, Phi_N_AGB, Phi_N_CCSN = returned_mass(Fe_H, 'Nitrogen')
    # Phi_O, time_array, Phi_O_AGB, Phi_O_CCSN = returned_mass(Fe_H, 'Oxygen')
    # Phi_H, time_array, Phi_H_AGB, Phi_H_CCSN = returned_mass(Fe_H, 'Hydrogen')
    #
    # xrange = 12 + np.log10(Phi_O / Phi_H) + np.log10(mH_in_cgs / mO_in_cgs) - 0.9
    # plt.plot(xrange, np.log10(Phi_N / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), '-', color='tab:blue', label='Total ([Fe/H]=-2)')
    # plt.plot(xrange, np.log10(Phi_N_AGB / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), '--', color='tab:blue')#, label='AGB')
    # plt.plot(xrange, np.log10(Phi_N_CCSN / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), ':', color='tab:blue')#, label='CCSN')
    #
    # Fe_H = -1
    # Phi_N, time_array, Phi_N_AGB, Phi_N_CCSN = returned_mass(Fe_H, 'Nitrogen')
    # Phi_O, time_array, Phi_O_AGB, Phi_O_CCSN = returned_mass(Fe_H, 'Oxygen')
    # Phi_H, time_array, Phi_H_AGB, Phi_H_CCSN = returned_mass(Fe_H, 'Hydrogen')
    #
    # xrange = 12 + np.log10(Phi_O / Phi_H) + np.log10(mH_in_cgs / mO_in_cgs) - 0.9
    # plt.plot(xrange, np.log10(Phi_N / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), '-', color='tab:orange', label='Total ([Fe/H]=-1)')
    # plt.plot(xrange, np.log10(Phi_N_AGB / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), '--', color='tab:orange')#, label='AGB')
    # plt.plot(xrange, np.log10(Phi_N_CCSN / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), ':', color='tab:orange')#, label='CCSN')
    #
    # Fe_H = 0
    # Phi_N, time_array, Phi_N_AGB, Phi_N_CCSN = returned_mass(Fe_H, 'Nitrogen')
    # Phi_O, time_array, Phi_O_AGB, Phi_O_CCSN = returned_mass(Fe_H, 'Oxygen')
    # Phi_H, time_array, Phi_H_AGB, Phi_H_CCSN = returned_mass(Fe_H, 'Hydrogen')
    #
    # xrange = 12 + np.log10(Phi_O / Phi_H) + np.log10(mH_in_cgs / mO_in_cgs) - 0.9
    # plt.plot(xrange, np.log10(Phi_N / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), '-', color='crimson', label='Total ([Fe/H]=0)')
    # plt.plot(xrange, np.log10(Phi_N_AGB / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), '--', color='crimson')#, label='AGB')
    # plt.plot(xrange, np.log10(Phi_N_CCSN / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), ':', color='crimson')#, label='CCSN')
    #
    # Fe_H = 1
    # Phi_N, time_array, Phi_N_AGB, Phi_N_CCSN = returned_mass(Fe_H, 'Nitrogen')
    # Phi_O, time_array, Phi_O_AGB, Phi_O_CCSN = returned_mass(Fe_H, 'Oxygen')
    # Phi_H, time_array, Phi_H_AGB, Phi_H_CCSN = returned_mass(Fe_H, 'Hydrogen')
    #
    # xrange = 12 + np.log10(Phi_O / Phi_H) + np.log10(mH_in_cgs / mO_in_cgs) - 0.9
    # plt.plot(xrange, np.log10(Phi_N / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), '-', color='darkblue', label='Total ([Fe/H]=1)')
    # plt.plot(xrange, np.log10(Phi_N_AGB / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), '--', color='darkblue')#, label='AGB')
    # plt.plot(xrange, np.log10(Phi_N_CCSN / Phi_O) + np.log10(mO_in_cgs / mN_in_cgs), ':', color='darkblue')#, label='CCSN')
    #
    # plt.ylabel('$\log_{10}$(N/O)')
    # plt.xlabel('12+$\log_{10}$(O/H)')
    #
    # # plt.axis([-0.5, 13.5, 1e-5, 1])
    # # plt.yscale('log')
    # plt.legend(loc=[0.05, 0.2], labelspacing=0.2, handlelength=0.8,
    #            handletextpad=0.3, frameon=False, columnspacing=0.4,
    #            ncol=1, fontsize=10)
    # ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.savefig('NoverO_OohverH.png', dpi=300)