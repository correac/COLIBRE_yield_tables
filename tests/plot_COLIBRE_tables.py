import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.integrate import quad, romberg
from scipy.integrate import cumtrapz, romb, simpson
from scipy.interpolate import interp1d

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

# mass_fraction = np.array([0.73738788833, #H
#                           0.24924186942, #He
#                           0.0023647215,  #C
#                           0.0006928991,  #N
#                           0.00573271036, #O
#                           0.00125649278, #Ne
#                           0.00070797838, #Mg
#                           0.00066495154, #Si
#                           0.00129199252]) #Fe

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


def read_SNII_COLIBRE(metallicity_flag, metallicity):

    total_mass_fraction = 0.0133714

    #['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron'
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])

    factor = metallicity / total_mass_fraction

    # Write data to HDF5
    with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Ejected_mass_winds = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_ccsn"][:][:]
        Total_metals = data_file["/Yields/Z_"+metallicity_flag+"/Total_Mass_ejected"][:]

    select = np.where(Masses >= 8)[0]
    Masses = Masses[select]
    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    stellar_yields = np.zeros((num_elements, num_mass_bins))
    # total_yields = np.zeros(num_mass_bins)
    boost_factors = np.array([1,1,2.5,2,1,0.7,0.7,0.4,1])

    for i in range(num_mass_bins):
        stellar_yields[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]
        #stellar_yields[:, i] *= boost_factors
        # total_yields[i] = Total_metals[i] + factor * total_mass_fraction * Ejected_mass[i]

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

def plot_SNII():

    total_mass_fraction = 0.0133714
    IMF_int = integrate_IMF(0.1, 100)

    colibre_Z0 = read_SNII_COLIBRE('0.000',0.000) / IMF_int
    colibre_Z1 = read_SNII_COLIBRE('0.001',0.001) / IMF_int
    colibre_Z2 = read_SNII_COLIBRE('0.004',0.004) / IMF_int
    colibre_Z3 = read_SNII_COLIBRE('0.008',0.008) / IMF_int
    colibre_Z4 = read_SNII_COLIBRE('0.020',0.020) / IMF_int

    ####
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-',linewidth=0.3)
    plt.grid(which='minor', linestyle=':',linewidth=0.3)

    index = np.arange(9) * 1.5
    bar_width = 0.25
    width = 0.1
    opacity = 0.8
    plt.bar(index, colibre_Z0, bar_width, alpha=1, color='darkblue', label='Z=0.0')
    plt.bar(index + bar_width, colibre_Z1, bar_width, alpha=opacity, color='tab:blue', label='Z=0.001')
    plt.bar(index + 2 * bar_width, colibre_Z2, bar_width, alpha=1, color='tab:orange', label='Z=0.004')
    plt.bar(index + 3 * bar_width, colibre_Z3, bar_width, alpha=opacity, color='lightgreen', label='Z=0.008')
    plt.bar(index + 4 * bar_width, colibre_Z4, bar_width, alpha=opacity, color='darkgreen', label='Z=0.02')

    plt.ylabel('Returned Stellar Mass Fraction')
    plt.xticks(index + bar_width + 0.25, ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe'))

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.1, 0.94, 'Yields SNII', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.axis([-0.5, 13.5, 1e-5, 1])
    plt.yscale('log')
    plt.legend(loc=[0.5, 0.6], labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=1, fontsize=10)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.tight_layout()
    plt.savefig('./figures/SNIIYield_tables.png', dpi=300)


def read_AGB_COLIBRE(metallicity_flag, metallicity):

    # Elements:
    # ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium',
    # 'Silicon', 'Sulphur', 'Calcium', 'Iron', 'Strontium', 'Barium']
    indx = np.array([2, 3, 4, 5, 6, 7, 10, 11, 12])

    total_mass_fraction = 0.0133714

    mass_fraction_elements = np.array([
        0.0023647215,  # C
        0.0006928991,  # N
        0.00573271036,  # O
        0.00125649278,  # Ne
        0.00070797838,  # Mg
        0.00066495154,  # Si
        0.00129199252,
        0, 0])  # Fe



    # Write data to HDF5
    with h5py.File('../data/AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Yield_Z0p007 = data_file["/Yields/"+metallicity_flag+"/Yield"][:][:]
        Ejected_mass_Z0p007 = data_file["/Yields/"+metallicity_flag+"/Ejected_mass"][:]

    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction_elements)
    stellar_yields = np.zeros((num_elements, num_mass_bins))

    for i in range(num_mass_bins):
        stellar_yields[:, i] = Yield_Z0p007[indx, i] + (metallicity / total_mass_fraction) * mass_fraction_elements * Ejected_mass_Z0p007[i]

    new_mass_range = np.arange(1, 8.1, 0.1)
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

def plot_AGB():

    IMF_int = integrate_IMF(0.1, 100)

    with h5py.File('../data/AGB.hdf5', 'r') as data_file:
        Z_ind = [x.decode() for x in data_file['Yield_names']]
        Z = data_file["Metallicities"][:]
        num_Z = len(Z)

    colibre_AGB_yields = np.zeros((9,num_Z))
    for j in range(len(Z_ind)):
        print(Z_ind[j],Z[j])
        colibre_AGB_yields[:,j] = read_AGB_COLIBRE(Z_ind[j],Z[j]) / IMF_int


    # colibre_Z0a = read_AGB_COLIBRE('0.0001',0.0001) / IMF_int
    # colibre_Z0b = read_AGB_COLIBRE('0.001',0.001) / IMF_int
    # colibre_Z0 = read_AGB_COLIBRE('0.007',0.007) / IMF_int
    # colibre_Z1 = read_AGB_COLIBRE('0.014',0.014) / IMF_int
    # colibre_Z2 = read_AGB_COLIBRE('0.03',0.03) / IMF_int
    # colibre_Z3 = read_AGB_COLIBRE('0.06',0.06) / IMF_int
    # colibre_Z4 = read_AGB_COLIBRE('0.10',0.1) / IMF_int

    color_list = ['grey','darkgreen','yellowgreen','darkblue','tab:blue','lightblue',
                  'pink','purple','tab:orange','darkkhaki','tab:red','black','maroon']

    label_list = ['Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Iron', r'$[\times 10^{6}]$ Strontium', r'$[\times 10^{6}]$ Barium']

    ####
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(linestyle='-',linewidth=0.3)

    for i in range(9):
        if i <= 6:
            plt.plot(Z, colibre_AGB_yields[i,:],'-o',color=color_list[i],label=label_list[i])
        if i == 7:
            plt.plot(Z, colibre_AGB_yields[i,:] * 1e6,'--o',color=color_list[i],label=label_list[i])
        if i == 8:
            plt.plot(Z, colibre_AGB_yields[i,:] * 1e6,'--o',color=color_list[i],label=label_list[i])

    plt.ylabel('Returned Stellar Mass Fraction')
    plt.xlabel('Metallicity')

    #props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    #ax.text(0.05, 0.94, 'Yields AGB', transform=ax.transAxes,
    #        fontsize=12, verticalalignment='top', bbox=props)

    plt.axis([8e-5, 0.1, 1e-8, 0.15])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=3, fontsize=10)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('../figures/AGBYield_tables_metallicity.png', dpi=300)

    #
    # plt.figure()
    # ax = plt.subplot(1, 1, 1)
    # ax.grid(True)
    # plt.grid(which='major', linestyle='-',linewidth=0.3)
    # plt.grid(which='minor', linestyle=':',linewidth=0.3)
    #
    # index = np.arange(9) * 2.0
    # bar_width = 0.2
    # opacity = 0.8
    # plt.bar(index, colibre_Z0a, bar_width, alpha=1, color='pink', label='Z=0.0001')
    # plt.bar(index + bar_width, colibre_Z0b, bar_width, alpha=1, color='khaki', label='Z=0.001')
    # plt.bar(index + 2 * bar_width, colibre_Z0, bar_width, alpha=1, color='darkblue', label='Z=0.007')
    # plt.bar(index + 3 * bar_width, colibre_Z1, bar_width, alpha=opacity, color='tab:blue', label='Z=0.014')
    # plt.bar(index + 4 * bar_width, colibre_Z2, bar_width, alpha=1, color='tab:orange', label='Z=0.03')
    # plt.bar(index + 5 * bar_width, colibre_Z3, bar_width, alpha=opacity, color='lightgreen', label='Z=0.06')
    # plt.bar(index + 6 * bar_width, colibre_Z4, bar_width, alpha=opacity, color='darkgreen', label='Z=0.1')
    #
    # plt.ylabel('Returned Stellar Mass Fraction')
    # plt.xticks(index + bar_width + 0.25, ('C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe', 'Ba', 'Sr'))
    #
    # props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    # ax.text(0.1, 0.94, 'Yields AGB', transform=ax.transAxes,
    #         fontsize=12, verticalalignment='top', bbox=props)
    #
    # plt.tight_layout()
    # plt.axis([-0.5, 18, 1e-12, 10])
    # plt.yscale('log')
    # plt.legend(loc=[0.5, 0.68], labelspacing=0.2, handlelength=0.8,
    #            handletextpad=0.3, frameon=False, columnspacing=0.4,
    #            ncol=2, fontsize=10)
    # ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.tight_layout()
    # plt.savefig('../figures/AGBYield_tables.png', dpi=300)


if __name__ == "__main__":

    # plot_SNII()
    plot_AGB()
