import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

# Plot parameters
params = {
    "font.size": 11,
    "font.family": "Times",
    "text.usetex": True,
    "figure.figsize": (4, 2.8),
    "figure.subplot.left": 0.15,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.top": 0.96,
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

total_mass_fraction = 0.0133714

def read_AGB_COLIBRE():

    #['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron']
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])


    # Write data to HDF5
    with h5py.File('../data/AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Yield_Z0p0001 = data_file["/Yields/Z_0.0001/Yield"][:][:]
        Ejected_mass_Z0p0001 = data_file["/Yields/Z_0.0001/Ejected_mass"][:]
        Yield_Z0p001 = data_file["/Yields/Z_0.001/Yield"][:][:]
        Ejected_mass_Z0p001 = data_file["/Yields/Z_0.001/Ejected_mass"][:]
        Yield_Z0p004 = data_file["/Yields/Z_0.004/Yield"][:][:]
        Ejected_mass_Z0p004 = data_file["/Yields/Z_0.004/Ejected_mass"][:]
        Yield_Z0p007 = data_file["/Yields/Z_0.007/Yield"][:][:]
        Ejected_mass_Z0p007 = data_file["/Yields/Z_0.007/Ejected_mass"][:]
        Yield_Z0p014 = data_file["/Yields/Z_0.014/Yield"][:][:]
        Ejected_mass_Z0p014 = data_file["/Yields/Z_0.014/Ejected_mass"][:]
        Yield_Z0p03 = data_file["/Yields/Z_0.03/Yield"][:][:]
        Ejected_mass_Z0p03 = data_file["/Yields/Z_0.03/Ejected_mass"][:]

    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    num_metals = 6
    stellar_yields = np.zeros((num_elements, num_mass_bins, num_metals))
    total_yields = np.zeros(num_mass_bins)

    for i in range(num_mass_bins):
        stellar_yields[:, i, 0] = Yield_Z0p0001[indx, i] + (0.0001 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p0001[i]
        stellar_yields[:, i, 1] = Yield_Z0p001[indx, i] + (0.001 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p001[i]
        stellar_yields[:, i, 2] = Yield_Z0p004[indx, i] + (0.004 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p004[i]
        stellar_yields[:, i, 3] = Yield_Z0p007[indx, i] + (0.007 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p007[i]
        stellar_yields[:, i, 4] = Yield_Z0p014[indx, i] + (0.014 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p014[i]
        stellar_yields[:, i, 5] = Yield_Z0p03[indx, i] + (0.03 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p03[i]


    return Masses, stellar_yields

def plot_AGB():

    colibre_masses, colibre_yields = read_AGB_COLIBRE()

    carbon = 2
    nitrogen = 3
    oxygen = 4

    ######
    plt.figure()

    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(linestyle='-',linewidth=0.3)

    plt.plot(colibre_masses, np.log10(colibre_yields[nitrogen,:,0] / colibre_yields[oxygen,:,0]),'-',color='black',label='$Z=0.0001$')
    plt.plot(colibre_masses, np.log10(colibre_yields[nitrogen,:,1] / colibre_yields[oxygen,:,1]),'-',color='tab:green',label='$Z=0.001$')
    plt.plot(colibre_masses, np.log10(colibre_yields[nitrogen,:,2] / colibre_yields[oxygen,:,2]),'-',color='lightgreen',label='$Z=0.004$')
    plt.plot(colibre_masses, np.log10(colibre_yields[nitrogen,:,3] / colibre_yields[oxygen,:,3]),'-',color='darkblue',label='$Z=0.007$')
    plt.plot(colibre_masses, np.log10(colibre_yields[nitrogen,:,4] / colibre_yields[oxygen,:,4]),'-',color='tab:blue',label='$Z=0.014$')
    plt.plot(colibre_masses, np.log10(colibre_yields[nitrogen,:,5] / colibre_yields[oxygen,:,5]),'-',color='tab:orange',label='$Z=0.03$')

    plt.ylabel('$\log_{10}$(N/O)')
    plt.xlabel('Stellar Mass [M$_{\odot}$]')
    plt.axis([0.5, 10, -1, 2])

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.7, 0.93, 'Yields AGB', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.legend(loc=[0.02, 0.55], labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=1, fontsize=10)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('test_AGB.png', dpi=300)


def read_SNII_COLIBRE(metallicity_flag, metallicity):

    total_mass_fraction = 0.0133714

    #['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron'
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])

    factor = metallicity / total_mass_fraction

    # Write data to HDF5
    with h5py.File('../data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Ejected_mass_winds = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_ccsn"][:][:]
        Total_metals = data_file["/Yields/Z_"+metallicity_flag+"/Total_Mass_ejected"][:]

    select = np.where(Masses >= 8)[0]
    Masses = Masses[select]
    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    stellar_yields = np.zeros((num_elements, num_mass_bins))

    for i in range(num_mass_bins):
        stellar_yields[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]
        stellar_yields[:, i] = np.clip(stellar_yields[:, i], 1e-20, None)

    return Masses, stellar_yields

def plot_CCSN():

    total_mass_fraction = 0.0133714

    masses, colibre_Z0 = read_SNII_COLIBRE('0.000',0.000)
    _, colibre_Z1 = read_SNII_COLIBRE('0.001',0.001)
    _, colibre_Z2 = read_SNII_COLIBRE('0.004',0.004)
    _, colibre_Z3 = read_SNII_COLIBRE('0.008',0.008)
    _, colibre_Z4 = read_SNII_COLIBRE('0.020',0.020)

    carbon = 2
    nitrogen = 3
    oxygen = 4

    ####

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-',linewidth=0.3)
    plt.grid(which='minor', linestyle=':',linewidth=0.3)

    with_oxygen = np.where(colibre_Z0[oxygen,:] > 1e-10)[0]
    plt.plot(masses[with_oxygen], np.log10(colibre_Z0[nitrogen,with_oxygen]/colibre_Z0[oxygen,with_oxygen]),'-',color='darkgreen',label='Z=0.0')
    plt.plot(masses, np.log10(colibre_Z1[nitrogen,:]/colibre_Z1[oxygen,:]),'-',color='lightgreen',label='Z=0.001')
    plt.plot(masses, np.log10(colibre_Z2[nitrogen,:]/colibre_Z2[oxygen,:]),'-',color='darkblue',label='Z=0.004')
    plt.plot(masses, np.log10(colibre_Z3[nitrogen,:]/colibre_Z3[oxygen,:]),'-',color='tab:blue',label='Z=0.008')
    plt.plot(masses, np.log10(colibre_Z4[nitrogen,:]/colibre_Z4[oxygen,:]),'-',color='tab:orange',label='Z=0.02')

    plt.ylabel('$\log_{10}$(N/O)')
    plt.xlabel('Stellar Mass [M$_{\odot}$]')
    plt.axis([8, 40, -6, 2])

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.6, 0.93, 'Yields CCSN', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.legend(loc=[0.02, 0.74], labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=2, fontsize=10)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    plt.savefig('./test_CCSN.png', dpi=300)

if __name__ == "__main__":

    #plot_AGB()
    plot_CCSN()