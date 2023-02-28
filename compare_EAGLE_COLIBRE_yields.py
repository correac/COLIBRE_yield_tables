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
    with h5py.File('./data/AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Yield_Z0p007 = data_file["/Yields/Z_0.007/Yield"][:][:]
        Ejected_mass_Z0p007 = data_file["/Yields/Z_0.007/Ejected_mass"][:]
        Yield_Z0p014 = data_file["/Yields/Z_0.014/Yield"][:][:]
        Ejected_mass_Z0p014 = data_file["/Yields/Z_0.014/Ejected_mass"][:]
        Yield_Z0p03 = data_file["/Yields/Z_0.03/Yield"][:][:]
        Ejected_mass_Z0p03 = data_file["/Yields/Z_0.03/Ejected_mass"][:]

    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    num_metals = 3
    stellar_yields = np.zeros((num_elements, num_mass_bins, num_metals))
    total_yields = np.zeros(num_mass_bins)

    for i in range(num_mass_bins):
        stellar_yields[:, i, 0] = Yield_Z0p007[indx, i] + (0.007 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p007[i]
        stellar_yields[:, i, 1] = Yield_Z0p014[indx, i] + (0.014 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p014[i]
        stellar_yields[:, i, 2] = Yield_Z0p03[indx, i] + (0.03 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p03[i]


    return Masses, stellar_yields

def read_AGB_EAGLE():


    #['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron']
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])


    # Write data to HDF5
    with h5py.File('./data/EAGLE_yieldtables/AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Yield_Z0p004 = data_file["/Yields/Z_0.004/Yield"][:][:]
        Ejected_mass_Z0p004 = data_file["/Yields/Z_0.004/Ejected_mass"][:]
        Yield_Z0p008 = data_file["/Yields/Z_0.008/Yield"][:][:]
        Ejected_mass_Z0p008 = data_file["/Yields/Z_0.008/Ejected_mass"][:]
        Yield_Z0p019 = data_file["/Yields/Z_0.019/Yield"][:][:]
        Ejected_mass_Z0p019 = data_file["/Yields/Z_0.019/Ejected_mass"][:]

    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    num_metals = 3
    stellar_yields = np.zeros((num_elements, num_mass_bins, num_metals))
    total_yields = np.zeros(num_mass_bins)

    for i in range(num_mass_bins):
        stellar_yields[:, i, 0] = Yield_Z0p004[indx, i] + (0.004 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p004[i]
        stellar_yields[:, i, 1] = Yield_Z0p008[indx, i] + (0.008 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p008[i]
        stellar_yields[:, i, 2] = Yield_Z0p019[indx, i] + (0.019 / total_mass_fraction) * mass_fraction * Ejected_mass_Z0p019[i]


    return Masses, stellar_yields

def plot_AGB():

    colibre_masses, colibre_yields = read_AGB_COLIBRE()
    eagle_masses, eagle_yields = read_AGB_EAGLE()

    flag = 2
    ######
    plt.figure()

    ax = plt.subplot(1, 1, 1)
    ax.grid(True)
    plt.grid(linestyle='-',linewidth=0.3)

    plt.plot(colibre_masses, colibre_yields[flag,:,0],'-',color='black',label='Colibre')
    plt.plot(colibre_masses, colibre_yields[flag,:,0],'--',color='black',label='Eagle')
    plt.plot(colibre_masses, colibre_yields[flag,:,0],'--',color='white',label=' ')

    plt.plot(colibre_masses, colibre_yields[flag,:,0],'-',color='darkblue',label='$Z=0.007$')
    plt.plot(colibre_masses, colibre_yields[flag,:,1],'-',color='tab:blue',label='$Z=0.014$')
    plt.plot(colibre_masses, colibre_yields[flag,:,2],'-',color='tab:orange',label='$Z=0.03$')

    plt.plot(eagle_masses, eagle_yields[flag,:,0],'--',color='darkblue',label='$Z=0.004$')
    plt.plot(eagle_masses, eagle_yields[flag,:,1],'--',color='tab:blue',label='$Z=0.008$')
    plt.plot(eagle_masses, eagle_yields[flag,:,2],'--',color='tab:orange',label='$Z=0.019$')

    if flag == 0:
        ylabel = 'Hydrogen Mass Returned to ISM'
        filename = './figures/Comparison_Hydrogen_EAGLE_COLIBRE_AGB_Yield_tables.png'
    if flag == 1:
        ylabel = 'Helium Mass Returned to ISM'
        filename = './figures/Comparison_Helium_EAGLE_COLIBRE_AGB_Yield_tables.png'
    if flag == 2:
        ylabel = 'Carbon Mass Returned to ISM'
        filename = './figures/Comparison_Carbon_EAGLE_COLIBRE_AGB_Yield_tables.png'
        plt.yscale('log')


    plt.ylabel(ylabel)
    plt.xlabel('Stellar Mass [M$_{\odot}$]')
    plt.axis([0, 10, 0, 15])

    props = dict(boxstyle='round', fc='grey', ec='white', alpha=0.2)
    ax.text(0.04, 0.7, 'Yields AGB', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.legend(loc=[0.04, 0.73], labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=3, fontsize=10)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(filename, dpi=300)



if __name__ == "__main__":

    plot_AGB()
