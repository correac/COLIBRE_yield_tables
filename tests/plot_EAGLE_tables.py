import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

# Plot parameters
params = {
    "font.size": 11,
    "font.family":"Times",
    "text.usetex": True,
    "figure.figsize": (4, 3),
    "figure.subplot.left": 0.22,
    "figure.subplot.right": 0.95,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.top": 0.93,
    "figure.subplot.wspace": 0.25,
    "figure.subplot.hspace": 0.25,
    "lines.markersize": 3,
    "lines.linewidth": 1,
    "figure.max_open_warning": 0,
}
rcParams.update(params)


def plot_AGB_yield_tables():

    # Write data to HDF5
    with h5py.File('./data/EAGLE_yieldtables/AGB.hdf5', 'r') as data_file:
        Masses_Karakas = data_file["Masses"][:]
        Y_Z0004 = data_file["/Yields/Z_0.004/Yield"][:][:]
        Y_Z0008 = data_file["/Yields/Z_0.008/Yield"][:][:]
        Y_Z0019 = data_file["/Yields/Z_0.019/Yield"][:][:]

    elements = np.array(['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Magnesium','Iron'])
    indx = np.array([0, 1, 2, 3, 4, 6, 10])

    for i, elem in enumerate(elements):
        plt.figure()

        ax = plt.subplot(1, 1, 1)
        ax.grid(True)

        plt.plot(Masses_Karakas, Y_Z0004[indx[i], :], '-o', label='$0.004Z_{\odot}$', color='tab:orange')
        plt.plot(Masses_Karakas, Y_Z0008[indx[i], :], '-o', label='$0.008Z_{\odot}$', color='tab:blue')
        plt.plot(Masses_Karakas, Y_Z0019[indx[i], :], '-o', label='$0.019Z_{\odot}$', color='crimson')

        if indx[i] >= 11: plt.yscale('log')
        plt.xlim(1, 12)
        plt.ylabel('Net Yields '+elem+' [M$_{\odot}$]')
        plt.xlabel('Initial stellar mass [M$_{\odot}$]')
        plt.legend(loc=[0.5, 0.7], labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False, columnspacing=0.4,
                   ncol=2, fontsize=8)
        ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
        plt.savefig('./figures/EAGLE_AGB_Yield_tables_'+elem+'.png', dpi=200)

def factor(index, metallicity):

    init_abundance_Hydrogen = 0.73738788833  # Initial fraction of particle mass in Hydrogen
    init_abundance_Helium = 0.24924186942  # Initial fraction of particle mass in Helium
    total_mass_fraction = 0.0133714

    factor_Z = metallicity / total_mass_fraction
    factor_XY = (1. - factor_Z * total_mass_fraction) / (init_abundance_Hydrogen + init_abundance_Helium)

    if index <= 1:
        factor = factor_XY
    else:
        factor = factor_Z

    return factor


def plot_COLIBRE_SNII_yield_tables():

    # Read data
    with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Z000_mej_ccsn = data_file["/Yields/Z_0.000/Ejected_mass_in_ccsn"][:][:]
        Z000_mej_wind = data_file["/Yields/Z_0.000/Ejected_mass_in_winds"][:]
        Z001_mej_ccsn = data_file["/Yields/Z_0.001/Ejected_mass_in_ccsn"][:][:]
        Z001_mej_wind = data_file["/Yields/Z_0.001/Ejected_mass_in_winds"][:]
        Z004_mej_ccsn = data_file["/Yields/Z_0.004/Ejected_mass_in_ccsn"][:][:]
        Z004_mej_wind = data_file["/Yields/Z_0.004/Ejected_mass_in_winds"][:]
        Z008_mej_ccsn = data_file["/Yields/Z_0.008/Ejected_mass_in_ccsn"][:][:]
        Z008_mej_wind = data_file["/Yields/Z_0.008/Ejected_mass_in_winds"][:]
        Z020_mej_ccsn = data_file["/Yields/Z_0.020/Ejected_mass_in_ccsn"][:][:]
        Z020_mej_wind = data_file["/Yields/Z_0.020/Ejected_mass_in_winds"][:]
        Z050_mej_ccsn = data_file["/Yields/Z_0.050/Ejected_mass_in_ccsn"][:][:]
        Z050_mej_wind = data_file["/Yields/Z_0.050/Ejected_mass_in_winds"][:]
        Z000_mass_loss = data_file["/Yields/Z_0.000/Total_Mass_ejected"][:]
        Z001_mass_loss = data_file["/Yields/Z_0.001/Total_Mass_ejected"][:]
        Z004_mass_loss = data_file["/Yields/Z_0.004/Total_Mass_ejected"][:]
        Z008_mass_loss = data_file["/Yields/Z_0.008/Total_Mass_ejected"][:]
        Z020_mass_loss = data_file["/Yields/Z_0.020/Total_Mass_ejected"][:]
        Z050_mass_loss = data_file["/Yields/Z_0.050/Total_Mass_ejected"][:]

    # solar abundance values
    init_abundance_Hydrogen = 0.73738788833  # Initial fraction of particle mass in Hydrogen
    init_abundance_Helium = 0.24924186942  # Initial fraction of particle mass in Helium
    init_abundance_Carbon = 0.0023647215  # Initial fraction of particle mass in Carbon
    init_abundance_Nitrogen = 0.0006928991  # Initial fraction of particle mass in Nitrogen
    init_abundance_Oxygen = 0.00573271036  # Initial fraction of particle mass in Oxygen
    init_abundance_Neon = 0.00125649278  # Initial fraction of particle mass in Neon
    init_abundance_Magnesium = 0.00070797838  # Initial fraction of particle mass in Magnesium
    init_abundance_Silicon = 0.00066495154  # Initial fraction of particle mass in Silicon
    init_abundance_Iron = 0.00129199252  # Initial fraction of particle mass in Iron
    total_mass_fraction = 0.0133714

    init_abundance = np.array([init_abundance_Hydrogen,
                               init_abundance_Helium,
                               init_abundance_Carbon,
                               init_abundance_Nitrogen,
                               init_abundance_Oxygen,
                               init_abundance_Neon,
                               init_abundance_Magnesium,
                               init_abundance_Silicon,
                               init_abundance_Iron])

    num_species = 9

    Z000_Net_Yields = np.zeros((len(Z000_mej_ccsn[:, 0]), len(Z000_mej_ccsn[0, :])))
    Z001_Net_Yields = np.zeros((len(Z001_mej_ccsn[:, 0]), len(Z001_mej_ccsn[0, :])))
    Z004_Net_Yields = np.zeros((len(Z004_mej_ccsn[:, 0]), len(Z004_mej_ccsn[0, :])))
    Z008_Net_Yields = np.zeros((len(Z008_mej_ccsn[:, 0]), len(Z008_mej_ccsn[0, :])))
    Z020_Net_Yields = np.zeros((len(Z000_mej_ccsn[:, 0]), len(Z000_mej_ccsn[0, :])))
    Z050_Net_Yields = np.zeros((len(Z000_mej_ccsn[:, 0]), len(Z000_mej_ccsn[0, :])))

    for i in range(num_species):
        Z000_Net_Yields[i, :] = Z000_mej_ccsn[i, :] + Z000_mej_wind * init_abundance[i] * factor(i, 0) - init_abundance[i] * factor(i,0) * Z000_mass_loss[:]
        Z001_Net_Yields[i, :] = Z001_mej_ccsn[i, :] + Z001_mej_wind * init_abundance[i] * factor(i, 0.001) - init_abundance[i] * factor(i, 0.001) * Z001_mass_loss[:]
        Z004_Net_Yields[i, :] = Z004_mej_ccsn[i, :] + Z004_mej_wind * init_abundance[i] * factor(i, 0.004) - init_abundance[i] * factor(i, 0.004) * Z004_mass_loss[:]
        Z008_Net_Yields[i, :] = Z008_mej_ccsn[i, :] + Z008_mej_wind * init_abundance[i] * factor(i, 0.008) - init_abundance[i] * factor(i, 0.008) * Z008_mass_loss[:]
        Z020_Net_Yields[i, :] = Z020_mej_ccsn[i, :] + Z020_mej_wind * init_abundance[i] * factor(i, 0.02) - init_abundance[i] * factor(i, 0.02) * Z020_mass_loss[:]
        Z050_Net_Yields[i, :] = Z050_mej_ccsn[i, :] + Z050_mej_wind * init_abundance[i] * factor(i, 0.05) - init_abundance[i] * factor(i, 0.05) * Z050_mass_loss[:]

    ##################################################
    for i in range(num_species):

        plt.figure()

        ax = plt.subplot(1, 1, 1)
        ax.grid(True)

        plt.plot(Masses, Z000_Net_Yields[i, :], '-o', label='$0.000Z_{\odot}$', color='tab:orange')
        plt.plot(Masses, Z001_Net_Yields[i, :], '-o', label='$0.001Z_{\odot}$', color='tab:green')
        plt.plot(Masses, Z004_Net_Yields[i, :], '-o', label='$0.004Z_{\odot}$', color='tab:purple')
        plt.plot(Masses, Z008_Net_Yields[i, :], '-o', label='$0.008Z_{\odot}$', color='tab:red')
        plt.plot(Masses, Z020_Net_Yields[i, :], '-o', label='$0.020Z_{\odot}$', color='grey')
        plt.plot(Masses, Z050_Net_Yields[i, :], '-o', label='$0.050Z_{\odot}$', color='tab:blue')


        if i == 0:
            plt.ylabel('Net Yields Hydrogen [M$_{\odot}$]')
            file_name = 'Yield_tables_Hydrogen.png'
            plt.ylim([-15, 5])
        if i == 1:
            plt.ylabel('Net Yields Helium [M$_{\odot}$]')
            file_name = 'Yield_tables_Helium.png'
            plt.yscale('log')
        if i == 2:
            plt.ylabel('Net Yields Carbon [M$_{\odot}$]')
            file_name = 'Yield_tables_Carbon.png'
            plt.yscale('log')
        if i == 3:
            plt.ylabel('Net Yields Nitrogen [M$_{\odot}$]')
            file_name = 'Yield_tables_Nitrogen.png'
            plt.yscale('log')
        if i == 4:
            plt.ylabel('Net Yields Oxygen [M$_{\odot}$]')
            file_name = 'Yield_tables_Oxygen.png'
            plt.yscale('log')
            plt.ylim([1e-2, 1e2])
        if i == 5:
            plt.ylabel('Net Yields Neon [M$_{\odot}$]')
            file_name = 'Yield_tables_Neon.png'
            plt.yscale('log')
            plt.ylim([1e-3, 1e1])
        if i == 6:
            plt.ylabel('Net Yields Magnesium [M$_{\odot}$]')
            file_name = 'Yield_tables_Magnesium.png'
            plt.yscale('log')
            plt.ylim([1e-3, 1e1])
        if i == 7:
            plt.ylabel('Net Yields Silicon [M$_{\odot}$]')
            file_name = 'Yield_tables_Silicon.png'
            plt.yscale('log')
            plt.ylim([1e-2, 1e1])
        if i == 8:
            plt.ylabel('Net Yields Iron [M$_{\odot}$]')
            file_name = 'Yield_tables_Iron.png'
            plt.ylim([-0.1, 0.3])
            # plt.yscale('log')

        plt.xlabel('Initial stellar mass [M$_{\odot}$]')
        # plt.axis([1,50,1e-2,1e1])
        plt.xlim([6, 100])
        plt.legend(loc='upper left', labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
                   columnspacing=0.4, ncol=2, fontsize=7)
        ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
        plt.savefig('./figures/COLIBRE_CCSN_'+file_name, dpi=300)
    
if __name__ == "__main__":

    # Make and combine AGB yield tables
    plot_AGB_yield_tables()

    plot_COLIBRE_SNII_yield_tables()