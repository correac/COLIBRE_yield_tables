
import h5py
import numpy as np

from AGB_tables import make_AGB_tables
from SNIa_tables import make_SNIa_tables
from CCSN_tables import make_CCSN_tables

from pylab import *
import matplotlib.pyplot as plt

# Plot parameters
params = {
    "font.size": 11,
    "font.family":"Times",
    "text.usetex": True,
    "figure.figsize": (4, 3),
    "figure.subplot.left": 0.18,
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
    with h5py.File('./data/AGB.hdf5', 'r') as data_file:
        Masses_Karakas = data_file["Masses"][:]
        Y_Z0007 = data_file["/Yields/Z_0.007/Yield"][:][:]
        Y_Z0014 = data_file["/Yields/Z_0.014/Yield"][:][:]
        Y_Z003 = data_file["/Yields/Z_0.03/Yield"][:][:]
        Y_Z004 = data_file["/Yields/Z_0.04/Yield"][:][:]
        Y_Z005 = data_file["/Yields/Z_0.05/Yield"][:][:]
        Y_Z006 = data_file["/Yields/Z_0.06/Yield"][:][:]
        Y_Z007 = data_file["/Yields/Z_0.07/Yield"][:][:]
        Y_Z008 = data_file["/Yields/Z_0.08/Yield"][:][:]
        Y_Z009 = data_file["/Yields/Z_0.09/Yield"][:][:]
        Y_Z010 = data_file["/Yields/Z_0.10/Yield"][:][:]

    elements = np.array(['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Magnesium','Iron','Strontium','Barium'])
    indx = np.array([0, 1, 2, 3, 4, 6, 10, 11, 12])

    for i, elem in enumerate(elements):
        plt.figure()

        ax = plt.subplot(1, 1, 1)
        ax.grid(True)

        plt.plot(Masses_Karakas, Y_Z0007[indx[i], :], '--o', label='$0.007Z_{\odot}$', color='tab:orange')
        plt.plot(Masses_Karakas, Y_Z0014[indx[i], :], '--o', label='$0.014Z_{\odot}$', color='tab:blue')
        plt.plot(Masses_Karakas, Y_Z003[indx[i], :], '--o', label='$0.03Z_{\odot}$', color='crimson')
        plt.plot(Masses_Karakas, Y_Z004[indx[i], :], '-o', label='$0.04Z_{\odot}$', color='lightgreen')
        plt.plot(Masses_Karakas, Y_Z005[indx[i], :], '-o', label='$0.05Z_{\odot}$', color='darkblue')
        plt.plot(Masses_Karakas, Y_Z006[indx[i], :], '-o', label='$0.06Z_{\odot}$', color='darksalmon')
        plt.plot(Masses_Karakas, Y_Z007[indx[i], :], '-o', label='$0.07Z_{\odot}$', color='tab:green')
        plt.plot(Masses_Karakas, Y_Z008[indx[i], :], '-o', label='$0.08Z_{\odot}$', color='tab:purple')
        plt.plot(Masses_Karakas, Y_Z009[indx[i], :], '-o', label='$0.09Z_{\odot}$', color='pink')
        plt.plot(Masses_Karakas, Y_Z010[indx[i], :], '-o', label='$0.10Z_{\odot}$', color='grey')

        if indx[i] >= 11: plt.yscale('log')
        plt.xlim(1, 12)
        plt.ylabel('Net Yields '+elem+' [M$_{\odot}$]')
        plt.xlabel('Initial stellar mass [M$_{\odot}$]')
        plt.legend(loc=[0.5, 0.7], labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False, columnspacing=0.4,
                   ncol=2, fontsize=8)
        ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
        plt.savefig('./figures/AGB_Yield_tables_'+elem+'.png', dpi=200)


def plot_SNIa_yield_tables():

    # Read data
    with h5py.File('./data/SNIa_Kobayashi2020.hdf5', 'r') as data_file:
        Y = data_file["/Yield"][:]
        Sp_Y = data_file["/Species_names"][:]

    # Write data to HDF5
    with h5py.File('./data/EAGLE_yieldtables/SNIa.hdf5', 'r') as data_file:
        Z = data_file["/Yield"][:]
        Sp_Z = data_file["/Species_names"][:]

    Z = np.array([Z[0], Z[1], Z[5], Z[6], Z[7], Z[9], Z[11], Z[13], Z[25]])

    plt.figure()

    ax = plt.subplot(1, 1, 1)
    ax.grid(True)

    plt.plot(np.arange(len(Y)), Y, 'o', color='tab:blue', label='Kobayashi et al. (2020)')
    plt.plot(np.arange(len(Z)), Z, 'o', color='tab:orange', label='Thielemann et al. (2013)')

    plt.yscale('log')
    plt.ylabel('Yields [M$_{\odot}$]')
    plt.xlabel('Elements')

    y_major = matplotlib.ticker.LogLocator(base=10.0, numticks=10)
    ax.yaxis.set_major_locator(y_major)
    y_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.grid(b=True, which='major', color='grey', linestyle='-', lw=0.2, zorder=0)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    plt.legend(labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=True, columnspacing=0.4, ncol=1,
               fontsize=8)

    plt.axis([0, 8.4, 1e-7, 1e0])
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"])

    plt.savefig('./figures/SNIa_yields.png', dpi=200)

def plot_CCSN_yield_tables():

    # Read data
    with h5py.File('./data/SNII_Nomoto2013.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Z000_mej_ccsn = data_file["/Yields/Z_0.004/Ejected_mass_in_ccsn"][:][:]
        Z000_mej_wind = data_file["/Yields/Z_0.004/Ejected_mass_in_winds"][:]
        Z008_mej_ccsn = data_file["/Yields/Z_0.008/Ejected_mass_in_ccsn"][:][:]
        Z008_mej_wind = data_file["/Yields/Z_0.008/Ejected_mass_in_winds"][:]
        Z050_mej_ccsn = data_file["/Yields/Z_0.020/Ejected_mass_in_ccsn"][:][:]
        Z050_mej_wind = data_file["/Yields/Z_0.020/Ejected_mass_in_winds"][:]
        Z000_mass_loss = data_file["/Yields/Z_0.004/Total_Mass_ejected"][:]
        Z008_mass_loss = data_file["/Yields/Z_0.008/Total_Mass_ejected"][:]
        Z050_mass_loss = data_file["/Yields/Z_0.020/Total_Mass_ejected"][:]

    with h5py.File('./data/EAGLE_yieldtables/SNII.hdf5', 'r') as data_file:
        Masses_Marigo = data_file["Masses"][:]
        Y_Z0007_Marigo = data_file["/Yields/Z_0.004/Yield"][:][:]
        Y_Z0014_Marigo = data_file["/Yields/Z_0.008/Yield"][:][:]
        Y_Z003_Marigo = data_file["/Yields/Z_0.05/Yield"][:][:]

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
    Z008_Net_Yields = np.zeros((len(Z000_mej_ccsn[:, 0]), len(Z000_mej_ccsn[0, :])))
    Z050_Net_Yields = np.zeros((len(Z000_mej_ccsn[:, 0]), len(Z000_mej_ccsn[0, :])))

    for i in range(num_species):
        Z000_Net_Yields[i, :] = Z000_mej_ccsn[i, :] + Z000_mej_wind * init_abundance[i] - init_abundance[
            i] * Z000_mass_loss[:]
        Z008_Net_Yields[i, :] = Z008_mej_ccsn[i, :] + Z008_mej_wind * init_abundance[i] - init_abundance[
            i] * Z008_mass_loss[:]
        Z050_Net_Yields[i, :] = Z050_mej_ccsn[i, :] + Z050_mej_wind * init_abundance[i] - init_abundance[
            i] * Z050_mass_loss[:]

    ##################################################
    for i in range(num_species):

        plt.figure()

        ax = plt.subplot(1, 1, 1)
        ax.grid(True)

        plt.plot(Masses, Z000_Net_Yields[i, :], '-o', label='$0.004Z_{\odot}$ (Kobayashi+ 06)', color='tab:orange')
        plt.plot(Masses, Z008_Net_Yields[i, :], '-o', label='$0.008Z_{\odot}$ (Kobayashi+ 06)', color='tab:red')
        plt.plot(Masses, Z050_Net_Yields[i, :], '-o', label='$0.020Z_{\odot}$ (Kobayashi+ 06)', color='tab:blue')

        plt.plot(Masses_Marigo, Y_Z0007_Marigo[i, :], '--o', label='$0.004Z_{\odot}$ (Portinari+ 98)',
                 color='tab:green')
        plt.plot(Masses_Marigo, Y_Z0014_Marigo[i, :], '--o', label='$0.008Z_{\odot}$ (Portinari+ 98)', color='darkblue')
        plt.plot(Masses_Marigo, Y_Z003_Marigo[i, :], '--o', label='$0.019Z_{\odot}$ (Portinari+ 98)',
                 color='tab:purple')

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
        plt.xlim([1, 50])
        plt.legend(loc='upper left', labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
                   columnspacing=0.4, ncol=2, fontsize=8)
        ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
        plt.savefig('./figures/CCSN_'+file_name, dpi=200)

if __name__ == "__main__":

    # Make and combine AGB yield tables
    make_AGB_tables()
    make_SNIa_tables()
    make_CCSN_tables()

    plot_AGB_yield_tables()

    # Make SNIa yield tables
    make_SNIa_tables()
    plot_SNIa_yield_tables()

    # Make CCSN yield tables
    make_CCSN_tables()
    plot_CCSN_yield_tables()