import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

# Plot parameters
params = {
    "font.size": 11,
    "font.family": "Times",
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

def read_AGB_data(Z_ind):

    with h5py.File('./data/AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        yields = data_file["/Yields/"+Z_ind+"/Yield"][:][:]

    return Masses, yields

def plot_AGB_tables():

    with h5py.File('./data/AGB.hdf5', 'r') as data_file:
        elements = [x.decode() for x in data_file['Species_names']]
        Z_ind = [x.decode() for x in data_file['Yield_names']]

    color_list = ['grey','darkgreen','yellowgreen','darkblue','tab:blue','lightblue',
                  'pink','purple','tab:orange','darkkhaki','tab:red','black','maroon']

    # Elements:
    # ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium',
    # 'Silicon', 'Sulphur', 'Calcium', 'Iron', 'Strontium', 'Barium']

    # Plot parameters
    params = {
        "font.size": 11,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (9, 7.6),
        "figure.subplot.left": 0.08,
        "figure.subplot.right": 0.98,
        "figure.subplot.bottom": 0.06,
        "figure.subplot.top": 0.98,
        "figure.subplot.wspace": 0.35,
        "figure.subplot.hspace": 0.1,
        "lines.markersize": 3,
        "lines.linewidth": 1,
        "figure.max_open_warning": 0,
    }
    rcParams.update(params)

    plt.figure()
    ax = plt.subplot(3, 3, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    indx = 2 # Carbon!

    for j in range(len(Z_ind)):
        Masses, yields = read_AGB_data(Z_ind[j])
        label = Z_ind[j].replace("Z_","")
        plt.plot(Masses, yields[indx, :], '-o', label=label, color=color_list[j])

    plt.axis([1, 12, -0.15, 0.1])
    plt.ylabel('Net Yields Carbon [M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    xticks = np.array([1, 2, 4, 6, 8, 10, 12])
    labels = ["1", "2", "4", "6", "8", "10", "12"]
    plt.xticks(xticks, labels)

    props = dict(boxstyle='square', fc='white', ec='black', alpha=1)
    ax.text(0.73, 0.94, 'Carbon', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    ####
    ax = plt.subplot(3, 3, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    indx = 3 # Nitrogen!

    for j in range(len(Z_ind)):
        Masses, yields = read_AGB_data(Z_ind[j])
        label = Z_ind[j].replace("Z_","")
        plt.plot(Masses, yields[indx, :], '-o', label=label, color=color_list[j])

    plt.axis([1, 12, -0.01, 0.2])
    plt.ylabel('Net Yields Nitrogen [M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    xticks = np.array([1, 2, 4, 6, 8, 10, 12])
    labels = ["1", "2", "4", "6", "8", "10", "12"]
    plt.xticks(xticks, labels)

    plt.legend(loc=[0.01, 0.4], labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
               columnspacing=0.4, ncol=2, fontsize=11)

    props = dict(boxstyle='square', fc='white', ec='black', alpha=1)
    ax.text(0.7, 0.94, 'Nitrogen', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    ####
    ax = plt.subplot(3, 3, 3)
    plt.grid(linestyle='-', linewidth=0.3)

    indx = 4 # Oxygen!

    for j in range(len(Z_ind)):
        Masses, yields = read_AGB_data(Z_ind[j])
        label = Z_ind[j].replace("Z_","")
        plt.plot(Masses, yields[indx, :], '-o', label=label, color=color_list[j])

    plt.axis([1, 12, -0.07, 0.02])
    plt.ylabel('Net Yields Oxygen [M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    xticks = np.array([1, 2, 4, 6, 8, 10, 12])
    labels = ["1", "2", "4", "6", "8", "10", "12"]
    plt.xticks(xticks, labels)

    props = dict(boxstyle='square', fc='white', ec='black', alpha=1)
    ax.text(0.71, 0.94, 'Oxygen', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    ####
    ax = plt.subplot(3, 3, 4)
    plt.grid(linestyle='-', linewidth=0.3)

    indx = 5  # Neon!

    for j in range(len(Z_ind)):
        Masses, yields = read_AGB_data(Z_ind[j])
        label = Z_ind[j].replace("Z_", "")
        plt.plot(Masses, yields[indx, :] / 1e-3, '-o', label=label, color=color_list[j])

    plt.axis([1, 12, -2, 8])
    plt.ylabel(r'Net Yields Neon [$\times 10^{-3}$ M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    xticks = np.array([1, 2, 4, 6, 8, 10, 12])
    labels = ["1", "2", "4", "6", "8", "10", "12"]
    plt.xticks(xticks, labels)

    props = dict(boxstyle='square', fc='white', ec='black', alpha=1)
    ax.text(0.78, 0.94, 'Neon', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    ####
    ax = plt.subplot(3, 3, 5)
    plt.grid(linestyle='-', linewidth=0.3)

    indx = 6  # Magnesium!

    for j in range(len(Z_ind)):
        Masses, yields = read_AGB_data(Z_ind[j])
        label = Z_ind[j].replace("Z_", "")
        plt.plot(Masses, yields[indx, :]/1e-3, '-o', label=label, color=color_list[j])

    plt.axis([1, 12, -0.2, 2])
    plt.ylabel(r'Net Yields Magnesium [$\times 10^{-3}$ M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    xticks = np.array([1, 2, 4, 6, 8, 10, 12])
    labels = ["1", "2", "4", "6", "8", "10", "12"]
    plt.xticks(xticks, labels)

    props = dict(boxstyle='square', fc='white', ec='black', alpha=1)
    ax.text(0.62, 0.94, 'Magnesium', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    ####
    ax = plt.subplot(3, 3, 6)
    plt.grid(linestyle='-', linewidth=0.3)

    indx = 7 # Silicon!

    for j in range(len(Z_ind)):
        Masses, yields = read_AGB_data(Z_ind[j])
        label = Z_ind[j].replace("Z_", "")
        plt.plot(Masses, yields[indx, :]/1e-4, '-o', label=label, color=color_list[j])

    plt.axis([1, 12, -0.5, 1.8])
    plt.ylabel(r'Net Yields Silicon [$\times 10^{-4}$ M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    xticks = np.array([1, 2, 4, 6, 8, 10, 12])
    labels = ["1", "2", "4", "6", "8", "10", "12"]
    plt.xticks(xticks, labels)

    props = dict(boxstyle='square', fc='white', ec='black', alpha=1)
    ax.text(0.72, 0.94, 'Silicon', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    ####
    ax = plt.subplot(3, 3, 7)
    plt.grid(linestyle='-', linewidth=0.3)

    indx = 10 # Iron!

    for j in range(len(Z_ind)):
        Masses, yields = read_AGB_data(Z_ind[j])
        label = Z_ind[j].replace("Z_", "")
        plt.plot(Masses, yields[indx, :] / 1e-5, '-o', label=label, color=color_list[j])

    plt.axis([1, 12, -5, 8])
    plt.ylabel(r'Net Yields Iron [$\times 10^{-5}$ M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    xticks = np.array([1, 2, 4, 6, 8, 10, 12])
    labels = ["1", "2", "4", "6", "8", "10", "12"]
    plt.xticks(xticks, labels)

    props = dict(boxstyle='square', fc='white', ec='black', alpha=1)
    ax.text(0.81, 0.94, 'Iron', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    ####
    ax = plt.subplot(3, 3, 8)
    plt.grid(linestyle='-', linewidth=0.3)

    indx = 11  # Barium!

    for j in range(len(Z_ind)):
        Masses, yields = read_AGB_data(Z_ind[j])
        label = Z_ind[j].replace("Z_", "")
        plt.plot(Masses, yields[indx, :], '-o', label=label, color=color_list[j])

    plt.axis([1, 12, 1e-11, 1e-5])
    plt.yscale('log')
    plt.ylabel(r'Net Yields Barium [M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    xticks = np.array([1, 2, 4, 6, 8, 10, 12])
    labels = ["1", "2", "4", "6", "8", "10", "12"]
    plt.xticks(xticks, labels)

    props = dict(boxstyle='square', fc='white', ec='black', alpha=1)
    ax.text(0.73, 0.94, 'Barium', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    ####
    ax = plt.subplot(3, 3, 9)
    plt.grid(linestyle='-', linewidth=0.3)

    indx = 12  # Strontium!

    for j in range(len(Z_ind)):
        Masses, yields = read_AGB_data(Z_ind[j])
        label = Z_ind[j].replace("Z_", "")
        plt.plot(Masses, yields[indx, :], '-o', label=label, color=color_list[j])

    plt.axis([1, 12, 1e-12, 1e-5])
    plt.yscale('log')
    plt.ylabel(r'Net Yields Strontium [M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    xticks = np.array([1, 2, 4, 6, 8, 10, 12])
    labels = ["1", "2", "4", "6", "8", "10", "12"]
    plt.xticks(xticks, labels)

    props = dict(boxstyle='square', fc='white', ec='black', alpha=1)
    ax.text(0.66, 0.94, 'Strontium', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    plt.savefig('./figures/AGB_Yield_tables_elements.png', dpi=400)

# def plot_AGB_tables():
#
#     with h5py.File('./data/AGB.hdf5', 'r') as data_file:
#         elements = [x.decode() for x in data_file['Species_names']]
#         Z_ind = [x.decode() for x in data_file['Yield_names']]
#         indx = np.arange(len(Z_ind))
#
#     color_list = ['grey','darkgreen','lightgreen','darkblue','tab:blue','lightblue',
#                   'pink','purple','tab:orange','crimson','black','brown','darkgrey']
#
#     for i, elem in enumerate(indx):
#
#         plt.figure()
#         ax = plt.subplot(1, 1, 1)
#         ax.grid(True)
#
#         for j in range(len(Z_ind)):
#             Masses, yields = read_AGB_data(Z_ind[j])
#             label = Z_ind[j].replace("Z_","")
#             plt.plot(Masses, yields[indx[i], :], '-o', label=label, color=color_list[j])
#
#         if indx[i] >= 11: plt.yscale('log')
#         plt.xlim(1, 12)
#         plt.ylabel('Net Yields ' + elements[elem] + ' [M$_{\odot}$]')
#         plt.xlabel('Initial stellar mass [M$_{\odot}$]')
#         plt.legend(labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
#                    columnspacing=0.4, ncol=3, fontsize=8)
#         ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
#         plt.savefig('./figures/AGB_Yield_tables_' + elements[elem] + '.png', dpi=200)


def plot_SNIa_yield_tables():
    # Read data
    with h5py.File('./data/SNIa.hdf5', 'r') as data_file:
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

    ax.grid(True, which='major', color='grey', linestyle='-', lw=0.2, zorder=0)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    plt.legend(labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=True, columnspacing=0.4, ncol=1,
               fontsize=8)

    plt.axis([0, 8.4, 1e-7, 1e0])
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"])

    plt.savefig('./figures/SNIa_yields.png', dpi=200)

def read_CCSN_data(Z_ind):

    with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        mej_ccsn = data_file["/Yields/"+Z_ind+"/Ejected_mass_in_ccsn"][:][:]
        mej_wind = data_file["/Yields/"+Z_ind+"/Ejected_mass_in_winds"][:]
        mass_loss = data_file["/Yields/"+Z_ind+"/Total_Mass_ejected"][:]

    return Masses, mej_wind, mej_ccsn, mass_loss

def plot_CCSN_yield_tables():

    # solar abundance values
    init_abundance_Hydrogen = 0.73738788833  # Initial fraction of particle mass in Hydrogen
    init_abundance_Helium = 0.24924186942  # Initial fraction of particle mass in Helium
    init_abundance_Carbon = 0.0023647215  # Initial fraction of particle mass in Carbon
    init_abundance_Nitrogen = 0.0006928991  # Initial fraction of particle mass in Nitrogen
    init_abundance_Oxygen = 0.00573271036  # Initial fraction of particle mass in Oxygen
    init_abundance_Neon = 0.00125649278  # Initial fraction of particle mass in Neon
    init_abundance_Magnesium = 0.00070797838  # Initial fraction of particle mass in Magnesium
    init_abundance_Silicon = 0.00066495154  # Initial fraction of particle mass in Silicon
    init_abundance_Sulphur = 0
    init_abundance_Calcium = 0
    init_abundance_Iron = 0.00129199252  # Initial fraction of particle mass in Iron

    init_abundance = np.array([init_abundance_Hydrogen,
                               init_abundance_Helium,
                               init_abundance_Carbon,
                               init_abundance_Nitrogen,
                               init_abundance_Oxygen,
                               init_abundance_Neon,
                               init_abundance_Magnesium,
                               init_abundance_Silicon,
                               init_abundance_Sulphur,
                               init_abundance_Calcium,
                               init_abundance_Iron])

    with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
        elements = [x.decode() for x in data_file['Species_names']]
        Z_ind = [x.decode() for x in data_file['Yield_names']]
        indx = np.arange(len(Z_ind))

    # Plot parameters
    params = {
        "font.size": 11,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (4, 3),
        "figure.subplot.left": 0.15,
        "figure.subplot.right": 0.97,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.93,
        "figure.subplot.wspace": 0.25,
        "figure.subplot.hspace": 0.25,
        "lines.markersize": 3,
        "lines.linewidth": 1,
        "figure.max_open_warning": 0,
    }
    rcParams.update(params)

    for i, elem in enumerate(indx):

        plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.grid(True)
        min_yields = 0

        for j in range(len(Z_ind)):
            Masses, mej_wind, mej_ccsn, mass_loss = read_CCSN_data(Z_ind[j])
            net_yields = mej_ccsn[i, :] + mej_wind * init_abundance[i] - init_abundance[i] * mass_loss[:]
            label = Z_ind[j].replace("Z_","")
            plt.plot(Masses, net_yields, '-o', label=label)
            min_yields = min(0,np.min(net_yields))

        if min_yields == 0: plt.yscale('log')
        plt.ylabel('Net Yields ' + elements[elem] + ' [M$_{\odot}$]')
        plt.xlabel('Initial stellar mass [M$_{\odot}$]')
        plt.xlim([6, 100])
        plt.legend(labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
                   columnspacing=0.4, ncol=2, fontsize=7)
        ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
        file_name = 'Yield_tables_'+ elements[elem] +'.png'
        plt.savefig('./figures/CCSN_' + file_name, dpi=300)


    # # Plot parameters
    # params = {
    #     "font.size": 11,
    #     "font.family": "Times",
    #     "text.usetex": True,
    #     "figure.figsize": (4, 3),
    #     "figure.subplot.left": 0.15,
    #     "figure.subplot.right": 0.97,
    #     "figure.subplot.bottom": 0.15,
    #     "figure.subplot.top": 0.93,
    #     "figure.subplot.wspace": 0.25,
    #     "figure.subplot.hspace": 0.25,
    #     "lines.markersize": 3,
    #     "lines.linewidth": 1,
    #     "figure.max_open_warning": 0,
    # }
    # rcParams.update(params)
    #
    # Z_list = np.array(['00', '01', '04', '08', '20', '50'])
    # for Zi in Z_list:
    #     # Read data
    #     with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
    #         Masses = data_file["Masses"][:]
    #         Z000_mej_ccsn = data_file["/Yields/Z_0.0" + Zi + "/Ejected_mass_in_ccsn"][:][:]
    #         Z000_mej_wind = data_file["/Yields/Z_0.0" + Zi + "/Ejected_mass_in_winds"][:]
    #         Z000_mass_loss = data_file["/Yields/Z_0.0" + Zi + "/Total_Mass_ejected"][:]
    #
    #     plt.figure()
    #
    #     ax = plt.subplot(1, 1, 1)
    #     ax.grid(True)
    #
    #     plt.plot(Masses, Masses, '--', lw=1, color='grey')
    #     plt.plot(Masses, Z000_mass_loss, '-', label='Total mass lost', color='black')
    #     plt.plot(Masses, Z000_mej_wind, '-', label='Mass ejected in stellar winds', color='grey')
    #     plt.plot(Masses, Z000_mej_ccsn[0, :], '-o', label='Mass ejected in CCSN, H', color='tab:blue')
    #     plt.plot(Masses, Z000_mej_ccsn[1, :], '-o', label='Mass ejected in CCSN, He', color='darkblue')
    #     plt.plot(Masses, Z000_mej_ccsn[2, :], '-o', label='Mass ejected in CCSN, C', color='tab:orange')
    #     plt.plot(Masses, Z000_mej_ccsn[3, :], '-o', label='Mass ejected in CCSN, N', color='tab:red')
    #     plt.plot(Masses, Z000_mej_ccsn[4, :], '-o', label='Mass ejected in CCSN, O', color='tab:green')
    #     plt.plot(Masses, Z000_mej_ccsn[5, :], '-o', label='Mass ejected in CCSN, Ne', color='purple')
    #     plt.plot(Masses, Z000_mej_ccsn[6, :], '-o', label='Mass ejected in CCSN, Mg', color='pink')
    #     plt.plot(Masses, Z000_mej_ccsn[7, :], '-o', label='Mass ejected in CCSN, Si', color='darkgreen')
    #     plt.plot(Masses, Z000_mej_ccsn[8, :], '-o', label='Mass ejected in CCSN, Fe', color='violet')
    #
    #     mej = Z000_mej_wind + np.sum(Z000_mej_ccsn[:, :], axis=0)
    #     plt.plot(Masses, mej, '--', color='black')
    #
    #     plt.text(42, 2e-3, 'Metallicity Z=0.0' + Zi + ' Z$_{\odot}$')
    #     plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    #     plt.ylabel('Mass ejected [M$_{\odot}$]')
    #     plt.xlim([6, 100])
    #     plt.ylim([1e-3, 5e3])
    #     plt.yscale('log')
    #     plt.legend(loc='upper left', labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
    #                columnspacing=0.4, ncol=2, fontsize=7)
    #     ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    #     plt.savefig("./figures/CCSN_mass_ejected_Z0" + Zi + ".png", dpi=300)


def plot_compare_CCSN_yield_tables():
    # Read data
    with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
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

    with h5py.File('./data/SNII.hdf5', 'r') as data_file:
        Masses_2 = data_file["Masses"][:]
        Z000_mej_ccsn_2 = data_file["/Yields/Z_0.004/Ejected_mass_in_ccsn"][:][:]
        Z000_mej_wind_2 = data_file["/Yields/Z_0.004/Ejected_mass_in_winds"][:]
        Z008_mej_ccsn_2 = data_file["/Yields/Z_0.008/Ejected_mass_in_ccsn"][:][:]
        Z008_mej_wind_2 = data_file["/Yields/Z_0.008/Ejected_mass_in_winds"][:]
        Z050_mej_ccsn_2 = data_file["/Yields/Z_0.020/Ejected_mass_in_ccsn"][:][:]
        Z050_mej_wind_2 = data_file["/Yields/Z_0.020/Ejected_mass_in_winds"][:]
        Z000_mass_loss_2 = data_file["/Yields/Z_0.004/Total_Mass_ejected"][:]
        Z008_mass_loss_2 = data_file["/Yields/Z_0.008/Total_Mass_ejected"][:]
        Z050_mass_loss_2 = data_file["/Yields/Z_0.020/Total_Mass_ejected"][:]

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

    ##################################################
    for i in range(num_species):

        plt.figure()

        ax = plt.subplot(1, 1, 1)
        ax.grid(True)

        plt.plot(Masses, Z000_mej_ccsn[i, :], '-o', label='$0.004Z_{\odot}$ (Kobayashi+ 06)', color='tab:orange')
        plt.plot(Masses, Z008_mej_ccsn[i, :], '-o', label='$0.008Z_{\odot}$ (Kobayashi+ 06)', color='tab:red')
        plt.plot(Masses, Z050_mej_ccsn[i, :], '-o', label='$0.020Z_{\odot}$ (Kobayashi+ 06)', color='tab:blue')

        plt.plot(Masses, Z000_mej_ccsn[i, :], '--', color='tab:orange')
        plt.plot(Masses, Z008_mej_ccsn[i, :], '--', color='tab:red')
        plt.plot(Masses, Z050_mej_ccsn[i, :], '--', color='tab:blue')

        if i == 0:
            plt.ylabel('Mass Ejected Hydrogen [M$_{\odot}$]')
            file_name = 'Mass_Ejected_Hydrogen.png'
            # plt.ylim([-15, 5])
        if i == 1:
            plt.ylabel('Mass Ejected Helium [M$_{\odot}$]')
            file_name = 'Mass_Ejected_Helium.png'
            # plt.yscale('log')
        if i == 2:
            plt.ylabel('Mass Ejected Carbon [M$_{\odot}$]')
            file_name = 'Mass_Ejected_Carbon.png'
            # plt.yscale('log')
        if i == 3:
            plt.ylabel('Mass Ejected Nitrogen [M$_{\odot}$]')
            file_name = 'Mass_Ejected_Nitrogen.png'
            # plt.yscale('log')
        if i == 4:
            plt.ylabel('Mass Ejected Oxygen [M$_{\odot}$]')
            file_name = 'Mass_Ejected_Oxygen.png'
            # plt.yscale('log')
            # plt.ylim([1e-2, 1e2])
        if i == 5:
            plt.ylabel('Mass Ejected Neon [M$_{\odot}$]')
            file_name = 'Mass_Ejected_Neon.png'
            # plt.yscale('log')
            # plt.ylim([1e-3, 1e1])
        if i == 6:
            plt.ylabel('Mass Ejected Magnesium [M$_{\odot}$]')
            file_name = 'Mass_Ejected_Magnesium.png'
            # plt.yscale('log')
            # plt.ylim([1e-3, 1e1])
        if i == 7:
            plt.ylabel('Mass Ejected Silicon [M$_{\odot}$]')
            file_name = 'Mass_Ejected_Silicon.png'
            # plt.yscale('log')
            # plt.ylim([1e-2, 1e1])
        if i == 8:
            plt.ylabel('Mass Ejected Iron [M$_{\odot}$]')
            file_name = 'Mass_Ejected_Iron.png'
            # plt.ylim([-0.1, 0.3])
            # plt.yscale('log')

        plt.xlabel('Initial stellar mass [M$_{\odot}$]')
        # plt.axis([1,50,1e-2,1e1])
        plt.xlim([6, 100])
        plt.legend(loc='upper left', labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
                   columnspacing=0.4, ncol=1, fontsize=8)
        ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
        plt.savefig('./figures/CCSN_' + file_name, dpi=200)