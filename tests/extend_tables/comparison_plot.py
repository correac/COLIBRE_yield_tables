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

def plot_original():

    # Read data
    with h5py.File('../../data/AGB.hdf5', 'r') as data_file:
        mass_bins = data_file["Masses"][:]
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

    plt.plot(mass_bins, Y_Z0007[i, :], '--', color='tab:purple')
    plt.plot(mass_bins, Y_Z0014[i, :], '--', color='tab:blue')
    plt.plot(mass_bins, Y_Z003[i, :], '--', color='tab:green')
    plt.plot(mass_bins, Y_Z005[i, :], '--', color='tab:pink')
    plt.plot(mass_bins, Y_Z007[i, :], '--', color='tab:orange')
    plt.plot(mass_bins, Y_Z010[i, :], '--', color='tab:red')

class make_yield_tables:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.0001, 0.001, 0.004, 0.007, 0.014, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        self.num_Z_bins = len(self.Z_bins)

        self.species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56]) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.5, 6., 7., 8., 9., 10., 11., 12.])
        self.num_mass_bins = len(self.mass_bins)


yields = make_yield_tables()

# Read data
with h5py.File('extendedAGB.hdf5', 'r') as data_file:
    Y_Z00001 = data_file["/Yields/Z_0.001/Yield"][:][:]
    Y_Z0001 = data_file["/Yields/Z_0.001/Yield"][:][:]
    Y_Z0004 = data_file["/Yields/Z_0.004/Yield"][:][:]
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

for i in range(yields.num_species):

    plt.figure()

    ax = plt.subplot(1, 1, 1)
    ax.grid(True)

    plt.plot(yields.mass_bins, Y_Z00001[i, :], '-', label='Z=0.0001', color='grey')
    plt.plot(yields.mass_bins, Y_Z0001[i, :], '-', label='Z=0.001', color='brown')
    plt.plot(yields.mass_bins, Y_Z0004[i, :], '-', label='Z=0.004', color='greenyellow')
    plt.plot(yields.mass_bins, Y_Z0007[i, :], '-', label='Z=0.007', color='tab:purple')
    plt.plot(yields.mass_bins, Y_Z0014[i, :], '-', label='Z=0.014', color='tab:blue')
    plt.plot(yields.mass_bins, Y_Z003[i, :], '-', label='Z=0.003', color='tab:green')
    plt.plot(yields.mass_bins, Y_Z005[i, :], '-', label='Z=0.005', color='tab:pink')
    plt.plot(yields.mass_bins, Y_Z007[i, :], '-', label='Z=0.007', color='tab:orange')
    plt.plot(yields.mass_bins, Y_Z010[i, :], '-', label='Z=0.01', color='tab:red')
    plot_original()

    # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
    if i == 0:
        plt.ylabel('Net Yields Hydrogen [M$_{\odot}$]')
        file_name = 'Yield_tables_Hydrogen.png'
        # plt.ylim([-15, 5])
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
        # plt.yscale('log')
        # plt.ylim([1e-2, 1e2])
    if i == 5:
        plt.ylabel('Net Yields Neon [M$_{\odot}$]')
        file_name = 'Yield_tables_Neon.png'
        plt.yscale('log')
        # plt.ylim([1e-3, 1e1])
    if i == 6:
        plt.ylabel('Net Yields Magnesium [M$_{\odot}$]')
        file_name = 'Yield_tables_Magnesium.png'
        # plt.yscale('log')
        # plt.ylim([1e-3, 1e1])
    if i == 7:
        plt.ylabel('Net Yields Silicon [M$_{\odot}$]')
        file_name = 'Yield_tables_Silicon.png'
        # plt.yscale('log')
        # plt.ylim([1e-2, 1e1])
    if i == 8:
        plt.ylabel('Net Yields S [M$_{\odot}$]')
        file_name = 'Yield_tables_S.png'
        # plt.ylim([-0.1, 0.3])
        # plt.yscale('log')
    if i == 9:
        plt.ylabel('Net Yields Ca [M$_{\odot}$]')
        file_name = 'Yield_tables_Ca.png'
        # plt.ylim([-0.1, 0.3])
        # plt.yscale('log')
    if i == 10:
        plt.ylabel('Net Yields Iron [M$_{\odot}$]')
        file_name = 'Yield_tables_Iron.png'
        # plt.ylim([-0.1, 0.3])
        # plt.yscale('log')
    if i == 11:
        plt.ylabel('Net Yields Sr [M$_{\odot}$]')
        file_name = 'Yield_tables_Sr.png'
        # plt.ylim([-0.1, 0.3])
        plt.yscale('log')
    if i == 12:
        plt.ylabel('Net Yields Ba [M$_{\odot}$]')
        file_name = 'Yield_tables_Ba.png'
        # plt.ylim([-0.1, 0.3])
        plt.yscale('log')

    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    # plt.axis([1,50,1e-2,1e1])
    plt.legend(loc='lower right', labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False,
               columnspacing=0.4, ncol=1, fontsize=7)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('./' + file_name, dpi=300)