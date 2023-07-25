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

def read_AGB_data(Z_ind, file):

    with h5py.File(file, 'r') as data_file:
        Masses = data_file["Masses"][:]
        yields = data_file["/Yields/"+Z_ind+"/Yield"][:][:]

    return Masses, yields

def plot_AGB_tables(file,output_file):

    with h5py.File(file, 'r') as data_file:
        elements = [x.decode() for x in data_file['Species_names']]
        Z_ind = [x.decode() for x in data_file['Yield_names']]
        indx = np.arange(len(Z_ind))

    for i, elem in enumerate(indx):

        plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.grid(True)

        for j in range(len(Z_ind)):
            Masses, yields = read_AGB_data(Z_ind[j], file)
            label = Z_ind[j].replace("Z_","")
            plt.plot(Masses, yields[indx[i], :], '-o', label=label)

        if indx[i] >= 11: plt.yscale('log')
        plt.xlim(1, 12)
        plt.ylabel('Net Yields ' + elements[elem] + ' [M$_{\odot}$]')
        plt.xlabel('Initial stellar mass [M$_{\odot}$]')
        plt.legend(labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
                   columnspacing=0.4, ncol=3, fontsize=8)
        ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
        plt.savefig(output_file + elements[elem] + '.png', dpi=200)


def read_AGB_data_mej(Z_ind, file):

    with h5py.File(file, 'r') as data_file:
        Masses = data_file["Masses"][:]
        m_ej = data_file["/Yields/"+Z_ind+"/Ejected_mass"][:]
        m_total = data_file["/Yields/"+Z_ind+"/Total_Metals"][:]

    return Masses, m_ej, m_total


def plot_AGB_tables_mej(file, output_file):

    with h5py.File(file, 'r') as data_file:
        Z_ind = [x.decode() for x in data_file['Yield_names']]


    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)

    for j in range(len(Z_ind)):
        Masses, m_ej, _ = read_AGB_data_mej(Z_ind[j], file)
        label = Z_ind[j].replace("Z_","")
        plt.plot(Masses, m_ej, '-o', label=label)

    plt.yscale('log')
    plt.xlim(1, 12)
    plt.ylabel('Mass ejected [M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    plt.legend(labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
               columnspacing=0.4, ncol=3, fontsize=8)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_file + 'mej.png', dpi=200)

    ###

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.grid(True)

    for j in range(len(Z_ind)):
        Masses, _, m_total = read_AGB_data_mej(Z_ind[j], file)
        label = Z_ind[j].replace("Z_", "")
        plt.plot(Masses, m_total, '-o', label=label)

    plt.yscale('log')
    plt.xlim(1, 12)
    plt.ylabel('Total Metal Mass [M$_{\odot}$]')
    plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    plt.legend(labelspacing=0.2, handlelength=0.8, handletextpad=0.3, frameon=False,
               columnspacing=0.4, ncol=3, fontsize=8)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig(output_file + 'mtotal.png', dpi=200)

if __name__ == "__main__":

    file = 'AGB_test.hdf5'
    output_file = 'New_AGB_Yield_tables_'
    plot_AGB_tables_mej(file, output_file)
    plot_AGB_tables(file, output_file)

    file = 'AGB_fiducial.hdf5'
    output_file = 'Fiducial_AGB_Yield_tables_'
    plot_AGB_tables_mej(file, output_file)
    plot_AGB_tables(file, output_file)
