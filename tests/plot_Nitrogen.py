import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit

def func(x,a,b,c):
    f = a + b*x + c *x**2
    return f

def plot_AGB_Zdependence():

    Znew = np.arange(np.log10(1e-4),np.log10(0.1),0.01)
    Znew = 10**Znew

    # Write data to HDF5
    with h5py.File('../data/AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Z = data_file["Metallicities"][:]
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

    i = 3
    j = 0
    N = np.array([Y_Z0007[i, j],Y_Z0014[i, j],Y_Z003[i, j],Y_Z004[i, j],Y_Z005[i, j],Y_Z006[i, j],Y_Z007[i, j],Y_Z008[i, j],Y_Z009[i, j],Y_Z010[i, j]])
    plt.plot(Z, N, '-', label='M=1M$_{\odot}$', color='tab:orange')

    popt, pcov = curve_fit(func, Z, np.log10(N))
    plt.plot(Znew,10**func(Znew,*popt),'--', color='tab:orange')

    j = 4
    N = np.array([Y_Z0007[i, j],Y_Z0014[i, j],Y_Z003[i, j],Y_Z004[i, j],Y_Z005[i, j],Y_Z006[i, j],Y_Z007[i, j],Y_Z008[i, j],Y_Z009[i, j],Y_Z010[i, j]])
    plt.plot(Z, N, '-', label='M=2M$_{\odot}$', color='tab:blue')

    popt, pcov = curve_fit(func, Z, np.log10(N))
    plt.plot(Znew,10**func(Znew,*popt),'--', color='tab:blue')

    j = 8
    N = np.array([Y_Z0007[i, j],Y_Z0014[i, j],Y_Z003[i, j],Y_Z004[i, j],Y_Z005[i, j],Y_Z006[i, j],Y_Z007[i, j],Y_Z008[i, j],Y_Z009[i, j],Y_Z010[i, j]])
    plt.plot(Z, N, '-', label='M=3M$_{\odot}$', color='crimson')

    popt, pcov = curve_fit(func, Z, np.log10(N))
    plt.plot(Znew,10**func(Znew,*popt),'--', color='crimson')

    j = 12
    N = np.array([Y_Z0007[i, j],Y_Z0014[i, j],Y_Z003[i, j],Y_Z004[i, j],Y_Z005[i, j],Y_Z006[i, j],Y_Z007[i, j],Y_Z008[i, j],Y_Z009[i, j],Y_Z010[i, j]])
    plt.plot(Z, N, '-', label='M=4M$_{\odot}$', color='lightgreen')

    popt, pcov = curve_fit(func, Z, np.log10(N))
    plt.plot(Znew,10**func(Znew,*popt),'--', color='lightgreen')

    j = 14
    N = np.array([Y_Z0007[i, j],Y_Z0014[i, j],Y_Z003[i, j],Y_Z004[i, j],Y_Z005[i, j],Y_Z006[i, j],Y_Z007[i, j],Y_Z008[i, j],Y_Z009[i, j],Y_Z010[i, j]])
    plt.plot(Z, N, '-', label='M=4.5M$_{\odot}$', color='green')

    select = np.where(Z>=0.04)[0]
    popt, pcov = curve_fit(func, Z[select], np.log10(N[select]))
    plt.plot(Znew,10**func(Znew,*popt),'--', color='green')

    j = 16
    N = np.array([Y_Z0007[i, j],Y_Z0014[i, j],Y_Z003[i, j],Y_Z004[i, j],Y_Z005[i, j],Y_Z006[i, j],Y_Z007[i, j],Y_Z008[i, j],Y_Z009[i, j],Y_Z010[i, j]])
    plt.plot(Z, N, '-', label='M=5M$_{\odot}$', color='darkblue')

    popt, pcov = curve_fit(func, Z[select], np.log10(N[select]))
    plt.plot(Znew,10**func(Znew,*popt),'--', color='darkblue')

    j = 18
    N = np.array([Y_Z0007[i, j],Y_Z0014[i, j],Y_Z003[i, j],Y_Z004[i, j],Y_Z005[i, j],Y_Z006[i, j],Y_Z007[i, j],Y_Z008[i, j],Y_Z009[i, j],Y_Z010[i, j]])
    plt.plot(Z, N, '-', label='M=6M$_{\odot}$', color='darksalmon')

    popt, pcov = curve_fit(func, Z[select], np.log10(N[select]))
    plt.plot(Znew,10**func(Znew,*popt),'--', color='darksalmon')

    j = 20
    N = np.array([Y_Z0007[i, j],Y_Z0014[i, j],Y_Z003[i, j],Y_Z004[i, j],Y_Z005[i, j],Y_Z006[i, j],Y_Z007[i, j],Y_Z008[i, j],Y_Z009[i, j],Y_Z010[i, j]])
    plt.plot(Z, N, '-', label='M=8M$_{\odot}$', color='brown')

    popt, pcov = curve_fit(func, Z, np.log10(N))
    plt.plot(Znew,10**func(Znew,*popt),'--', color='brown')


def plot_original_AGB():

    # Write data to HDF5
    with h5py.File('../data/AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
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

    i = 3
    elem = 'N'

    plt.plot(Masses, Y_Z0007[i, :], '-', label='$Z=0.007$', color='tab:orange')
    plt.plot(Masses, Y_Z0014[i, :], '-', label='$Z=0.014$', color='tab:blue')
    plt.plot(Masses, Y_Z003[i, :], '-', label='$Z=0.03$', color='crimson')
    plt.plot(Masses, Y_Z004[i, :], '-', label='$Z=0.04$', color='lightgreen')
    plt.plot(Masses, Y_Z005[i, :], '-', label='$Z=0.05$', color='darkblue')
    plt.plot(Masses, Y_Z006[i, :], '-', label='$Z=0.06$', color='darksalmon')
    plt.plot(Masses, Y_Z007[i, :], '-', label='$Z=0.07$', color='tab:green')
    plt.plot(Masses, Y_Z008[i, :], '-', label='$Z=0.08$', color='tab:purple')
    plt.plot(Masses, Y_Z009[i, :], '-', label='$Z=0.09$', color='pink')
    plt.plot(Masses, Y_Z010[i, :], '-', label='$Z=0.10$', color='grey')

def plot_modified_AGB():

    # Write data to HDF5
    with h5py.File('../data/AGB_newNlowZ.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Y_Z00001 = data_file["/Yields/Z_0.0001/Yield"][:][:]
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

    i = 3
    elem = 'N'

    plt.plot(Masses, Y_Z00001[i, :], '-', color='darkgreen')
    plt.plot(Masses, Y_Z0001[i, :], '-', color='black')
    plt.plot(Masses, Y_Z0004[i, :], '-', color='lightblue')
    plt.plot(Masses, Y_Z0007[i, :], '--', color='tab:orange')
    plt.plot(Masses, Y_Z0014[i, :], '--', color='tab:blue')
    # plt.plot(Masses, Y_Z003[i, :], '--', color='crimson')
    # plt.plot(Masses, Y_Z004[i, :], '--', color='lightgreen')
    # plt.plot(Masses, Y_Z005[i, :], '--', color='darkblue')
    # plt.plot(Masses, Y_Z006[i, :], '--', color='darksalmon')
    # plt.plot(Masses, Y_Z007[i, :], '--', color='tab:green')
    # plt.plot(Masses, Y_Z008[i, :], '--', color='tab:purple')
    # plt.plot(Masses, Y_Z009[i, :], '--', color='pink')
    # plt.plot(Masses, Y_Z010[i, :], '--', color='grey')

def plot_Doherty():

    # Write data to HDF5
    with h5py.File('../data/Doherty2014/AGB_Doherty2014.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Y_Z0007 = data_file["/Yields/Z_0.004/Yield"][:][:]
        Y_Z0014 = data_file["/Yields/Z_0.008/Yield"][:][:]
        Y_Z003 = data_file["/Yields/Z_0.02/Yield"][:][:]

    i = 3
    plt.plot(Masses, Y_Z0007[i, :], '--', label='Z=0.004', color='tab:orange')
    plt.plot(Masses, Y_Z0014[i, :], '--', label='Z=0.008', color='tab:blue')
    plt.plot(Masses, Y_Z003[i, :], '--', label='Z=0.02', color='crimson')

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

elements = np.array(['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Magnesium','Iron','Strontium','Barium'])
indx = np.array([0, 1, 2, 3, 4, 6, 10, 11, 12])

plt.figure()
ax = plt.subplot(1, 1, 1)
ax.grid(True)
ax.grid(linestyle='-',linewidth=0.3)

plot_original_AGB()
plot_modified_AGB()
#plot_Doherty()

plt.xlim(1, 12)
plt.ylim(1e-5, 1)
plt.yscale('log')
plt.ylabel('Net Yields Nitrogen [M$_{\odot}$]')
plt.xlabel('Initial stellar mass [M$_{\odot}$]')
plt.legend(loc=[0.01, 0.72], labelspacing=0.2, handlelength=0.8,
           handletextpad=0.3, frameon=False, columnspacing=0.4,
           ncol=3, fontsize=8)
ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
plt.savefig('./Nitrogen_Yields.png', dpi=300)

#####

plt.figure()
ax = plt.subplot(1, 1, 1)
ax.grid(True)
ax.grid(linestyle='-',linewidth=0.3)

plot_AGB_Zdependence()

plt.yscale('log')
plt.ylabel('Net Yields Nitrogen [M$_{\odot}$]')
plt.xlabel('Metallicity Z$_{\odot}$')
plt.legend(loc=[0.01, 0.72], labelspacing=0.2, handlelength=0.8,
           handletextpad=0.3, frameon=False, columnspacing=0.4,
           ncol=3, fontsize=8)
ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
plt.savefig('./Nitrogen_Yields_Zdependence.png', dpi=300)

