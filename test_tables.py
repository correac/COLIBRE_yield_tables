import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy import interpolate

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

def calculate_imf(log_mass_range):

    num_bins = len(log_mass_range)
    imf = np.zeros(num_bins)
    mass = 10**log_mass_range

    for i in range(num_bins):

        if mass[i] > 1:
            imf[i] = 0.237912 * mass[i]**(-2.3)
        else:
            imf[i] = 0.852464 * \
                  np.exp((log_mass_range[i] - np.log10(0.079)) *
                         (log_mass_range[i] - np.log10(0.079)) / (-2.0 * 0.69 * 0.69)) / mass[i]

    imf_all = imf * mass**2
    sum_imf = np.sum ( imf * mass**2 )
    sum_imf -= 0.5 * (imf_all[0] + imf_all[-1])
    sum_imf *= 0.2 * np.log(10)

    return imf / sum_imf

def integrate_imf(mass, stellar_yields):

    log_mass_range = np.arange(0, np.log10(100), 0.2)
    num_bins = len(log_mass_range)
    integrand = np.zeros(num_bins)

    imf = calculate_imf(log_mass_range)
    f = interpolate.interp1d(mass, stellar_yields)

    for i in range(num_bins-1):
        if np.min(mass) > 10**log_mass_range[i]:
            stellar_yields_bin = stellar_yields[0]
        #elif 10**log_mass_range[i] > 6:
        elif np.max(mass) < 10 ** log_mass_range[i]:
            stellar_yields_bin = stellar_yields[-1]
        else:
            stellar_yields_bin = f(10**log_mass_range[i])

        integrand[i] = stellar_yields_bin * imf[i] * 10**log_mass_range[i]

    return np.sum( integrand )



def plot_metal_relations_AGB():

    metallicity_list = np.array(['0.007', '0.014', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10'])
    metal_array = np.array([0.007,0.014,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10])

    mass_fraction = np.array([0.73738788833, #H
                              0.24924186942, #He
                              0.0023647215,  #C
                              0.0006928991,  #N
                              0.00573271036, #O
                              0.00125649278, #Ne
                              0.00070797838, #Mg
                              0.00066495154, #Si
                              0.00129199252, #Fe
                              0.0, #Sr,
                              0.0]) #Ba

    total_mass_fraction = 0.0133714

    elements = np.array(['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron','Strontium','Barium'])
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12])

    total_metal = np.zeros(len(metal_array))
    metal_elem_sum = np.zeros(len(metal_array))
    metal_elem_zero_sum = np.zeros(len(metal_array))

    for j, metallicity in enumerate(metallicity_list):

        # Write data to HDF5
        with h5py.File('./data/AGB.hdf5', 'r') as data_file:
            Masses = data_file["Masses"][:]
            Yield = data_file["/Yields/Z_"+metallicity+"/Yield"][:][:]
            Ejected_mass = data_file["/Yields/Z_"+metallicity+"/Ejected_mass"][:]
            Total_metals = data_file["/Yields/Z_"+metallicity+"/Total_Metals"][:]

        factor = metal_array[j] / total_mass_fraction

        num_mass_bins = len(Masses)
        num_elements = len(mass_fraction)
        stellar_yields = np.zeros((num_elements, num_mass_bins))
        total_yields = np.zeros(num_mass_bins)
        metal_mass_released = np.zeros(num_elements)
        metal_mass_released_zero = np.zeros(num_elements)

        for i in range(num_mass_bins):
            stellar_yields[:, i] = Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]
            total_yields[i] = Total_metals[i] + factor * total_mass_fraction * Ejected_mass[i]

        for i in range(num_elements):
            metal_mass_released[i] = integrate_imf(Masses, stellar_yields[i,:])
            metal_mass_released_zero[i] = np.maximum(metal_mass_released[i], 0)

        print('===')
        print(metallicity)
        print(metal_mass_released)
        print(metal_mass_released_zero)

        total_metal_mass_released = integrate_imf(Masses, total_yields)

        total_metal[j] = total_metal_mass_released
        metal_elem_sum[j] = np.sum(metal_mass_released[2:])
        metal_elem_zero_sum[j] = np.sum(metal_mass_released_zero[2:])


    plt.figure()

    ax = plt.subplot(1, 1, 1)
    ax.grid(True)

    plt.plot(metal_array, total_metal,'-' ,color='tab:blue',label='Total metals')
    plt.plot(metal_array, metal_elem_sum,'-' ,color='tab:orange',label='Sum of metals')
    plt.plot(metal_array, metal_elem_zero_sum,'-' ,color='tab:red',label='Sum of metals (negative-to-zero)')

    #plt.axis([0.,0.1,0.,0.15])
    plt.ylabel('Total metal mass produced [M$_{\odot}$]')
    plt.xlabel('Star metallicity')
    plt.legend(loc=[0.02, 0.75], labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=1, fontsize=10)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('./figures/Test_AGB_Yield_tables.png', dpi=300)


# def plot_metal_relations_CCSN():
#
#     metallicity_list = np.array(['0.000', '0.001', '0.004', '0.008', '0.020', '0.050'])
#     metal_array = np.array([0.00,0.001,0.004,0.008, 0.02, 0.05])
#
#     mass_fraction = np.array([0.73738788833, #H
#                               0.24924186942, #He
#                               0.0023647215,  #C
#                               0.0006928991,  #N
#                               0.00573271036, #O
#                               0.00125649278, #Ne
#                               0.00070797838, #Mg
#                               0.00066495154, #Si
#                               0.00129199252]) #Fe
#
#     total_mass_fraction = 0.0133714
#
#     elements = np.array(['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron'])
#     indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
#
#     total_metal = np.zeros(len(metal_array))
#     metal_elem_sum = np.zeros(len(metal_array))
#     metal_elem_zero_sum = np.zeros(len(metal_array))
#
#     for j, metallicity in enumerate(metallicity_list):
#
#         # Write data to HDF5
#         with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
#             Masses = data_file["Masses"][:]
#             Ejected_mass_ccsn = data_file["/Yields/Z_"+metallicity+"/Ejected_mass_in_ccsn"][:][:]
#             Ejected_mass_winds = data_file["/Yields/Z_"+metallicity+"/Ejected_mass_in_winds"][:]
#             Total_mass_ej = data_file["/Yields/Z_"+metallicity+"/Total_Mass_ejected"][:]
#
#         factor = metal_array[j] / total_mass_fraction
#
#         num_mass_bins = len(Masses)
#         num_elements = len(mass_fraction)
#         stellar_yields = np.zeros((num_elements, num_mass_bins))
#         total_yields = np.zeros(num_mass_bins)
#         metal_mass_released = np.zeros(num_elements)
#         metal_mass_released_zero = np.zeros(num_elements)
#
#         for i in range(num_mass_bins):
#             stellar_yields[:, i] = Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]
#             total_yields[i] = Total_metals[i] + factor * total_mass_fraction * Ejected_mass[i]
#
#         for i in range(num_elements):
#             metal_mass_released[i] = integrate_imf(Masses, stellar_yields[i,:])
#             metal_mass_released_zero[i] = np.maximum(metal_mass_released[i], 0)
#
#         print('===')
#         print(metallicity)
#         print(metal_mass_released)
#         print(metal_mass_released_zero)
#
#         total_metal_mass_released = integrate_imf(Masses, total_yields)
#
#         total_metal[j] = total_metal_mass_released
#         metal_elem_sum[j] = np.sum(metal_mass_released[2:])
#         metal_elem_zero_sum[j] = np.sum(metal_mass_released_zero[2:])
#
#
#     plt.figure()
#
#     ax = plt.subplot(1, 1, 1)
#     ax.grid(True)
#
#     plt.plot(metal_array, total_metal,'-' ,color='tab:blue',label='Total metals')
#     plt.plot(metal_array, metal_elem_sum,'-' ,color='tab:orange',label='Sum of metals')
#     plt.plot(metal_array, metal_elem_zero_sum,'-' ,color='tab:red',label='Sum of metals (negative-to-zero)')
#
#     #plt.axis([0.,0.1,0.,0.15])
#     plt.ylabel('Total metal mass produced [M$_{\odot}$]')
#     plt.xlabel('Star metallicity')
#     plt.legend(loc=[0.02, 0.75], labelspacing=0.2, handlelength=0.8,
#                handletextpad=0.3, frameon=False, columnspacing=0.4,
#                ncol=1, fontsize=10)
#     ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
#     plt.savefig('./figures/Test_AGB_Yield_tables.png', dpi=300)

if __name__ == "__main__":

    plot_metal_relations_AGB()
    # plot_metal_relations_CCSN()