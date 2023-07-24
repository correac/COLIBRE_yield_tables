import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from datetime import datetime

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

class make_yield_tables:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.007, 0.014, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        self.num_Z_bins = len(self.Z_bins)

        self.species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56]) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.5, 6., 7., 8., 9., 10., 11., 12.])
        self.num_mass_bins = len(self.mass_bins)

def modify_AGB_yield_tables():

    yields = make_yield_tables()

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
        Mej_Z0007 = data_file["/Yields/Z_0.007/Ejected_mass"][:]
        Mej_Z0014 = data_file["/Yields/Z_0.014/Ejected_mass"][:]
        Mej_Z003 = data_file["/Yields/Z_0.03/Ejected_mass"][:]

        Mej_Z004 = data_file["/Yields/Z_0.04/Ejected_mass"][:]
        Mej_Z005 = data_file["/Yields/Z_0.05/Ejected_mass"][:]
        Mej_Z006 = data_file["/Yields/Z_0.06/Ejected_mass"][:]
        Mej_Z007 = data_file["/Yields/Z_0.07/Ejected_mass"][:]
        Mej_Z008 = data_file["/Yields/Z_0.08/Ejected_mass"][:]
        Mej_Z009 = data_file["/Yields/Z_0.09/Ejected_mass"][:]
        Mej_Z010 = data_file["/Yields/Z_0.10/Ejected_mass"][:]

        Mtot_Z0007 = data_file["/Yields/Z_0.007/Total_Metals"][:]
        Mtot_Z0014 = data_file["/Yields/Z_0.014/Total_Metals"][:]
        Mtot_Z003 = data_file["/Yields/Z_0.03/Total_Metals"][:]

        Mtot_Z004 = data_file["/Yields/Z_0.04/Total_Metals"][:]
        Mtot_Z005 = data_file["/Yields/Z_0.05/Total_Metals"][:]
        Mtot_Z006 = data_file["/Yields/Z_0.06/Total_Metals"][:]
        Mtot_Z007 = data_file["/Yields/Z_0.07/Total_Metals"][:]
        Mtot_Z008 = data_file["/Yields/Z_0.08/Total_Metals"][:]
        Mtot_Z009 = data_file["/Yields/Z_0.09/Total_Metals"][:]
        Mtot_Z010 = data_file["/Yields/Z_0.10/Total_Metals"][:]

    # Modification
    factor = 0.5
    i = 3
    lowM = np.where(Masses <= 6)[0]
    Y_Z0007[i, lowM] *= factor
    Y_Z0014[i, lowM] *= factor
    Y_Z003[i, lowM] *= factor
    Y_Z004[i, lowM] *= factor
    Y_Z005[i, lowM] *= factor
    Y_Z006[i, lowM] *= factor
    Y_Z007[i, lowM] *= factor
    Y_Z008[i, lowM] *= factor
    Y_Z009[i, lowM] *= factor
    Y_Z010[i, lowM] *= factor


    elements = np.array(['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Magnesium','Iron','Strontium','Barium'])
    indx = np.array([0, 1, 2, 3, 4, 6, 10, 11, 12])

    # Write data to HDF5
    with h5py.File('../data/AGB_0p5NlowM.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Net yields for AGB stars (in units of solar mass) taken from Cinquegrana & Karakas (2021) "
        description += "for the metallicity range Z=0.04-0.1, and initial mass range 1-8 Msun. For the mass range 9-12Msun, these tables have been extrapolated. "
        description += "Additionally, the net yields for AGB stars in the metallicity bins Z=0.03, 0.014 and 0.007 and initial mass range 1-8 Msun are taken from Karakas & Lugaro (2016). "
        description += "For the mass range 9-12Msun, we have interpolated the data from the tables of Doherty et al. (2014). We decreased yields of Nitrogen in all "
        description += "metallicity bins larger than 0.03 by a factor of 2."
        Header.attrs["Description"] = np.string_(description)

        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"] = np.string_(contact)

        # date_int = int(datetime.today().strftime('%Y%m%d'))
        # date_string = data_file.create_dataset('Date_string', data=np.array([date_int]))
        date_string = data_file.create_dataset('Date_string', data=np.array([20220718]))

        MH = data_file.create_dataset('Masses', data=yields.mass_bins)
        MH.attrs["Description"] = np.string_("Mass bins in units of Msolar")

        MH = data_file.create_dataset('Metallicities', data=yields.Z_bins)

        MH.attrs["Description"] = np.string_("Metallicity bins")

        MH = data_file.create_dataset('Number_of_metallicities', data=yields.num_Z_bins)

        MH = data_file.create_dataset('Number_of_masses', data=yields.num_mass_bins)

        MH = data_file.create_dataset('Number_of_species', data=np.array([13]))

        Z_names = ['Z_0.007', 'Z_0.014', 'Z_0.03', 'Z_0.04', 'Z_0.05', 'Z_0.06', 'Z_0.07', 'Z_0.08', 'Z_0.09', 'Z_0.10']
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        MH = data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Z_names = ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur',
                   'Calcium', 'Iron', 'Strontium', 'Barium']
        Element_names = np.string_(Z_names)
        dt = h5py.string_dtype(encoding='ascii')
        MH = data_file.create_dataset('Species_names', dtype=dt, data=Element_names)

        Reference = np.string_([
                                   'Cinquegrana & Karakas (2021) MNRAS, 3045C; Karakas & Lugaro (2016) ApJ, 825, 26K; Doherty, C L. Gil-Pons, P. Lau, H.H.B Lattanzio, J C and Siess, L. 2014. MNRAS, 437,195'])
        MH = data_file.create_dataset('Reference', data=Reference)

        Data = data_file.create_group('Yields')

        Z0001 = Data.create_group('Z_0.007')
        MH = Z0001.create_dataset('Yield', data=Y_Z0007)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z0007)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z0007)

        Z0001 = Data.create_group('Z_0.014')
        MH = Z0001.create_dataset('Yield', data=Y_Z0014)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z0014)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z0014)

        Z0001 = Data.create_group('Z_0.03')
        MH = Z0001.create_dataset('Yield', data=Y_Z003)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z003)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z003)

        Z0001 = Data.create_group('Z_0.04')
        MH = Z0001.create_dataset('Yield', data=Y_Z004)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z004)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z004)

        Z0001 = Data.create_group('Z_0.05')
        MH = Z0001.create_dataset('Yield', data=Y_Z005)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z005)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z005)

        Z0001 = Data.create_group('Z_0.06')
        MH = Z0001.create_dataset('Yield', data=Y_Z006)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z006)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z006)

        Z0001 = Data.create_group('Z_0.07')
        MH = Z0001.create_dataset('Yield', data=Y_Z007)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z007)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z007)

        Z0001 = Data.create_group('Z_0.08')
        MH = Z0001.create_dataset('Yield', data=Y_Z008)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z008)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z008)

        Z0001 = Data.create_group('Z_0.09')
        MH = Z0001.create_dataset('Yield', data=Y_Z009)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z009)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z009)

        Z0001 = Data.create_group('Z_0.10')
        MH = Z0001.create_dataset('Yield', data=Y_Z010)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z010)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z010)

    # i = 3
    # elem = 'N'
    # plt.figure()
    #
    # ax = plt.subplot(1, 1, 1)
    # ax.grid(True)
    #
    # plt.plot(Masses, Y_Z0007[i, :], '--o', label='$0.007Z_{\odot}$', color='tab:orange')
    # plt.plot(Masses, Y_Z0014[i, :], '--o', label='$0.014Z_{\odot}$', color='tab:blue')
    # plt.plot(Masses, Y_Z003[i, :], '--o', label='$0.03Z_{\odot}$', color='crimson')
    # plt.plot(Masses, Y_Z004[i, :], '-o', label='$0.04Z_{\odot}$', color='lightgreen')
    # plt.plot(Masses, Y_Z005[i, :], '-o', label='$0.05Z_{\odot}$', color='darkblue')
    # plt.plot(Masses, Y_Z006[i, :], '-o', label='$0.06Z_{\odot}$', color='darksalmon')
    # plt.plot(Masses, Y_Z007[i, :], '-o', label='$0.07Z_{\odot}$', color='tab:green')
    # plt.plot(Masses, Y_Z008[i, :], '-o', label='$0.08Z_{\odot}$', color='tab:purple')
    # plt.plot(Masses, Y_Z009[i, :], '-o', label='$0.09Z_{\odot}$', color='pink')
    # plt.plot(Masses, Y_Z010[i, :], '-o', label='$0.10Z_{\odot}$', color='grey')
    #
    #
    # if indx[i] >= 11: plt.yscale('log')
    # plt.xlim(1, 12)
    # plt.ylabel('Net Yields '+elem+' [M$_{\odot}$]')
    # plt.xlabel('Initial stellar mass [M$_{\odot}$]')
    # plt.legend(loc=[0.5, 0.7], labelspacing=0.2, handlelength=0.8,
    #            handletextpad=0.3, frameon=False, columnspacing=0.4,
    #            ncol=2, fontsize=8)
    # ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    # plt.savefig('./figures/Modified_AGB_Yield_tables_'+elem+'.png', dpi=200)

if __name__ == "__main__":
    modify_AGB_yield_tables()