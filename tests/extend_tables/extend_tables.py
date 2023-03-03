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


class make_yield_tables:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.0001, 0.001, 0.004, 0.007, 0.014, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        self.num_Z_bins = len(self.Z_bins)

        self.species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56]) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.5, 6., 7., 8., 9., 10., 11., 12.])
        self.num_mass_bins = len(self.mass_bins)

def extend_AGB_yield_tables():

    yields = make_yield_tables()

    # Read data
    with h5py.File('../../data/AGB.hdf5', 'r') as data_file:
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

    with h5py.File('../../data/Fishlock2014/AGB_Fishlock2014.hdf5', 'r') as data_file:
        Y_Z0001 = data_file["/Yields/Z_0.001/Yield"][:][:]
        Mej_Z0001 = data_file["/Yields/Z_0.001/Ejected_mass"][:]
        Mtot_Z0001 = data_file["/Yields/Z_0.001/Total_Metals"][:]

    with h5py.File('../../data/Karakas2010/AGB_Karakas2010.hdf5', 'r') as data_file:
        Y_Z00001 = data_file["/Yields/Z_0.0001/Yield"][:][:]
        Y_Z0004 = data_file["/Yields/Z_0.004/Yield"][:][:]
        Mej_Z00001 = data_file["/Yields/Z_0.0001/Ejected_mass"][:]
        Mej_Z0004 = data_file["/Yields/Z_0.004/Ejected_mass"][:]
        Mtot_Z00001 = data_file["/Yields/Z_0.0001/Total_Metals"][:]
        Mtot_Z0004 = data_file["/Yields/Z_0.004/Total_Metals"][:]


    elements = np.array(['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Magnesium','Iron','Strontium','Barium'])
    indx = np.array([0, 1, 2, 3, 4, 6, 10, 11, 12])

    # Write data to HDF5
    with h5py.File('./extendedAGB.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Net yields for AGB stars (in units of solar mass) are taken from Cinquegrana & Karakas (2021) "
        description += "for the metallicity range Z=0.04-0.1 and initial mass range 1-8 Msun. For the mass range 9-12Msun, these tables have been extrapolated. "
        description += "The net yields for AGB stars in the metallicity bins Z=0.03, 0.014 and 0.007 and initial mass range 1-8 Msun are taken from Karakas & Lugaro (2016). "
        description += "The net yields for metallicity bins Z=0.001 are taken from Fishlock et al. (2014) and for the metallicity bins Z=0.0001 and Z=0.004 are taken from Karakas et al. (2010). "
        description += "For the mass range 9-12Msun, we have interpolated the data from the tables of Doherty et al. (2014)."
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

        Z_names = ['Z_0.0001','Z_0.001','Z_0.004', 'Z_0.007', 'Z_0.014', 'Z_0.03', 'Z_0.04', 'Z_0.05', 'Z_0.06', 'Z_0.07', 'Z_0.08', 'Z_0.09', 'Z_0.10']
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        MH = data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Z_names = ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur',
                   'Calcium', 'Iron', 'Strontium', 'Barium']
        Element_names = np.string_(Z_names)
        dt = h5py.string_dtype(encoding='ascii')
        MH = data_file.create_dataset('Species_names', dtype=dt, data=Element_names)

        Reference = np.string_([
            'Cinquegrana & Karakas (2021) MNRAS, 3045C; '
            'Karakas & Lugaro (2016) ApJ, 825, 26K; '
            'Doherty, C L. Gil-Pons, P. Lau, H.H.B Lattanzio, J C and Siess, L. 2014. MNRAS, 437,195; '
            'Fishlock, C.; Karakas, A.; Lugaro, M.; Yong, D., 2014, ApJ, 797, 1, 44, 25; '
            'Karakas, A., et al., 2010, MNRAS, 477, 1, 421'
            ])
        MH = data_file.create_dataset('Reference', data=Reference)

        Data = data_file.create_group('Yields')

        Z0001 = Data.create_group('Z_0.0001')
        MH = Z0001.create_dataset('Yield', data=Y_Z00001)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z00001)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z00001)

        Z0001 = Data.create_group('Z_0.001')
        MH = Z0001.create_dataset('Yield', data=Y_Z0001)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z0001)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z0001)

        Z0001 = Data.create_group('Z_0.004')
        MH = Z0001.create_dataset('Yield', data=Y_Z0004)
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z0004)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z0004)

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

    for i in range(yields.num_species):

        plt.figure()

        ax = plt.subplot(1, 1, 1)
        ax.grid(True)

        plt.plot(yields.mass_bins, Y_Z00001[i, :], '-o', label='Z=0.0001')
        plt.plot(yields.mass_bins, Y_Z0001[i, :], '-o', label='Z=0.001')
        plt.plot(yields.mass_bins, Y_Z0004[i, :], '-o', label='Z=0.004')
        plt.plot(yields.mass_bins, Y_Z0007[i, :], '-o', label='Z=0.007')
        plt.plot(yields.mass_bins, Y_Z0014[i, :], '-o', label='Z=0.014')
        plt.plot(yields.mass_bins, Y_Z003[i, :], '-o', label='Z=0.003')
        plt.plot(yields.mass_bins, Y_Z005[i, :], '-o', label='Z=0.005')
        plt.plot(yields.mass_bins, Y_Z007[i, :], '-o', label='Z=0.007')
        plt.plot(yields.mass_bins, Y_Z010[i, :], '-o', label='Z=0.01')

        # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        if i == 0:
            plt.ylabel('Net Yields Hydrogen [M$_{\odot}$]')
            file_name = 'Yield_tables_Hydrogen.png'
            #plt.ylim([-15, 5])
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
            #plt.yscale('log')
            #plt.ylim([1e-2, 1e2])
        if i == 5:
            plt.ylabel('Net Yields Neon [M$_{\odot}$]')
            file_name = 'Yield_tables_Neon.png'
            plt.yscale('log')
            #plt.ylim([1e-3, 1e1])
        if i == 6:
            plt.ylabel('Net Yields Magnesium [M$_{\odot}$]')
            file_name = 'Yield_tables_Magnesium.png'
            #plt.yscale('log')
            #plt.ylim([1e-3, 1e1])
        if i == 7:
            plt.ylabel('Net Yields Silicon [M$_{\odot}$]')
            file_name = 'Yield_tables_Silicon.png'
            #plt.yscale('log')
            #plt.ylim([1e-2, 1e1])
        if i == 8:
            plt.ylabel('Net Yields S [M$_{\odot}$]')
            file_name = 'Yield_tables_S.png'
            #plt.ylim([-0.1, 0.3])
            # plt.yscale('log')
        if i == 9:
            plt.ylabel('Net Yields Ca [M$_{\odot}$]')
            file_name = 'Yield_tables_Ca.png'
            #plt.ylim([-0.1, 0.3])
            # plt.yscale('log')
        if i == 10:
            plt.ylabel('Net Yields Iron [M$_{\odot}$]')
            file_name = 'Yield_tables_Iron.png'
            #plt.ylim([-0.1, 0.3])
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

if __name__ == "__main__":

    extend_AGB_yield_tables()

