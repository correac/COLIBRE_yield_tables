import numpy as np
import h5py
from scipy import interpolate
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

def plot_Fishlock(index):
    # Write data to HDF5
    with h5py.File('../Fishlock2014/AGB_Fishlock2014.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Y_Z0007 = data_file["/Yields/Z_0.001/Yield"][:][:]

    plt.plot(Masses, Y_Z0007[index, :], '-o', color='tab:green', label='Fishlock+, Z=0.001')


def plot_karakas(index):
    # Write data to HDF5
    with h5py.File('../AGB.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Y_Z0007 = data_file["/Yields/Z_0.007/Yield"][:][:]
        Y_Z0014 = data_file["/Yields/Z_0.014/Yield"][:][:]
        Y_Z0030 = data_file["/Yields/Z_0.03/Yield"][:][:]

    plt.plot(Masses, Y_Z0007[index, :], '-o', color='tab:blue',label='Karakas+, Z=0.007')
    plt.plot(Masses, Y_Z0014[index, :], '-o', color='crimson',label='Karakas+, Z=0.014')
    plt.plot(Masses, Y_Z0030[index, :], '-o', color='grey',label='Karakas+, Z=0.03')

class make_yield_tables:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.0001, 0.004])
        self.num_Z_bins = len(self.Z_bins)

        self.species = np.array(
            [1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56])  # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array(
            [1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.5, 6., 7., 8., 9.,
             10., 11., 12.])
        self.num_mass_bins = len(self.mass_bins)

        self.Z0001_yields = np.zeros((self.num_species, self.num_mass_bins))
        self.Z004_yields = np.zeros((self.num_species, self.num_mass_bins))

        self.file_ending = np.array(['_z0001','_z004'])

        self.Z0001_mass_ejected = np.zeros(self.num_mass_bins)
        self.Z0001_total_metals = None
        self.Z004_mass_ejected = np.zeros(self.num_mass_bins)
        self.Z004_total_metals = None

    def read_data(self):

        species_list = np.array(['p','He','C','N','O','Ne','Mg','Si','S','Ca','Fe','g','g'])

        for k in range(self.num_Z_bins):  # two loops, for Z=0.0001 and Z=0.004

            data = np.loadtxt("./data.txt", comments='#', usecols=(0, 2, 3, 5, 6, 7, 8))
            species = np.loadtxt("./data.txt", comments='#', usecols=(4), dtype='str')
            z_bins = np.where(data[:,1] == self.Z_bins[k])[0]
            data = data[z_bins,:]
            species = species[z_bins]

            for i in range(0, self.num_species):

                sp = species_list[i]
                select_species = np.where(species==sp)[0]
                if len(select_species)==0:continue # Element not found

                # Here yields are net yields, defined as the total mass (of each element)
                # that is expelled to the ISM during the star lifetime: Xi x Mej
                # minus the initial abundance of the element in the star: (Xi - X0) x Mej
                data_yields_prev = data[select_species, 4]
                initial_mass, index = np.unique(data[select_species, 0], return_index=True)
                index = np.append(index,len(data[select_species, 0]))
                final_mass = np.unique(data[select_species, 2])
                mass_ejected = initial_mass - final_mass
                data_yields = np.zeros(len(initial_mass))
                for ii in range(len(data_yields)):data_yields[ii] = np.sum(data_yields_prev[index[ii]:index[ii+1]])

                if i == 3: data_yields *= 2  # Boosting Nitrogen

                f = interpolate.interp1d(initial_mass, data_yields)
                g = interpolate.interp1d(initial_mass, mass_ejected)

                for j, m in enumerate(self.mass_bins):

                    if (m >= np.min(initial_mass)) & (m <= np.max(initial_mass)):
                        f_m = f(m)
                        g_m = g(m)
                    elif (m > np.max(initial_mass)):
                        f_m = data_yields[-1] * m / np.max(initial_mass)
                        g_m = mass_ejected[-1] * m / np.max(initial_mass)
                    elif (m < np.min(initial_mass)):
                        f_m = data_yields[0] * m / np.min(initial_mass)
                        g_m = mass_ejected[0] * m / np.min(initial_mass)

                    if k ==0:
                        self.Z0001_yields[i, j] = f_m
                        if i == 0:
                            self.Z0001_mass_ejected[j] = g_m
                    if k ==1:
                        self.Z004_yields[i, j] = f_m
                        if i==0:
                            self.Z004_mass_ejected[j] = g_m

        self.Z0001_total_metals = np.sum(self.Z0001_yields[2:, :], axis=0)
        self.Z004_total_metals = np.sum(self.Z004_yields[2:, :], axis=0)

    def complete_for_s_process(self):

        # with h5py.File('../Fishlock2014/AGB_Fishlock2014.hdf5', 'r') as data_file:
        #     Y_Z001 = data_file["/Yields/Z_0.001/Yield"][:][:]
        with h5py.File('../AGB.hdf5', 'r') as data_file:
            Y_Z007 = data_file["/Yields/Z_0.007/Yield"][:][:]
            Y_Z014 = data_file["/Yields/Z_0.014/Yield"][:][:]

        Zbins = np.array([0.007, 0.014])
        Sr_data_yields = np.zeros((len(self.mass_bins), 2))
        Ba_data_yields = np.zeros((len(self.mass_bins), 2))

        for j, m in enumerate(self.mass_bins):

            # # For Z=0.004 we interpolate ...
            # Sr
            Sr_data_yields[:, 0] = Y_Z007[-2, :]
            Sr_data_yields[:, 1] = Y_Z014[-2, :]
            # f_Sr = interpolate.interp1d(Zbins, Sr_data_yields[j, :])
            # self.Z004_yields[-2,j] = f_Sr(0.004)
            #
            # # Ba
            Ba_data_yields[:, 0] = Y_Z007[-1, :]
            Ba_data_yields[:, 1] = Y_Z014[-1, :]
            # f_Ba = interpolate.interp1d(Zbins, Ba_data_yields[j, :])
            # self.Z004_yields[-1, j] = f_Ba(0.004)

            # For Z=0.0001 we extrapolate ...
            # Sr
            # f(x1) = a * x1 + b
            # f(x2) = a * x2 + b
            # b = f(x1) - a x1
            # f(x2) = a (x2 - x1) + f(x1)
            # a = f(x2) - f(x1) / (x2 - x1)
            alpha = (np.log10(Sr_data_yields[j, 1]) - np.log10(Sr_data_yields[j, 0])) / (Zbins[1] - Zbins[0])
            beta = np.log10(Sr_data_yields[j, 1]) - alpha * Zbins[1]
            self.Z0001_yields[-2, j] = 2 * 10**(alpha * 0.0001 + beta)
            self.Z004_yields[-2, j] = 2 * 10**(alpha * 0.004 + beta)
            # self.Z0001_yields[-2, j] = Sr_data_yields[j, 0] * 0.0001 / 0.007
            # self.Z004_yields[-2, j] = Sr_data_yields[j, 0] * 0.004 / 0.007

            # Ba
            alpha = (Ba_data_yields[j, 1] - Ba_data_yields[j, 0]) / (Zbins[1] - Zbins[0])
            beta = Ba_data_yields[j, 1] - alpha * Zbins[1]
            self.Z0001_yields[-1, j] = alpha * 0.0001 + beta
            self.Z004_yields[-1, j] = alpha * 0.004 + beta
            # self.Z0001_yields[-1, j] = Ba_data_yields[j, 0] * 0.0001 / 0.007
            # self.Z004_yields[-1, j] = Ba_data_yields[j, 0] * 0.004 / 0.007


    def output_table(self):

        with h5py.File('./AGB_Karakas2010.hdf5', 'w') as data_file:
            Header = data_file.create_group('Header')

            description = "Net yields for AGB stars (in units of solar mass) taken from Karakas et al. (2010). "
            description += "These yields were calculated for the initial mass range 1-6 Msun. The range of yields in the "
            description += "mass range 1-8 Msun correspond to interpolation and extrapolation of the original yields."
            Header.attrs["Description"] = np.string_(description)

            contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
            contact += " website: camilacorrea.com"
            Header.attrs["Contact"] = np.string_(contact)

            MH = data_file.create_dataset('Masses', data=self.mass_bins)
            MH.attrs["Description"] = np.string_("Mass bins in units of Msolar")

            MH = data_file.create_dataset('Metallicities', data=self.Z_bins)

            MH.attrs["Description"] = np.string_("Metallicity bins")

            MH = data_file.create_dataset('Number_of_metallicities', data=self.num_Z_bins)

            MH = data_file.create_dataset('Number_of_masses', data=self.num_mass_bins)

            MH = data_file.create_dataset('Number_of_species', data=np.array([13]))

            Z_names = ['Z_0.0001','Z_0.004']
            var = np.array(Z_names, dtype='S')
            dt = h5py.special_dtype(vlen=str)
            MH = data_file.create_dataset('Yield_names', dtype=dt, data=var)

            Z_names = ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur',
                       'Calcium', 'Iron', 'Strontium', 'Barium']
            Element_names = np.string_(Z_names)
            dt = h5py.string_dtype(encoding='ascii')
            MH = data_file.create_dataset('Species_names', dtype=dt, data=Element_names)

            Reference = np.string_(['Karakas, A., et al., 2010, MNRAS, 477, 1, 421'])
            MH = data_file.create_dataset('Reference', data=Reference)

            Data = data_file.create_group('Yields')

            Z0001 = Data.create_group('Z_0.0001')
            MH = Z0001.create_dataset('Yield', data=self.Z0001_yields)
            MH = Z0001.create_dataset('Ejected_mass', data=self.Z0001_mass_ejected)
            MH = Z0001.create_dataset('Total_Metals', data=self.Z0001_total_metals)

            Z0001 = Data.create_group('Z_0.004')
            MH = Z0001.create_dataset('Yield', data=self.Z004_yields)
            MH = Z0001.create_dataset('Ejected_mass', data=self.Z004_mass_ejected)
            MH = Z0001.create_dataset('Total_Metals', data=self.Z004_total_metals)

    def plot_tables(self):

        for i in range(self.num_species):

            plt.figure()

            ax = plt.subplot(1, 1, 1)
            ax.grid(True)

            plt.plot(self.mass_bins, self.Z0001_yields[i, :], '-o', color='tab:orange',label='extrapolation, Z=0.0001')
            plt.plot(self.mass_bins, self.Z004_yields[i, :], '-o', color='tab:purple',label='extrapolation, Z=0.004')
            plot_Fishlock(i)
            plot_karakas(i)

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

    Karakas_yields = make_yield_tables()
    Karakas_yields.read_data()
    Karakas_yields.complete_for_s_process()
    Karakas_yields.output_table()
    Karakas_yields.plot_tables()