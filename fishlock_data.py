import numpy as np
import h5py
from utils import interpolate_data

class make_yield_tables:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.001])
        self.num_Z_bins = len(self.Z_bins)

        self.species = np.array(
            [1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56])  # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array(
            [1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.5, 6., 7., 8., 9.,
             10., 11., 12.])
        self.num_mass_bins = len(self.mass_bins)

        self.yields = np.zeros((self.num_species, self.num_mass_bins))

        self.file_ending = np.array(['_z001'])

        self.mass_ejected = None
        self.total_metals = None

    def read_data(self):

        data = np.loadtxt("./data/Fishlock2014/data2.txt", comments='#', usecols=(0,2,3,4,5,6,7))
        mass_ejected_list = np.zeros((self.num_species, self.num_mass_bins))

        for i, sp in enumerate(self.species):

            # Here yields are net yields, defined as the total mass (of each element)
            # that is expelled to the ISM during the star lifetime: Xi x Mej
            # minus the initial abundance of the element in the star: (Xi - X0) x Mej
            data_yields = data[data[:, 1] == sp, 2]
            initial_mass = data[data[:, 1] == sp, 0]
            mass_ejected = data[data[:, 1] == sp, 4]

            #if i == 3: data_yields *= 1.5 # Boosting Nitrogen!!

            self.yields[i, :] = interpolate_data(initial_mass, data_yields, self.mass_bins)
            mass_ejected_list[i, :] = interpolate_data(initial_mass, mass_ejected, self.mass_bins)

        self.mass_ejected = np.sum(mass_ejected_list, axis=0)
        self.total_metals = np.sum(self.yields[2:, :], axis=0)

    def complete_for_s_process(self):

        with h5py.File('./data/AGB.hdf5', 'r') as data_file:
            Y_Z007 = data_file["/Yields/Z_0.007/Yield"][:][:]
            Y_Z014 = data_file["/Yields/Z_0.014/Yield"][:][:]

        Zbins = np.array([0.007, 0.014])
        Sr_data_yields = np.zeros((len(self.mass_bins), 2))
        Ba_data_yields = np.zeros((len(self.mass_bins), 2))

        for j, m in enumerate(self.mass_bins):

            # Sr
            Sr_data_yields[:, 0] = Y_Z007[-2, :]
            Sr_data_yields[:, 1] = Y_Z014[-2, :]
            # Ba
            Ba_data_yields[:, 0] = Y_Z007[-1, :]
            Ba_data_yields[:, 1] = Y_Z014[-1, :]

            # For Z=0.001 we extrapolate ...
            # Sr
            alpha = (np.log10(Sr_data_yields[j, 1]) - np.log10(Sr_data_yields[j, 0])) / (Zbins[1] - Zbins[0])
            beta = np.log10(Sr_data_yields[j, 1]) - alpha * Zbins[1]
            self.yields[-2, j] = 2 * 10**(alpha * 0.001 + beta)

            # Ba
            alpha = (Ba_data_yields[j, 1] - Ba_data_yields[j, 0]) / (Zbins[1] - Zbins[0])
            beta = Ba_data_yields[j, 1] - alpha * Zbins[1]
            self.yields[-1, j] = alpha * 0.001 + beta

    def output_table(self):

        with h5py.File('./data/Fishlock2014/AGB_Fishlock2014.hdf5', 'w') as data_file:
            Header = data_file.create_group('Header')

            description = "Net yields for AGB stars (in units of solar mass) taken from Fishlock et al. (2014). "
            description += "These yields were calculated for the initial mass range 1-7 Msun. The range of yields in the "
            description += "mass range 1-12 Msun correspond to interpolation and extrapolation of the original yields."
            Header.attrs["Description"] = np.string_(description)

            contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
            contact += " website: camilacorrea.com"
            Header.attrs["Contact"] = np.string_(contact)

            mass_data = data_file.create_dataset('Masses', data=self.mass_bins)
            mass_data.attrs["Description"] = np.string_("Mass bins in units of Msolar")

            Z_data = data_file.create_dataset('Metallicities', data=self.Z_bins)
            Z_data.attrs["Description"] = np.string_("Metallicity bins")

            data_file.create_dataset('Number_of_metallicities', data=self.num_Z_bins)
            data_file.create_dataset('Number_of_masses', data=self.num_mass_bins)
            data_file.create_dataset('Number_of_species', data=np.array([13]))

            Z_names = ['Z_0.001']
            var = np.array(Z_names, dtype='S')
            dt = h5py.special_dtype(vlen=str)
            data_file.create_dataset('Yield_names', dtype=dt, data=var)

            Element_names = np.string_(['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur',
                       'Calcium', 'Iron', 'Strontium', 'Barium'])
            dt = h5py.string_dtype(encoding='ascii')
            data_file.create_dataset('Species_names', dtype=dt, data=Element_names)

            Reference = np.string_(['Fishlock, C.; Karakas, A.; Lugaro, M.; Yong, D., 2014, ApJ, 797, 1, 44, 25'])
            data_file.create_dataset('Reference', data=Reference)

            Data = data_file.create_group('Yields')

            data_group = Data.create_group('Z_0.001')
            data_group.create_dataset('Yield', data=self.yields)
            data_group.create_dataset('Ejected_mass', data=self.mass_ejected)
            data_group.create_dataset('Total_Metals', data=self.total_metals)



def make_Fishlock_table():

    Fishlock_yields = make_yield_tables()
    Fishlock_yields.read_data()
    Fishlock_yields.complete_for_s_process()
    Fishlock_yields.output_table()