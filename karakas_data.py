import numpy as np
import h5py
from utils import interpolate_data

class make_yields_info:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.007, 0.014, 0.03])
        self.num_Z_bins = len(self.Z_bins)

        self.species = np.array(
            [1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56])  # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array(
            [1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.5, 6., 7., 8.])
        self.num_mass_bins = len(self.mass_bins)

        self.file_ending = np.array(['_z007', '_z014', '_z030'])

def Doherty_2014_data(Z_bins):

    file = './data/Doherty2014/AGB_Doherty2014.hdf5'
    with h5py.File(file, 'r') as data_file:
        masses = data_file["Masses"][:]
        z_bins = data_file["Metallicities"][:]
        yields = data_file["/Yields/"+Z_bins+"/Yield"][:][:]
        mej = data_file["/Yields/"+Z_bins+"/Ejected_mass"][:]
        mtot = data_file["/Yields/"+Z_bins+"/Total_Metals"][:]

    return masses, z_bins, yields, mej, mtot


def make_new_data_range(yields_info, z_index):

    masses, z_bins, yield_0004, mej_0004, _ = Doherty_2014_data('Z_0.004')
    _, _, yield_0008, mej_0008, _ = Doherty_2014_data('Z_0.008')
    _, _, yield_0020, mej_0020, _ = Doherty_2014_data('Z_0.02')

    num_mass_bins = len(masses)

    net_yields = np.zeros((yields_info.num_species, num_mass_bins))
    mass_ejected = np.zeros(num_mass_bins)

    for i, m in enumerate(masses):

        mej = np.zeros(3)
        mej[0] = mej_0004[i]
        mej[1] = mej_0008[i]
        mej[2] = mej_0020[i]

        mej_interpolated = interpolate_data(z_bins, mej, yields_info.Z_bins)
        mass_ejected[i] = mej_interpolated[z_index]

        for j in range(yields_info.num_species):

            species = np.zeros(3)
            species[0] = yield_0004[j, i]
            species[1] = yield_0008[j, i]
            species[2] = yield_0020[j, i]

            yield_interpolated = interpolate_data(z_bins, species, yields_info.Z_bins)
            net_yields[j, i] = yield_interpolated[z_index]

    total_metals = np.sum(net_yields[2:, :], axis=0)
    return net_yields, mass_ejected, total_metals


def calculate_mass_ejected(yields_info, index):

    data = np.loadtxt("./data/Karakas2016/data" + yields_info.file_ending[index] + ".txt", comments='#')
    initial_mass_data = data[0, :]
    final_mass_data = data[1, :]

    final_mass = interpolate_data(initial_mass_data, final_mass_data, yields_info.mass_bins)
    mass_ejected = yields_info.mass_bins - final_mass
    return mass_ejected

def calculate_net_yields(yields_info, index):

    data = np.loadtxt("./data/Karakas2016/data" + yields_info.file_ending[index] + ".txt", comments='#')
    initial_mass = data[0, :]
    final_mass = data[1, :]
    mass_ejected = initial_mass - final_mass

    data = np.loadtxt("./data/Karakas2016/yield" + yields_info.file_ending[index] + ".txt", comments='#',
                      usecols=[1, 2, 3, 4, 5, 6])

    initial_data = np.loadtxt("./data/Karakas2016/initial" + yields_info.file_ending[index] + ".txt", comments='#',
                              usecols=[1, 2, 3, 4, 5, 6])

    yields_interpolated = np.zeros((yields_info.num_species, yields_info.num_mass_bins))

    for i, sp in enumerate(yields_info.species):

        # Here yields are defined as the total mass (of each element)
        # that is expelled to the ISM during the star lifetime: Xi x Mej
        data_yields = data[data[:, 0] == sp, 5]

        # We next substract the initial abundance of an element multiplied by mass ejected.
        # To obtain the next yields: (Xi - X0) x Mej
        data_yields -= initial_data[initial_data[:, 0] == sp, 5] * mass_ejected

        yields_interpolated[i, :] = interpolate_data(initial_mass, data_yields, yields_info.mass_bins)

        if i == yields_info.num_species - 1 :
            yields_interpolated[i, :] *= (1. / 2.)  # Here I apply a correction factor of 2 for Barium

        #if (i == 3) & (index == 0):
        #    yields_interpolated *= 1.5 # Here I apply a boost factor of 2 for Nitrogen

    select = np.where(yields_info.mass_bins == 4.0)[0]  # Don't remember why I did this correction
    yields_interpolated[10, select] = yields_interpolated[10, select - 2]
    return yields_interpolated


def extrapolate_karakas_data_with_doherty(yields_info, net_yields, mass_ejected, z_index):

    mass_range = np.array([9., 10., 11., 12.])
    num_mass_bins = len(mass_range)

    yields_doherty, mej_doherty, _ = make_new_data_range(yields_info, z_index)
    masses_doherty, _, _, _, _ = Doherty_2014_data('Z_0.004')

    new_yield_data = np.zeros((yields_info.num_species, num_mass_bins))

    new_mej_data = interpolate_data(masses_doherty, mej_doherty, mass_range)

    for k in range(0, yields_info.num_species):
        new_yield_data[k, :] = interpolate_data(masses_doherty, yields_doherty[k, :], mass_range)

        if k >= yields_info.num_species - 2: # I'm extrapolating here for the s-process elements
            new_yield_data[k, :] = interpolate_data(yields_info.mass_bins, net_yields[k, :], mass_range)

    new_mass_range = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.,
                               3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.,
                               5.5, 6., 7., 8., 9., 10., 11., 12.])


    num_mass_bins = len(new_mass_range)
    new_net_yields = np.zeros((yields_info.num_species, num_mass_bins))

    new_net_yields[:, 0:yields_info.num_mass_bins] = net_yields[:, 0:yields_info.num_mass_bins]
    new_net_yields[:, yields_info.num_mass_bins:yields_info.num_mass_bins+num_mass_bins] = new_yield_data[:, :]

    new_mej = np.zeros(num_mass_bins)
    new_mej[0:yields_info.num_mass_bins] = mass_ejected
    new_mej[yields_info.num_mass_bins:] = new_mej_data

    return new_net_yields, new_mej


def make_Karakas_table():

    yields_info = make_yields_info()

    with h5py.File('./data/Karakas2016/AGB_Karakas2016.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Net yields for AGB stars (in units of solar mass) taken from Karakas & Lugaro (2016). "
        description += "These yields were calculated for the initial mass range 1-8 Msun. The range of yields in the "
        description += "mass range 9-12 Msun correspond to interpolation from the super-AGB stars work from Doherty et al. (2014)."
        Header.attrs["Description"] = np.string_(description)

        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"] = np.string_(contact)

        Z_data = data_file.create_dataset('Metallicities', data=yields_info.Z_bins)
        Z_data.attrs["Description"] = np.string_("Metallicity bins")

        data_file.create_dataset('Number_of_metallicities', data=yields_info.num_Z_bins)
        data_file.create_dataset('Number_of_species', data=np.array([13]))

        Z_names = ['Z_0.007', 'Z_0.014', 'Z_0.03']
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Element_names = np.string_(
            ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur',
             'Calcium', 'Iron', 'Strontium', 'Barium'])
        dt = h5py.string_dtype(encoding='ascii')
        data_file.create_dataset('Species_names', dtype=dt, data=Element_names)

        Reference = np.string_(['Karakas, A. I. & Lugaro, M. (2016) ApJ, 825, 26; Doherty, C. L. Gil-Pons, P. Lau, H. H. B Lattanzio, J. C. and Siess, L. (2014) MNRAS, 437, 195'])
        data_file.create_dataset('Reference', data=Reference)

        Data = data_file.create_group('Yields')

        for i, Zi, in enumerate(yields_info.Z_bins):

            mass_ejected = calculate_mass_ejected(yields_info, i)
            net_yields = calculate_net_yields(yields_info, i)

            net_yields, mass_ejected = extrapolate_karakas_data_with_doherty(yields_info, net_yields, mass_ejected, i)
            total_metals = np.sum(net_yields[2:, :], axis=0)

            data_group = Data.create_group(Z_names[i])
            data_group.create_dataset('Yield', data=net_yields)
            data_group.create_dataset('Ejected_mass', data=mass_ejected)
            data_group.create_dataset('Total_Metals', data=total_metals)

        # Updating full mass range...
        new_mass_range = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.,
                                   3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.,
                                   5.5, 6., 7., 8., 9., 10., 11., 12.])
        yields_info.mass_bins = new_mass_range
        yields_info.num_mass_bins = len(new_mass_range)

        mass_data = data_file.create_dataset('Masses', data=yields_info.mass_bins)
        mass_data.attrs["Description"] = np.string_("Mass bins in units of Msolar")
        data_file.create_dataset('Number_of_masses', data=yields_info.num_mass_bins)






