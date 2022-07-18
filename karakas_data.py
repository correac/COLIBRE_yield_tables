import numpy as np
import h5py
from scipy import interpolate


class read_doherty_data:

    def __init__(self):
        file = './data/Doherty2014/AGB_Doherty2014.hdf5'
        with h5py.File(file, 'r') as data_file:
            self.masses = data_file["Masses"][:]
            self.z_bins = data_file["Metallicities"][:]
            self.yield_0004 = data_file["/Yields/Z_0.004/Yield"][:][:]
            self.yield_0008 = data_file["/Yields/Z_0.008/Yield"][:][:]
            self.yield_002 = data_file["/Yields/Z_0.02/Yield"][:][:]
            self.mej_Z0004 = data_file["/Yields/Z_0.004/Ejected_mass"][:]
            self.mej_Z0008 = data_file["/Yields/Z_0.008/Ejected_mass"][:]
            self.mej_Z002 = data_file["/Yields/Z_0.02/Ejected_mass"][:]
            self.mtot_Z0004 = data_file["/Yields/Z_0.004/Total_Metals"][:]
            self.mtot_Z0008 = data_file["/Yields/Z_0.008/Total_Metals"][:]
            self.mtot_Z002 = data_file["/Yields/Z_0.02/Total_Metals"][:]

        self.yield_z0007 = None
        self.yield_z0014 = None
        self.yield_z003 = None

        self.Z007_mass_ejected = None
        self.Z014_mass_ejected = None
        self.Z030_mass_ejected = None

        self.Z007_total_metals = None
        self.Z014_total_metals = None
        self.Z030_total_metals = None

    def calculate_total_metals(self):
        self.Z007_total_metals = np.sum( self.yield_z0007[2:,:], axis = 0)
        self.Z014_total_metals = np.sum( self.yield_z0014[2:,:], axis = 0)
        self.Z030_total_metals = np.sum( self.yield_z003[2:,:], axis = 0)

        # self.Z007_total_metals = - self.yield_z0007[0, :] - self.yield_z0007[1, :]
        # self.Z014_total_metals = - self.yield_z0014[0, :] - self.yield_z0014[1, :]
        # self.Z030_total_metals = - self.yield_z003[0, :] - self.yield_z003[1, :]


class make_yield_tables:

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

        self.Z007_yields = np.zeros((self.num_species, self.num_mass_bins))
        self.Z014_yields = np.zeros((self.num_species, self.num_mass_bins))
        self.Z030_yields = np.zeros((self.num_species, self.num_mass_bins))

        self.file_ending = np.array(['_z007', '_z014', '_z030'])

        self.Z007_mass_ejected = None
        self.Z014_mass_ejected = None
        self.Z030_mass_ejected = None

        self.Z007_total_metals = None
        self.Z014_total_metals = None
        self.Z030_total_metals = None

    def add_mass_data(self, mass_ejected, final_mass, z_bin):

        if z_bin == 0.007:
            self.Z007_mass_ejected = mass_ejected
            self.Z007_final_mass = final_mass

        if z_bin == 0.014:
            self.Z014_mass_ejected = mass_ejected
            self.Z014_final_mass = final_mass

        if z_bin == 0.030:
            self.Z030_mass_ejected = mass_ejected
            self.Z030_final_mass = final_mass

    def calculate_total_metals(self):

        self.Z007_total_metals = np.sum( self.Z007_yields[2:,:], axis = 0)
        self.Z014_total_metals = np.sum( self.Z014_yields[2:,:], axis = 0)
        self.Z030_total_metals = np.sum( self.Z030_yields[2:,:], axis = 0)

        # self.Z007_total_metals = - self.Z007_yields[0, :] - self.Z007_yields[1, :]
        # self.Z014_total_metals = - self.Z014_yields[0, :] - self.Z014_yields[1, :]
        # self.Z030_total_metals = - self.Z030_yields[0, :] - self.Z030_yields[1, :]


def make_new_data_range(yields_doherty):
    Z_bins = np.array([0.007, 0.014, 0.03])

    num_mass_bins = len(yields_doherty.masses)
    species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56])
    num_species = len(species)

    yield_0007 = np.zeros((num_species, num_mass_bins))
    yield_0014 = np.zeros((num_species, num_mass_bins))
    yield_003 = np.zeros((num_species, num_mass_bins))

    for i, m in enumerate(yields_doherty.masses):

        for j in range(num_species):

            species = np.zeros(3)
            species[0] = yields_doherty.yield_0004[j, i]
            species[1] = yields_doherty.yield_0008[j, i]
            species[2] = yields_doherty.yield_002[j, i]
            f = interpolate.interp1d(yields_doherty.z_bins, species)

            for k, z in enumerate(Z_bins):
                if (z >= np.min(yields_doherty.z_bins)) & (z <= np.max(yields_doherty.z_bins)):
                    fi = f(z)
                elif (z > np.max(yields_doherty.z_bins)):
                    fi = species[2] * z / np.max(yields_doherty.z_bins)
                elif (z < np.min(yields_doherty.z_bins)):
                    fi = species[0] * z / np.min(yields_doherty.z_bins)

                if z == 0.007: yield_0007[j, i] = fi
                if z == 0.014: yield_0014[j, i] = fi
                if z == 0.03: yield_003[j, i] = fi

    yields_doherty.yield_z0007 = yield_0007
    yields_doherty.yield_z0014 = yield_0014
    yields_doherty.yield_z003 = yield_003

    mej_0007 = np.zeros(num_mass_bins)
    mej_0014 = np.zeros(num_mass_bins)
    mej_003 = np.zeros(num_mass_bins)

    for i, m in enumerate(yields_doherty.masses):

        mej = np.zeros(3)
        mej[0] = yields_doherty.mej_Z0004[i]
        mej[1] = yields_doherty.mej_Z0008[i]
        mej[2] = yields_doherty.mej_Z002[i]
        f = interpolate.interp1d(yields_doherty.z_bins, mej)

        for k, z in enumerate(Z_bins):
            if (z >= np.min(yields_doherty.z_bins)) & (z <= np.max(yields_doherty.z_bins)):
                fi = f(z)
            elif (z > np.max(yields_doherty.z_bins)):
                fi = mej[2] * z / np.max(yields_doherty.z_bins)
            elif (z < np.min(yields_doherty.z_bins)):
                fi = mej[0] * z / np.min(yields_doherty.z_bins)

            if z == 0.007: mej_0007[i] = fi
            if z == 0.014: mej_0014[i] = fi
            if z == 0.03: mej_003[i] = fi

    yields_doherty.Z007_mass_ejected = mej_0007
    yields_doherty.Z014_mass_ejected = mej_0014
    # yields_doherty.Z030_mass_ejected = mej_003
    yields_doherty.Z030_mass_ejected = yields_doherty.mej_Z002.copy()

    yields_doherty.calculate_total_metals()
    return


def calculate_mass_ejected(yields, index, z_bin):
    data = np.loadtxt("./data/Karakas2016/data" + yields.file_ending[index] + ".txt", comments='#')
    initial_mass_data = data[0, :]
    final_mass_data = data[1, :]
    f = interpolate.interp1d(initial_mass_data, final_mass_data)

    mass_ejected = np.zeros(yields.num_mass_bins)
    final_mass = np.zeros(yields.num_mass_bins)
    for i, m in enumerate(yields.mass_bins):

        if (m >= np.min(initial_mass_data)) & (m <= np.max(initial_mass_data)):
            fi_m = f(m)
        elif (m >= np.max(initial_mass_data)):
            fi_m = np.max(final_mass_data) * m / np.max(initial_mass_data)
        elif (m <= np.min(initial_mass_data)):
            fi_m = np.min(final_mass_data) * m / np.min(initial_mass_data)

        f_m = m - fi_m
        mass_ejected[i] = f_m
        final_mass[i] = fi_m

    yields.add_mass_data(mass_ejected, final_mass, z_bin)
    return


def calculate_net_yields(yields, index, z_bin):
    data = np.loadtxt("./data/Karakas2016/data" + yields.file_ending[index] + ".txt", comments='#')
    initial_mass = data[0, :]
    final_mass = data[1, :]
    mass_ejected = initial_mass - final_mass

    data = np.loadtxt("./data/Karakas2016/yield" + yields.file_ending[index] + ".txt", comments='#',
                      usecols=[1, 2, 3, 4, 5, 6])

    initial_data = np.loadtxt("./data/Karakas2016/initial" + yields.file_ending[index] + ".txt", comments='#',
                              usecols=[1, 2, 3, 4, 5, 6])

    for i in range(0, yields.num_species):

        sp = yields.species[i]
        # Here yields are defined as the total mass (of each element)
        # that is expelled to the ISM during the star lifetime: Xi x Mej
        data_yields = data[data[:, 0] == sp, 5]

        # We next substract the initial abundance of an element multiplied by mass ejected.
        # To obtain the next yields: (Xi - X0) x Mej
        data_yields -= initial_data[initial_data[:, 0] == sp, 5] * mass_ejected

        f = interpolate.interp1d(initial_mass, data_yields)

        if i == yields.num_species - 1:
            boost = (1. / 2.)  # I will be applying a correction factor of 2 for Barium
        else:
            boost = 1.0

        for j, m in enumerate(yields.mass_bins):

            if (m >= np.min(initial_mass)) & (m <= np.max(initial_mass)):
                f_m = f(m) * boost
            elif (m > np.max(initial_mass)):
                f_m = data_yields[-1] * m / np.max(initial_mass) * boost
            elif (m < np.min(initial_mass)):
                f_m = data_yields[0] * m / np.min(initial_mass) * boost

            if z_bin == 0.007: yields.Z007_yields[i, j] = f_m
            if z_bin == 0.014: yields.Z014_yields[i, j] = f_m
            if z_bin == 0.030: yields.Z030_yields[i, j] = f_m

    select = np.where(yields.mass_bins == 4.0)[0]
    yields.Z030_yields[10, select] = yields.Z030_yields[10, select - 2]

    return


def extrapolate_s_process_data(yields, Z_bin):
    mass_range = np.array([9., 10., 11., 12.])
    num_mass_bins = len(mass_range)

    new_yield_data = np.zeros((yields.num_species, num_mass_bins))

    for k in range(yields.num_species - 2, yields.num_species):

        if Z_bin == 0.007: data = yields.Z007_yields[k, 0:len(yields.mass_bins)]
        if Z_bin == 0.014: data = yields.Z014_yields[k, 0:len(yields.mass_bins)]
        if Z_bin == 0.03: data = yields.Z030_yields[k, 0:len(yields.mass_bins)]

        f = interpolate.interp1d(yields.mass_bins, data)

        for j, m in enumerate(mass_range):
            if (m >= np.min(yields.mass_bins)) & (m <= np.max(yields.mass_bins)):
                f_m = f(m)
            elif (m > np.max(yields.mass_bins)):
                f_m = data[-1] * m / np.max(yields.mass_bins)
            elif (m < np.min(yields.mass_bins)):
                f_m = data[0] * m / np.min(yields.mass_bins)

            if Z_bin == 0.007: yields.Z007_yields[k, len(yields.mass_bins) + j] = f_m.copy()
            if Z_bin == 0.014: yields.Z014_yields[k, len(yields.mass_bins) + j] = f_m.copy()
            if Z_bin == 0.03: yields.Z030_yields[k, len(yields.mass_bins) + j] = f_m.copy()


def extrapolate_yield_data(yields, yields_doherty, Z_bin):
    mass_range = np.array([9., 10., 11., 12.])
    num_mass_bins = len(mass_range)

    new_yield_data = np.zeros((yields.num_species, num_mass_bins))

    for k in range(0, yields.num_species):

        if Z_bin == 0.007: data = yields_doherty.yield_z0007[k, :]
        if Z_bin == 0.014: data = yields_doherty.yield_z0014[k, :]
        if Z_bin == 0.03: data = yields_doherty.yield_z003[k, :]

        f = interpolate.interp1d(yields_doherty.masses, data)

        for j, m in enumerate(mass_range):
            if (m >= np.min(yields_doherty.masses)) & (m <= np.max(yields_doherty.masses)):
                f_m = f(m)
            elif (m > np.max(yields_doherty.masses)):
                f_m = data[-1] * m / np.max(yields_doherty.masses)
            elif (m < np.min(yields_doherty.masses)):
                f_m = data[0] * m / np.min(yields_doherty.masses)

            new_yield_data[k, j] = f_m

    new_mass_range = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.,
                               3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.,
                               5.5, 6., 7., 8., 9., 10., 11., 12.])
    num_mass_bins = len(new_mass_range)
    new_data = np.zeros((yields.num_species, num_mass_bins))
    for k in range(0, yields.num_species):

        if Z_bin == 0.007:
            new_data[k, 0:yields.num_mass_bins] = yields.Z007_yields[k, :]
        if Z_bin == 0.014:
            new_data[k, 0:yields.num_mass_bins] = yields.Z014_yields[k, :]
        if Z_bin == 0.03:
            new_data[k, 0:yields.num_mass_bins] = yields.Z030_yields[k, :]

        new_data[k, yields.num_mass_bins:] = new_yield_data[k, :]

    if Z_bin == 0.007: yields.Z007_yields = new_data.copy()
    if Z_bin == 0.014: yields.Z014_yields = new_data.copy()
    if Z_bin == 0.03: yields.Z030_yields = new_data.copy()


def extrapolate_mej_data(yields, yields_doherty, Z_bin):
    mass_range = np.array([9., 10., 11., 12.])
    num_mass_bins = len(mass_range)

    new_mej_data = np.zeros(num_mass_bins)

    if Z_bin == 0.007: data = yields_doherty.Z007_mass_ejected[:].copy()
    if Z_bin == 0.014: data = yields_doherty.Z014_mass_ejected[:].copy()
    if Z_bin == 0.03: data = yields_doherty.Z030_mass_ejected[:].copy()

    f = interpolate.interp1d(yields_doherty.masses, data)

    for j, m in enumerate(mass_range):

        if (m >= np.min(yields_doherty.masses)) & (m <= np.max(yields_doherty.masses)):
            f_m = f(m)
        elif (m > np.max(yields_doherty.masses)):
            f_m = data[-1] * m / np.max(yields_doherty.masses)

        new_mej_data[j] = f_m

    new_mass_range = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.,
                               3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.,
                               5.5, 6., 7., 8., 9., 10., 11., 12.])
    num_mass_bins = len(new_mass_range)
    new_data = np.zeros(num_mass_bins)
    if Z_bin == 0.007:
        new_data[0:yields.num_mass_bins] = yields.Z007_mass_ejected.copy()
    if Z_bin == 0.014:
        new_data[0:yields.num_mass_bins] = yields.Z014_mass_ejected.copy()
    if Z_bin == 0.03:
        new_data[0:yields.num_mass_bins] = yields.Z030_mass_ejected.copy()

    new_data[yields.num_mass_bins:] = new_mej_data[:]

    if Z_bin == 0.007: yields.Z007_mass_ejected = new_data.copy()
    if Z_bin == 0.014: yields.Z014_mass_ejected = new_data.copy()
    if Z_bin == 0.03: yields.Z030_mass_ejected = new_data.copy()


def output_Karakas_table(yields):

    with h5py.File('./data/Karakas2016/AGB_Karakas2016.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Net yields for AGB stars (in units of solar mass) taken from Karakas & Lugaro (2016). "
        description += "These yields were calculated for the initial mass range 1-8 Msun. The range of yields in the "
        description += "mass range 9-12 Msun correspond to interpolation from the super-AGB stars work from Doherty et al. (2014)."
        Header.attrs["Description"] = np.string_(description)

        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"] = np.string_(contact)

        MH = data_file.create_dataset('Masses', data=yields.mass_bins)
        MH.attrs["Description"] = np.string_("Mass bins in units of Msolar")

        MH = data_file.create_dataset('Metallicities', data=yields.Z_bins)

        MH.attrs["Description"] = np.string_("Metallicity bins")

        MH = data_file.create_dataset('Number_of_metallicities', data=yields.num_Z_bins)

        MH = data_file.create_dataset('Number_of_masses', data=yields.num_mass_bins)

        MH = data_file.create_dataset('Number_of_species', data=np.array([13]))

        Z_names = ['Z_0.007', 'Z_0.014', 'Z_0.03']
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        MH = data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Z_names = ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur',
                   'Calcium', 'Iron', 'Strontium', 'Barium']
        Element_names = np.string_(Z_names)
        dt = h5py.string_dtype(encoding='ascii')
        MH = data_file.create_dataset('Species_names', dtype=dt, data=Element_names)

        Reference = np.string_(['Karakas, A. I. & Lugaro, M. (2016) ApJ, 825, 26; Doherty, C. L. Gil-Pons, P. Lau, H. H. B Lattanzio, J. C. and Siess, L. (2014) MNRAS, 437, 195'])
        MH = data_file.create_dataset('Reference', data=Reference)

        Data = data_file.create_group('Yields')

        Z0001 = Data.create_group('Z_0.03')
        MH = Z0001.create_dataset('Yield', data=yields.Z030_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z030_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z030_total_metals)

        Z001 = Data.create_group('Z_0.014')
        MH = Z001.create_dataset('Yield', data=yields.Z014_yields)
        MH = Z001.create_dataset('Ejected_mass', data=yields.Z014_mass_ejected)
        MH = Z001.create_dataset('Total_Metals', data=yields.Z014_total_metals)

        Z01 = Data.create_group('Z_0.007')
        MH = Z01.create_dataset('Yield', data=yields.Z007_yields)
        MH = Z01.create_dataset('Ejected_mass', data=yields.Z007_mass_ejected)
        MH = Z01.create_dataset('Total_Metals', data=yields.Z007_total_metals)


def make_Karakas_table():

    yields = make_yield_tables()

    for i, Zi, in enumerate(yields.Z_bins):
        calculate_mass_ejected(yields, i, Zi)
        calculate_net_yields(yields, i, Zi)

    yields.calculate_total_metals()

    yields_doherty = read_doherty_data()

    make_new_data_range(yields_doherty)

    for i, Zi, in enumerate(yields.Z_bins):
        extrapolate_yield_data(yields, yields_doherty, Zi)
        extrapolate_mej_data(yields, yields_doherty, Zi)
        extrapolate_s_process_data(yields, Zi)

    yields.calculate_total_metals()
    yields.mass_bins = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.,
                                 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5.,
                                 5.5, 6., 7., 8., 9., 10., 11., 12.])
    yields.num_mass_bins = len(yields.mass_bins)

    # Write data to HDF5
    output_Karakas_table(yields)





