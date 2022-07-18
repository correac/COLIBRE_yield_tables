#%%

#Doherty et al. (2014)

#%%

import numpy as np
import h5py
from scipy import interpolate
import re

class make_yield_tables:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.004,0.008,0.02])
        self.num_Z_bins = len(self.Z_bins)

        self.species_names = np.array(['h', 'he', 'c', 'n', 'o', 'ne', 'mg', 'si', 's', 'ca', 'fe', 'sr', 'ba']) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56]) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array([7., 7.5, 8., 8.5, 9.0])
        self.num_mass_bins = len(self.mass_bins)

        self.Z004_yields = np.zeros((self.num_species,self.num_mass_bins))
        self.Z008_yields = np.zeros((self.num_species,self.num_mass_bins))
        self.Z020_yields = np.zeros((self.num_species,self.num_mass_bins))

        self.Z004_mass_ejected = None
        self.Z008_mass_ejected = None
        self.Z020_mass_ejected = None

        self.Z004_total_metals = None
        self.Z008_total_metals = None
        self.Z020_total_metals = None

        self.file_ending = np.array(['z0004','z0008','z002'])


    def add_mass_data(self, mass_ejected, final_mass, z_bin):

        if z_bin == 0.004:
            self.Z004_mass_ejected = mass_ejected
            self.Z004_final_mass = final_mass
        if z_bin == 0.008:
            self.Z008_mass_ejected = mass_ejected
            self.Z008_final_mass = final_mass
        if z_bin == 0.02:
            self.Z020_mass_ejected = mass_ejected
            self.Z020_final_mass = final_mass


    def calculate_total_metals(self):

        self.Z004_total_metals = np.sum( self.Z004_yields[2:,:], axis = 0)
        self.Z008_total_metals = np.sum( self.Z008_yields[2:,:], axis = 0)
        self.Z020_total_metals = np.sum( self.Z020_yields[2:,:], axis = 0)
        # self.Z004_total_metals = - self.Z004_yields[0,:] - self.Z004_yields[1,:]
        # self.Z008_total_metals = - self.Z008_yields[0,:] - self.Z008_yields[1,:]
        # self.Z020_total_metals = - self.Z020_yields[0,:] - self.Z020_yields[1,:]


def calculate_yields(yields, index, z_bin, mass_range):

    data_yield = np.zeros((yields.num_species, len(mass_range)))
    data_mass_ejected = np.zeros(len(mass_range))
    initial_mass_data = mass_range.copy()

    for i, m in enumerate(mass_range):

        file_name = "./data/Doherty2014/m%.1f" % m +"_"+ yields.file_ending[index] + ".txt"
        data = np.loadtxt(file_name, dtype=np.str)

        data_len = len(data[:,0])
        yield_data = np.zeros(data_len)
        for j in range(data_len):
            data[j,0] = re.split('(\d+)',data[j,0])[0]
            yield_data[j] = float( data[j,1] )
            data_mass_ejected[i] += float( data[j,2] )

        for j, elem in enumerate(yields.species_names):

            which_species = np.where(data[:,0] == elem)[0]
            if len(which_species) == 0:continue
            data_yield[j, i] = np.sum( yield_data[which_species] )

        data_yield[0, i] = yield_data[0].copy() # completing for hydrogen..
        which_species = np.where(data[:, 0] == 'g')[0]
        data_yield[11, i] = yield_data[which_species].copy()  # completing for barium..
        data_yield[12, i] = yield_data[which_species].copy()  # completing for strontium..

    final_mass_data = initial_mass_data - data_mass_ejected

    # Interpolation function..
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

    for j in range(yields.num_species):

        # Interpolation function..
        f = interpolate.interp1d(initial_mass_data, data_yield[j, :])

        for i, m in enumerate(yields.mass_bins):

            if (m >= np.min(initial_mass_data)) & (m <= np.max(initial_mass_data)):
                f_m = f(m)
            elif (m >= np.max(initial_mass_data)):
                f_m = np.max(data_yield[j, :]) * m / np.max(initial_mass_data)
            elif (m <= np.min(initial_mass_data)):
                f_m = np.min(data_yield[j, :]) * m / np.min(initial_mass_data)

            if z_bin == 0.004: yields.Z004_yields[j, i] = f_m
            if z_bin == 0.008: yields.Z008_yields[j, i] = f_m
            if z_bin == 0.020: yields.Z020_yields[j, i] = f_m

    return


def make_Doherty_table():

    yields = make_yield_tables()

    for i, Zi, in enumerate(yields.Z_bins):

        if Zi == 0.008:
            mass_range = np.array([6.5,7.0,7.5,8,8.5])
        elif Zi == 0.004:
            mass_range = np.array([6.5,7.0,7.5,8])
        else:
            mass_range = np.array([7.0,7.5,8,8.5,9.0])

        calculate_yields(yields, i, Zi, mass_range)

    yields.calculate_total_metals()


    # Write data to HDF5
    with h5py.File('./data/Doherty2014/AGB_Doherty2014.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Net yields for AGB stars (in units of solar mass) taken from Doherty et al. (2014)"
        Header.attrs["Description"]=np.string_(description)

        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"]=np.string_(contact)

        MH = data_file.create_dataset('Masses', data=yields.mass_bins)
        MH.attrs["Description"]=np.string_("Mass bins in units of Msolar")

        MH = data_file.create_dataset('Metallicities', data=yields.Z_bins)

        MH.attrs["Description"]=np.string_("Metallicity bins")

        MH = data_file.create_dataset('Number_of_metallicities', data=yields.num_Z_bins)

        MH = data_file.create_dataset('Number_of_masses', data=yields.num_mass_bins)

        MH = data_file.create_dataset('Number_of_species', data=np.array([13]))

        Z_names = ['Z_0.004','Z_0.008','Z_0.02']
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        MH = data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Z_names = ['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Sulphur','Calcium','Iron','Strontium','Barium']
        Element_names = np.string_(Z_names)
        dt = h5py.string_dtype(encoding='ascii')
        MH = data_file.create_dataset('Species_names',dtype=dt,data=Element_names)

        Reference = np.string_(['Doherty, C L. Gil-Pons, P. Lau, H.H.B Lattanzio, J C and Siess, L. 2014. MNRAS, 437,195'])
        MH = data_file.create_dataset('Reference', data=Reference)

        Data = data_file.create_group('Yields')

        Z0001 = Data.create_group('Z_0.004')
        MH = Z0001.create_dataset('Yield', data=yields.Z004_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z004_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z004_total_metals)

        Z0001 = Data.create_group('Z_0.008')
        MH = Z0001.create_dataset('Yield', data=yields.Z008_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z008_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z008_total_metals)

        Z0001 = Data.create_group('Z_0.02')
        MH = Z0001.create_dataset('Yield', data=yields.Z020_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z020_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z020_total_metals)
