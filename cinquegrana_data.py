#%%

#Cinquegrana & Karakas 2021

#%%

import numpy as np
import h5py
from scipy import interpolate
import os.path
import re

class make_yield_tables:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        self.num_Z_bins = len(self.Z_bins)

        self.species_names = np.array(['h', 'he', 'c', 'n', 'o', 'ne', 'mg', 'si', 's', 'ca', 'fe', 'sr', 'ba']) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56]) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.5, 6., 7., 8., 9., 10., 11., 12.])
        self.num_mass_bins = len(self.mass_bins)

        self.Z040_yields = np.zeros((self.num_species,self.num_mass_bins))
        self.Z050_yields = np.zeros((self.num_species,self.num_mass_bins))
        self.Z060_yields = np.zeros((self.num_species,self.num_mass_bins))
        self.Z070_yields = np.zeros((self.num_species,self.num_mass_bins))
        self.Z080_yields = np.zeros((self.num_species,self.num_mass_bins))
        self.Z090_yields = np.zeros((self.num_species,self.num_mass_bins))
        self.Z100_yields = np.zeros((self.num_species,self.num_mass_bins))

        self.Z040_mass_ejected = None
        self.Z050_mass_ejected = None
        self.Z060_mass_ejected = None
        self.Z070_mass_ejected = None
        self.Z080_mass_ejected = None
        self.Z090_mass_ejected = None
        self.Z100_mass_ejected = None

        self.Z040_total_metals = None
        self.Z050_total_metals = None
        self.Z060_total_metals = None
        self.Z070_total_metals = None
        self.Z080_total_metals = None
        self.Z090_total_metals = None
        self.Z100_total_metals = None

        self.file_ending = np.array(['z04','z05','z06','z07','z08','z09','z10'])


    def add_mass_data(self, mass_ejected, final_mass, z_bin):

        if z_bin == 0.04:
            self.Z040_mass_ejected = mass_ejected
            self.Z040_final_mass = final_mass
        if z_bin == 0.05:
            self.Z050_mass_ejected = mass_ejected
            self.Z050_final_mass = final_mass
        if z_bin == 0.06:
            self.Z060_mass_ejected = mass_ejected
            self.Z060_final_mass = final_mass
        if z_bin == 0.07:
            self.Z070_mass_ejected = mass_ejected
            self.Z070_final_mass = final_mass
        if z_bin == 0.08:
            self.Z080_mass_ejected = mass_ejected
            self.Z080_final_mass = final_mass
        if z_bin == 0.09:
            self.Z090_mass_ejected = mass_ejected
            self.Z090_final_mass = final_mass
        if z_bin == 0.10:
            self.Z100_mass_ejected = mass_ejected
            self.Z100_final_mass = final_mass

    def calculate_total_metals(self):

        self.Z040_total_metals = - self.Z040_yields[0,:] - self.Z040_yields[1,:]
        self.Z050_total_metals = - self.Z050_yields[0,:] - self.Z050_yields[1,:]
        self.Z060_total_metals = - self.Z060_yields[0,:] - self.Z060_yields[1,:]
        self.Z070_total_metals = - self.Z070_yields[0,:] - self.Z070_yields[1,:]
        self.Z080_total_metals = - self.Z080_yields[0,:] - self.Z080_yields[1,:]
        self.Z090_total_metals = - self.Z090_yields[0,:] - self.Z090_yields[1,:]
        self.Z100_total_metals = - self.Z100_yields[0,:] - self.Z100_yields[1,:]




def calculate_yields(yields, index, z_bin, mass_range):

    data_yield = np.zeros((yields.num_species, len(mass_range)))
    data_mass_ejected = np.zeros(len(mass_range))
    initial_mass_data = mass_range.copy()

    for i, m in enumerate(mass_range):

        file_name = "./data/Cinquegrana2021/" + yields.file_ending[index] + "/yields_m%.1f" % m + yields.file_ending[index] + ".dat"
        if not os.path.isfile(file_name):
            file_name = "./data/Cinquegrana2021/" + yields.file_ending[index] + "/yields_m%i" % m + yields.file_ending[index] + ".dat"

        data = np.loadtxt(file_name, skiprows=2, dtype=np.str)

        data_len = len(data[:,0])
        yield_data = np.zeros(data_len)
        for j in range(data_len):
            data[j,0] = re.split('(\d+)',data[j,0])[0]
            yield_data[j] = float( data[j,2] )
            data_mass_ejected[i] += float( data[j,3] )

        for j, elem in enumerate(yields.species_names):

            which_species = np.where(data[:,0] == elem)[0]
            if len(which_species) == 0:continue
            data_yield[j, i] = np.sum( yield_data[which_species] )

        data_yield[0, i] = yield_data[2].copy() # completing for hydrogen..
        which_species = np.where(data[:, 0] == 'g')[0]
        data_yield[11, i] = 1e-3 * yield_data[which_species].copy()  # completing for strontium..
        data_yield[12, i] = 2e-4 * yield_data[which_species].copy()  # completing for barium..

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
                f_m = data_yield[j, -1] * m / np.max(initial_mass_data)
            elif (m <= np.min(initial_mass_data)):
                f_m = data_yield[j, 0] * m / np.min(initial_mass_data)

            if z_bin == 0.040: yields.Z040_yields[j, i] = f_m
            if z_bin == 0.050: yields.Z050_yields[j, i] = f_m
            if z_bin == 0.060: yields.Z060_yields[j, i] = f_m
            if z_bin == 0.070: yields.Z070_yields[j, i] = f_m
            if z_bin == 0.080: yields.Z080_yields[j, i] = f_m
            if z_bin == 0.090: yields.Z090_yields[j, i] = f_m
            if z_bin == 0.100: yields.Z100_yields[j, i] = f_m


    return

def make_Cinquegrana_table():

    yields = make_yield_tables()

    for i, Zi, in enumerate(yields.Z_bins):

        if Zi == 0.090:
            mass_range = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7.5,8])
        elif Zi == 0.100:
            mass_range = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,6,6.5,7.,7.5,8])
        else:
            mass_range = np.arange(1, 8.5, 0.5)

        calculate_yields(yields, i, Zi, mass_range)

    yields.calculate_total_metals()


    # Write data to HDF5
    with h5py.File('./data/Cinquegrana2021/AGB_Cinquegrana2021.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Net yields for AGB stars (in units of solar mass) taken from Cinquegrana & Karakas (2021)"
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

        Z_names = ['Z_0.04','Z_0.05','Z_0.06','Z_0.07','Z_0.08','Z_0.09','Z_0.10']
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        MH = data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Z_names = ['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Sulphur','Calcium','Iron','Strontium','Barium']
        Element_names = np.string_(Z_names)
        dt = h5py.string_dtype(encoding='ascii')
        MH = data_file.create_dataset('Species_names',dtype=dt,data=Element_names)

        Reference = np.string_(['Cinquegrana & Karakas (2021) 2021MNRAS...3045C'])
        MH = data_file.create_dataset('Reference', data=Reference)

        Data = data_file.create_group('Yields')

        Z0001 = Data.create_group('Z_0.04')
        MH = Z0001.create_dataset('Yield', data=yields.Z040_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z040_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z040_total_metals)

        Z0001 = Data.create_group('Z_0.05')
        MH = Z0001.create_dataset('Yield', data=yields.Z050_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z050_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z050_total_metals)

        Z0001 = Data.create_group('Z_0.06')
        MH = Z0001.create_dataset('Yield', data=yields.Z060_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z060_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z060_total_metals)

        Z0001 = Data.create_group('Z_0.07')
        MH = Z0001.create_dataset('Yield', data=yields.Z070_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z070_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z070_total_metals)

        Z0001 = Data.create_group('Z_0.08')
        MH = Z0001.create_dataset('Yield', data=yields.Z080_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z080_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z080_total_metals)

        Z0001 = Data.create_group('Z_0.09')
        MH = Z0001.create_dataset('Yield', data=yields.Z090_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z090_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z090_total_metals)

        Z0001 = Data.create_group('Z_0.10')
        MH = Z0001.create_dataset('Yield', data=yields.Z100_yields)
        MH = Z0001.create_dataset('Ejected_mass', data=yields.Z100_mass_ejected)
        MH = Z0001.create_dataset('Total_Metals', data=yields.Z100_total_metals)
