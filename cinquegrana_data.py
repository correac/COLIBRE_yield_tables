#%%

#Cinquegrana & Karakas 2021

#%%

import numpy as np
import h5py
import os.path
import re
from utils import interpolate_data


class make_yields_info:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        self.num_Z_bins = len(self.Z_bins)

        self.species_names = np.array(['h', 'he', 'c', 'n', 'o', 'ne', 'mg', 'si', 's', 'ca', 'fe', 'sr', 'ba']) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56]) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.5, 6., 7., 8., 9., 10., 11., 12.])
        self.num_mass_bins = len(self.mass_bins)

        self.file_ending = np.array(['z04','z05','z06','z07','z08','z09','z10'])

def calculate_yields(yields_info, index, mass_range):

    data_yield = np.zeros((yields_info.num_species, len(mass_range)))
    data_mass_ejected = np.zeros(len(mass_range))
    initial_mass_data = mass_range.copy()

    for i, m in enumerate(mass_range):

        file_name = "./data/Cinquegrana2021/" + yields_info.file_ending[index] + "/yields_m%.1f" % m + yields_info.file_ending[index] + ".dat"
        if not os.path.isfile(file_name):
            file_name = "./data/Cinquegrana2021/" + yields_info.file_ending[index] + "/yields_m%i" % m + yields_info.file_ending[index] + ".dat"

        data = np.loadtxt(file_name, skiprows=2, dtype=np.str)

        data_len = len(data[:,0])
        yield_data = np.zeros(data_len)
        for j in range(data_len):
            data[j,0] = re.split('(\d+)',data[j,0])[0]
            yield_data[j] = float( data[j,2] )
            data_mass_ejected[i] += float( data[j,3] )

        for j, elem in enumerate(yields_info.species_names):

            which_species = np.where(data[:,0] == elem)[0]
            if len(which_species) == 0:continue
            data_yield[j, i] = np.sum( yield_data[which_species] )

        data_yield[0, i] = yield_data[2].copy() # completing for hydrogen..
        which_species = np.where(data[:, 0] == 'g')[0]
        data_yield[11, i] = 1e-3 * yield_data[which_species].copy()  # completing for strontium..
        data_yield[12, i] = 2e-4 * yield_data[which_species].copy()  # completing for barium..

    final_mass_data = initial_mass_data - data_mass_ejected

    # Interpolating data...
    final_mass = interpolate_data(initial_mass_data, final_mass_data, yields_info.mass_bins)
    mass_ejected = yields_info.mass_bins - final_mass
    net_yields = np.zeros((yields_info.num_species, yields_info.num_mass_bins))

    for j in range(yields_info.num_species):

        # Interpolation function..
        net_yields[j, :] = interpolate_data(initial_mass_data, data_yield[j, :], yields_info.mass_bins)

    return net_yields, mass_ejected



def make_Cinquegrana_table():

    yields_info = make_yields_info()

    # Write data to HDF5
    with h5py.File('./data/Cinquegrana2021/AGB_Cinquegrana2021.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Net yields for AGB stars (in units of solar mass) taken from Cinquegrana & Karakas (2021)"
        Header.attrs["Description"] = np.string_(description)

        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"] = np.string_(contact)

        mass_data = data_file.create_dataset('Masses', data=yields_info.mass_bins)
        mass_data.attrs["Description"] = np.string_("Mass bins in units of Msolar")

        Z_data = data_file.create_dataset('Metallicities', data=yields_info.Z_bins)
        Z_data.attrs["Description"] = np.string_("Metallicity bins")

        data_file.create_dataset('Number_of_metallicities', data=yields_info.num_Z_bins)
        data_file.create_dataset('Number_of_masses', data=yields_info.num_mass_bins)
        data_file.create_dataset('Number_of_species', data=np.array([13]))

        Z_names = ['Z_0.04', 'Z_0.05', 'Z_0.06', 'Z_0.07', 'Z_0.08', 'Z_0.09', 'Z_0.10']
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Element_names = np.string_(['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur',
                   'Calcium', 'Iron', 'Strontium', 'Barium'])
        dt = h5py.string_dtype(encoding='ascii')
        data_file.create_dataset('Species_names', dtype=dt, data=Element_names)

        Reference = np.string_(['Cinquegrana & Karakas (2021) 2021MNRAS...3045C'])
        data_file.create_dataset('Reference', data=Reference)

        Data = data_file.create_group('Yields')

        for i, Zi, in enumerate(yields_info.Z_bins):

            if Zi == 0.090:
                mass_range = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7.5,8])
            elif Zi == 0.100:
                mass_range = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,6,6.5,7.,7.5,8])
            else:
                mass_range = np.arange(1, 8.5, 0.5)

            net_yields, mass_ejected = calculate_yields(yields_info, i, mass_range)
            total_metals = np.sum(net_yields[2:, :], axis=0)

            data_group = Data.create_group(Z_names[i])
            data_group.create_dataset('Yield', data=net_yields)
            data_group.create_dataset('Ejected_mass', data=mass_ejected)
            data_group.create_dataset('Total_Metals', data=total_metals)

