import h5py
import numpy as np
from scipy.interpolate import interp1d

# def print_COLIBRE():
#
#     total_mass_fraction = 0.0133714
#
#     file = '../data/SNII_linear_extrapolation.hdf5'
#     data_file = h5py.File(file, 'r')
#
#     elements_all = np.array([k.decode() for k in data_file['Species_names'][:]])
#     indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
#
#     metallicity_range = data_file['Metallicities'][:]
#     metallicity_flag = [k.decode() for k in data_file['Yield_names'][:]]
#
#     Masses = data_file["Masses"][:]
#
#     num_elements = 11
#     num_mass_bins = len(Masses)
#     num_metals = len(metallicity_range)
#
#     stellar_yields_ccsn = np.zeros((num_elements, num_mass_bins, num_metals))
#     stellar_yields_winds = np.zeros((num_mass_bins, num_metals))
#     stellar_yields = np.zeros((num_elements, num_mass_bins, num_metals))
#     total_mass_fraction = 0.0133714
#     mass_fraction = np.array([0.73738788833,  # H
#                               0.24924186942,  # He
#                               0.0023647215,  # C
#                               0.0006928991,  # N
#                               0.00573271036,  # O
#                               0.00125649278,  # Ne
#                               0.00070797838,  # Mg
#                               0.00066495154,  # Si
#                               0, 0,
#                               0.00129199252])  # Fe
#
#     for i, flag in enumerate(metallicity_flag):
#         stellar_yields_ccsn[:, :, i] = data_file["/Yields/" + flag + "/Ejected_mass_in_ccsn"][:][:]
#         stellar_yields_winds[:, i] = data_file["/Yields/" + flag + "/Ejected_mass_in_winds"][:]
#         factor = metallicity_range[i] / total_mass_fraction
#         for j in range(num_mass_bins):
#             stellar_yields[:, j, i] = stellar_yields_ccsn[:, j, i] + factor * mass_fraction *  stellar_yields_winds[j, i]
#
#     print('COLIBRE')
#     for i in range(num_metals):
#         print(stellar_yields[0,:,i], metallicity_range[i])
#     print('=====')

def read_SNII_COLIBRE(Z):

    total_mass_fraction = 0.0133714

    mass_fraction = np.array([0.73738788833,  # H
                              0.24924186942,  # He
                              0.0023647215,  # C
                              0.0006928991,  # N
                              0.00573271036,  # O
                              0.00125649278,  # Ne
                              0.00070797838,  # Mg
                              0.00066495154,  # Si
                              0, 0,
                              0.00129199252])  # Fe

    file = '../data/SNII_linear_extrapolation.hdf5'
    data_file = h5py.File(file, 'r')

    elements_all = np.array([k.decode() for k in data_file['Species_names'][:]])
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    metallicity_range = data_file['Metallicities'][:]
    metallicity_flag = [k.decode() for k in data_file['Yield_names'][:]]

    Masses = data_file["Masses"][:]

    num_elements = len(mass_fraction)
    num_mass_bins = len(Masses)
    stellar_yields_total = np.zeros((num_elements, num_mass_bins))
    stellar_yields_total_1 = np.zeros((num_elements, num_mass_bins))
    stellar_yields_total_2 = np.zeros((num_elements, num_mass_bins))

    indx_nearest = np.abs(metallicity_range - Z).argmin()

    if indx_nearest == len(metallicity_range) - 1:

        Ejected_mass_winds = data_file["/Yields/" + metallicity_flag[indx_nearest] + "/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/" + metallicity_flag[indx_nearest] + "/Ejected_mass_in_ccsn"][:][:]
        Ejected_total = data_file["/Yields/" + metallicity_flag[indx_nearest] + "/Total_Mass_ejected"][:]

        factor = metallicity_range[indx_nearest] / total_mass_fraction

        for i in range(num_mass_bins):
            stellar_yields_total[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]
            stellar_yields_total[:, i] -= factor * mass_fraction * Ejected_total[i]
        stellar_yields_total *= (1. + (Z - metallicity_range[indx_nearest]) / metallicity_range[indx_nearest])

    else:
        if metallicity_range[indx_nearest] > Z:
            indx_1 = indx_nearest - 1
            indx_2 = indx_nearest
        else:
            indx_1 = indx_nearest
            indx_2 = indx_nearest + 1

        Ejected_mass_winds = data_file["/Yields/" + metallicity_flag[indx_1] + "/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/" + metallicity_flag[indx_1] + "/Ejected_mass_in_ccsn"][:][:]
        Ejected_total = data_file["/Yields/" + metallicity_flag[indx_1] + "/Total_Mass_ejected"][:]

        factor = metallicity_range[indx_1] / total_mass_fraction

        for i in range(num_mass_bins):
            stellar_yields_total_1[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]
            stellar_yields_total_1[:, i] -= factor * mass_fraction * Ejected_total[i]

        Ejected_mass_winds = data_file["/Yields/" + metallicity_flag[indx_2] + "/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/" + metallicity_flag[indx_2] + "/Ejected_mass_in_ccsn"][:][:]
        Ejected_total = data_file["/Yields/" + metallicity_flag[indx_2] + "/Total_Mass_ejected"][:]

        factor = metallicity_range[indx_2] / total_mass_fraction

        for i in range(num_mass_bins):
            stellar_yields_total_2[:, i] = Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]
            stellar_yields_total_2[:, i] -= factor * mass_fraction * Ejected_total[i]

        b = (stellar_yields_total_2 - stellar_yields_total_1) / (metallicity_range[indx_2] - metallicity_range[indx_1])
        a = stellar_yields_total_1 - b * metallicity_range[indx_1]

        stellar_yields_total = a + b * Z

    return stellar_yields_total, Masses

def make_EAGLE_tables():

    file = '../data/EAGLE_yieldtables/SNII.hdf5'
    data_file = h5py.File(file, 'r')

    elements_all = np.array([k.decode() for k in data_file['Species_names'][:]])
    metallicity_range = data_file['Metallicities'][:]
    metallicity_flag = [k.decode() for k in data_file['Yield_names'][:]]

    Masses = data_file["Masses"][:]

    num_mass_bins = len(Masses)
    num_elements = 11
    num_metals =  len(metallicity_range)

    Yield = np.zeros((num_elements, num_mass_bins, num_metals))
    Ejected_mass = np.zeros((num_mass_bins, num_metals))
    Total_metals = np.zeros((num_mass_bins, num_metals))

    for i, flag in enumerate(metallicity_flag):
        Yield[:, :, i] = data_file["/Yields/" + flag + "/Yield"][:][:]
        Ejected_mass[:, i] = data_file["/Yields/" + flag + "/Ejected_mass"][:]
        Total_metals[:, i] = data_file["/Yields/" + flag + "/Total_Metals"][:]

    data_file.close()

    total_mass_fraction = 0.0133714
    mass_fraction = 0.73738788833  # H

    for i in range(num_metals):

        factor = metallicity_range[i] / total_mass_fraction
        mej = Yield[0, :, i] + factor * mass_fraction * Ejected_mass[:, i]
        print(i, metallicity_range[i])
        # print('EAGLE ====')
        # print(mej)
        mej_COLIBRE, masses_COLIBRE = read_SNII_COLIBRE(metallicity_range[i])
        # print('COLIBRE ====')
        # print(mej_COLIBRE[0, :])
        print('====')
        print('Modified')

        f = interp1d(masses_COLIBRE, mej_COLIBRE[0, :])
        new_yields = np.zeros(len(Masses))
        for j in range(len(Masses)):
            if Masses[j] > np.max(masses_COLIBRE):
                new_yields[j] = mej_COLIBRE[0, -1]
            else:
                new_yields[j] = f(Masses[j])

        #Yield[0, :, i] = new_yields - factor * mass_fraction * Ejected_mass[:, i]
        # new_yields -= factor * mass_fraction * Ejected_mass[:, i]
        print(Yield[0, :, i], new_yields)
        print('====')
        if i <= 2:
            Yield[0, :, i] = new_yields.copy()

    # Write data to HDF5
    with h5py.File('./modified_EAGLE_SNII.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')
        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"] = np.string_(contact)

        mass_data = data_file.create_dataset('Masses', data=Masses)
        mass_data.attrs["Description"] = np.string_("Mass bins in units of Msolar")

        Z_data = data_file.create_dataset('Metallicities', data=metallicity_range)
        Z_data.attrs["Description"] = np.string_("Metallicity bins")

        data_file.create_dataset('Number_of_metallicities', data=len(metallicity_range))
        data_file.create_dataset('Number_of_masses', data=len(Masses))
        data_file.create_dataset('Number_of_species', data=np.array([11]))

        Z_names = metallicity_flag.copy()
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Element_names = elements_all.copy()
        var = np.array(Element_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        data_file.create_dataset('Species_names', dtype=dt, data=var)

        Data = data_file.create_group('Yields')

        for i, flag in enumerate(Z_names):
            data_group = Data.create_group(flag)
            data_group.create_dataset('Yield', data=Yield[:, :, i])
            data_group.create_dataset('Ejected_mass', data=Ejected_mass[:, i])
            data_group.create_dataset('Total_Metals', data=Total_metals[:, i])
    
if __name__ == "__main__":

    # Make and combine AGB yield tables
    make_EAGLE_tables()