import h5py
import numpy as np
from scipy.optimize import curve_fit

def func(x,a,b,c):
    f = a + b*x + c *x**2
    return f

def extrapolate_AGB_yields():

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

    Z_old_bins = np.array([0.007, 0.014, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    Z_bins = np.array([0.0001, 0.001, 0.004, 0.007, 0.014, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])

    Y_Z00001 = Y_Z0007.copy()
    Y_Z0001 = Y_Z0007.copy()
    Y_Z0004 = Y_Z0007.copy()

    # extrapolate N:
    i = 3
    for j in range(0,25):
        N = np.array([Y_Z0007[i, j],Y_Z0014[i, j],Y_Z003[i, j],Y_Z004[i, j],Y_Z005[i, j],Y_Z006[i, j],Y_Z007[i, j],Y_Z008[i, j],Y_Z009[i, j],Y_Z010[i, j]])
        if j<=12 or j>=20:
            popt, pcov = curve_fit(func, Z_old_bins, np.log10(N))
            newN = 10**func(Z_bins,*popt)
            Y_Z00001[i,j] = newN[0]
            Y_Z0001[i,j] = newN[1]
            Y_Z0004[i,j] = newN[2]
            Y_Z0007[i, j] = newN[3]
            Y_Z0014[i, j] = newN[4]
        else:
            select = np.where(Z_old_bins >=0.04)[0]
            popt, pcov = curve_fit(func, Z_old_bins[select], np.log10(N[select]))
            newN = 10**func(Z_bins,*popt)
            Y_Z00001[i,j] = newN[0]
            Y_Z0001[i,j] = newN[1]
            Y_Z0004[i,j] = newN[2]
            Y_Z0007[i, j] = newN[3]
            Y_Z0014[i, j] = newN[4]


    Mtot_Z00001 = np.sum(Y_Z00001[2:, :], axis=0)
    Mtot_Z0001 = np.sum(Y_Z0001[2:, :], axis=0)
    Mtot_Z0004 = np.sum(Y_Z0004[2:, :], axis=0)

    Mej_Z00001 = Mej_Z0007.copy()
    Mej_Z0001 = Mej_Z0007.copy()
    Mej_Z0004 = Mej_Z0007.copy()

    return {'Y_Z00001':Y_Z00001, 'Y_Z0001':Y_Z0001,'Y_Z0004':Y_Z0004,'Y_Z0007':Y_Z0007,'Y_Z0014':Y_Z0014,
            'Mej_Z00001':Mej_Z00001, 'Mej_Z0001':Mej_Z0001, 'Mej_Z0004':Mej_Z0004,
            'Mtot_Z00001':Mtot_Z00001, 'Mtot_Z0001':Mtot_Z0001, 'Mtot_Z0004':Mtot_Z0004}


class make_yield_tables:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.0001, 0.001, 0.004, 0.007, 0.014, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
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


    new_data = extrapolate_AGB_yields()


    elements = np.array(['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Magnesium','Iron','Strontium','Barium'])
    indx = np.array([0, 1, 2, 3, 4, 6, 10, 11, 12])

    # Write data to HDF5
    with h5py.File('../data/AGB_newNlowZ.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Net yields for AGB stars (in units of solar mass) taken from Cinquegrana & Karakas (2021) "
        description += "for the metallicity range Z=0.04-0.1, and initial mass range 1-8 Msun. For the mass range 9-12Msun, these tables have been extrapolated. "
        description += "Additionally, the net yields for AGB stars in the metallicity bins Z=0.03, 0.014 and 0.007 and initial mass range 1-8 Msun are taken from Karakas & Lugaro (2016). "
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
                                   'Cinquegrana & Karakas (2021) MNRAS, 3045C; Karakas & Lugaro (2016) ApJ, 825, 26K; Doherty, C L. Gil-Pons, P. Lau, H.H.B Lattanzio, J C and Siess, L. 2014. MNRAS, 437,195'])
        MH = data_file.create_dataset('Reference', data=Reference)

        Data = data_file.create_group('Yields')

        Z0001 = Data.create_group('Z_0.0001')
        MH = Z0001.create_dataset('Yield', data=new_data['Y_Z00001'])
        MH = Z0001.create_dataset('Ejected_mass', data=new_data['Mej_Z00001'])
        MH = Z0001.create_dataset('Total_Metals', data=new_data['Mtot_Z00001'])

        Z0001 = Data.create_group('Z_0.001')
        MH = Z0001.create_dataset('Yield', data=new_data['Y_Z0001'])
        MH = Z0001.create_dataset('Ejected_mass', data=new_data['Mej_Z0001'])
        MH = Z0001.create_dataset('Total_Metals', data=new_data['Mtot_Z0001'])

        Z0001 = Data.create_group('Z_0.004')
        MH = Z0001.create_dataset('Yield', data=new_data['Y_Z0004'])
        MH = Z0001.create_dataset('Ejected_mass', data=new_data['Mej_Z0004'])
        MH = Z0001.create_dataset('Total_Metals', data=new_data['Mtot_Z0004'])

        Z0001 = Data.create_group('Z_0.007')
        MH = Z0001.create_dataset('Yield', data=new_data['Y_Z0007'])
        MH = Z0001.create_dataset('Ejected_mass', data=Mej_Z0007)
        MH = Z0001.create_dataset('Total_Metals', data=Mtot_Z0007)

        Z0001 = Data.create_group('Z_0.014')
        MH = Z0001.create_dataset('Yield', data=new_data['Y_Z0014'])
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


if __name__ == "__main__":

    modify_AGB_yield_tables()