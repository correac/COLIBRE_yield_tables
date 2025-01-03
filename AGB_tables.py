import numpy as np
import h5py

from karakas_data import make_Karakas_table
from doherty_data import make_Doherty_table
from cinquegrana_data import make_Cinquegrana_table
from fishlock_data import make_Fishlock_table
from karakas2010_data import make_Karakas2010_table

from plotter import plot_AGB_tables

from datetime import datetime

class make_yield_tables:

    def __init__(self):

        # Table details, let's specify metallicity and mass bins :
        self.Z_bins = np.array([0.0001, 0.001, 0.004, 0.007, 0.014, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        self.num_Z_bins = len(self.Z_bins)

        self.species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26, 38, 56]) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe, Sr & Ba
        self.num_species = len(self.species)

        self.mass_bins = np.array([1., 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.5, 6., 7., 8., 9., 10., 11., 12.])
        self.num_mass_bins = len(self.mass_bins)

def load_data(file, Z_bin):

    with h5py.File(file, 'r') as data_file:
        Yield = data_file["/Yields/"+Z_bin+"/Yield"][:][:]
        Mej = data_file["/Yields/"+Z_bin+"/Ejected_mass"][:]
        Mtot = data_file["/Yields/"+Z_bin+"/Total_Metals"][:]

    return {"Yield":Yield, "Mej":Mej, "Mtot":Mtot}


def combine_tables():

    yields = make_yield_tables()

    # Write data to HDF5
    with h5py.File('./data/AGB.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Net yields for AGB stars (in units of solar mass) taken from Cinquegrana & Karakas (2021) "
        description += "for the metallicity range Z=0.0001-0.1, and initial mass range 1-8 Msun. For the mass range 9-12Msun, these tables have been extrapolated. "
        description += "Additionally, the net yields for AGB stars in the metallicity bins Z=0.03, 0.014 and 0.007 and initial mass range 1-8 Msun are taken from Karakas & Lugaro (2016). "
        description += "The net yields for metallicity bins Z=0.001 are taken from Fishlock et al. (2014) and for the metallicity bins Z=0.0001 and Z=0.004 are taken from Karakas et al. (2010). "
        description += "For the mass range 9-12Msun, we have interpolated the data from the tables of Doherty et al. (2014)."
        Header.attrs["Description"] = np.string_(description)

        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila@camilacorrea.com,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"] = np.string_(contact)

        # date_int = int(datetime.today().strftime('%Y%m%d'))
        date_int = 20230328 # Frozen
        data_file.create_dataset('Date_string', data=np.array([date_int]))
        
        mass_data = data_file.create_dataset('Masses', data=yields.mass_bins)
        mass_data.attrs["Description"] = np.string_("Mass bins in units of Msolar")

        Z_data = data_file.create_dataset('Metallicities', data=yields.Z_bins)
        Z_data.attrs["Description"] = np.string_("Metallicity bins")

        data_file.create_dataset('Number_of_metallicities', data=yields.num_Z_bins)
        data_file.create_dataset('Number_of_masses', data=yields.num_mass_bins)
        data_file.create_dataset('Number_of_species', data=np.array([13]))

        Z_names = ['Z_0.0001','Z_0.001','Z_0.004', 'Z_0.007', 'Z_0.014', 'Z_0.03', 'Z_0.04', 'Z_0.05', 'Z_0.06', 'Z_0.07', 'Z_0.08', 'Z_0.09', 'Z_0.10']
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Element_names = np.string_(['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur',
                   'Calcium', 'Iron', 'Strontium', 'Barium'])
        dt = h5py.string_dtype(encoding='ascii')
        data_file.create_dataset('Species_names', dtype=dt, data=Element_names)

        Reference = np.string_([
            'Cinquegrana & Karakas (2021) MNRAS, 3045C; '
            'Karakas & Lugaro (2016) ApJ, 825, 26K; '
            'Doherty, C L. Gil-Pons, P. Lau, H.H.B Lattanzio, J C and Siess, L. 2014. MNRAS, 437,195; '
            'Fishlock, C.; Karakas, A.; Lugaro, M.; Yong, D., 2014, ApJ, 797, 1, 44, 25; '
            'Karakas, A., et al., 2010, MNRAS, 477, 1, 421'
            ])
        data_file.create_dataset('Reference', data=Reference)

        Data = data_file.create_group('Yields')

        for i, index in enumerate(Z_names):
            if i == 0 or i == 2:
                file = './data/Karakas2010/AGB_Karakas2010.hdf5'
            if i == 1:
                file = './data/Fishlock2014/AGB_Fishlock2014.hdf5'
            if i >= 3 and i <= 5:
                file = './data/Karakas2016/AGB_Karakas2016.hdf5'
            if i >= 6:
                file = './data/Cinquegrana2021/AGB_Cinquegrana2021.hdf5'

            data = load_data(file, index)
            data_group = Data.create_group(index)
            data_group.create_dataset('Yield', data=data['Yield'])
            data_group.create_dataset('Ejected_mass', data=data['Mej'])
            data_group.create_dataset('Total_Metals', data=data['Mtot'])


def make_AGB_tables():

    make_Doherty_table()
    make_Karakas_table()
    make_Karakas2010_table()
    make_Fishlock_table()
    make_Cinquegrana_table()

    combine_tables()
    plot_AGB_tables()
