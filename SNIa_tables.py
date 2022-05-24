#%%

# Creating Kobayashi + 2020 yield tables for SNIa.

#%%

import numpy as np
import h5py
from datetime import datetime

def make_SNIa_tables():

    #%%
    # Physical constants
    Msun = 1.9884e33 #gr
    mp_in_cgs = 1.6726e-24 #gr
    mH_in_cgs = 1.00784*mp_in_cgs
    mC_in_cgs = 12.0107*mp_in_cgs
    mN_in_cgs = 14.0067*mp_in_cgs
    mO_in_cgs = 15.999*mp_in_cgs
    mNe_in_cgs = 20.1797*mp_in_cgs
    mMg_in_cgs = 24.3050*mp_in_cgs
    mSi_in_cgs = 28.0855*mp_in_cgs
    mFe_in_cgs = 55.845*mp_in_cgs
    m56Fe_in_cgs = 56*mp_in_cgs
    mXi = np.array([0, 0, mC_in_cgs, mN_in_cgs, mO_in_cgs,
                    mNe_in_cgs, mMg_in_cgs, mSi_in_cgs, mFe_in_cgs])

    species_names = ['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen',
                     'Neon', 'Magnesium', 'Silicon', 'Iron']

    species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 26])

    num_species = len(species)

    #[Xi/^56 Fe] == [Xi/Fe]-[Xi/Fe]solar
    SNIa_element_abundance = np.array([0, 0, -3.3, 0, -1.9, -3.7, -2.05, -0.25, 0.3])
    Xi_Fe_Lodder = SNIa_element_abundance.copy()
    SNIa_element_mass = np.zeros(len(SNIa_element_abundance))

    # Solar abundances taken from Lodder+2010, [Xi/H]solar
    Xi_H_Lodder = np.array([12, 10.925, 8.39, 7.86, 8.73, 8.05, 7.54, 7.53, 7.46])

    Fe_H_Sun_Lodder = 7.46

    mass_ratios = np.array([0, 0,
                            mC_in_cgs/m56Fe_in_cgs,
                            mN_in_cgs/m56Fe_in_cgs,
                            mO_in_cgs/m56Fe_in_cgs,
                            mNe_in_cgs/m56Fe_in_cgs,
                            mMg_in_cgs/m56Fe_in_cgs,
                            mSi_in_cgs/m56Fe_in_cgs,
                            mFe_in_cgs/m56Fe_in_cgs])

    Xi_Fe_Lodder[2:] = Xi_H_Lodder[2:] - Fe_H_Sun_Lodder

    Mass_Fe_56 = 0.72 # Fig. 10 total mass in Msun

    # Remove normalization to solar abundances,
    # log10 X(Xi)/X(Fe), with N number of atoms of element Xi and Fe.
    SNIa_element_abundance[2:] += Xi_Fe_Lodder[2:]
    SNIa_element_mass[2:] = 10**SNIa_element_abundance[2:] * Mass_Fe_56 * mass_ratios[2:]

    # Set to zero Hydrogen, Helium and Nitrogen
    SNIa_element_mass[0] = 0
    SNIa_element_mass[1] = 0
    SNIa_element_mass[3] = 0
    SNIa_element_mass[-1] = 0.83 # Fig. 10 total mass in Msun

    # Here yields are defined as the total mass (of each element)
    # that is expelled to the ISM during the star lifetime

    # Here yields are defined as the total mass (of each element)
    # that is expelled to the ISM during the star lifetime

    metal_mass_lost = 1.37 #Msun

    # Test Niquel
    Ni_Fe = 0.2 #[Ni/Fe] - [Ni/Fe]solar
    mNi_in_cgs = 58.6934 * mp_in_cgs
    Ni_Fe_solar = 6.22 - Fe_H_Sun_Lodder
    Ni_Fe = Ni_Fe + Ni_Fe_solar #log10 X(Ni)/X(Fe)
    mass_Ni = 10**Ni_Fe * Mass_Fe_56 * (mNi_in_cgs / m56Fe_in_cgs)

    # Test Cr
    Cr_Fe = 0.1 #[Ni/Fe] - [Ni/Fe]solar
    mCr_in_cgs = 51.9961 * mp_in_cgs
    Cr_Fe_solar = 5.65 - Fe_H_Sun_Lodder
    Cr_Fe = Cr_Fe + Cr_Fe_solar #log10 X(Ni)/X(Fe)
    mass_Cr = 10**Cr_Fe * Mass_Fe_56 * (mCr_in_cgs / m56Fe_in_cgs)

    # Write data to HDF5
    with h5py.File('./data/SNIa.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Yields for SNIa stars (in units of solar mass) taken from Kobayashi et al. (2020)"
        Header.attrs["Description"]=np.string_(description)

        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"]=np.string_(contact)

        date_int = int(datetime.today().strftime('%Y%m%d'))
        date_string = data_file.create_dataset('Date_string', data=np.array([date_int]))

        Reference = np.string_(['Kobayashi, C., et al., (2020) ApJ Vol 895, 2, 138'])
        MH = data_file.create_dataset('Reference', data=Reference)

        MH = data_file.create_dataset('Number_of_species', data=np.array([num_species]))

        Element_names = np.string_(species_names)
        dt = h5py.string_dtype(encoding='ascii')
        MH = data_file.create_dataset('Species_names',dtype=dt,data=Element_names)

        MH = data_file.create_dataset('Total_Metals', data=metal_mass_lost)

        MH = data_file.create_dataset('Yield', data=SNIa_element_mass)

