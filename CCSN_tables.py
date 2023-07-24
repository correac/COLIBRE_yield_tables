#%%

# Making tables from Nomoto et al. (2013) dataset

#%%

import numpy as np
import h5py
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

'''
Nomoto2013 sn2 yields from 13Msun onwards
'''

from scipy.optimize import curve_fit
from scipy import interpolate
from plotter import plot_CCSN_yield_tables

def func(x,a,b):
    f = a + b*x**0.5 #Metallicity-dependent mass loss, see Kobayashi+ 2006
    return f

def linear(x,a,b):
    f = a + b * x
    return f

def extrapolate_mfinal(metallicity):

    masses = np.array((13, 15, 18, 20, 25, 30, 40))
    num_mass_bins = len(masses)

    mfinal = np.zeros((3,num_mass_bins))
    mfinal[0,:] = np.array([12.93, 14.92, 17.84, 19.72, 24.42, 29.05, 37.81])
    mfinal[1,:] = np.array([12.86, 14.39, 16.59, 19.52, 24.03, 27.56, 32.93])
    mfinal[2,:] = np.array([12.73, 14.14, 16.76, 18.36, 21.63, 24.58, 21.83])
    z_bins = np.array([0.001, 0.004, 0.02])

    mnew = np.zeros(num_mass_bins)
    for i in range(num_mass_bins):
        popt, pcov = curve_fit(func, z_bins, mfinal[:,i])
        mnew[i] = func(metallicity, *popt)

    return mnew

def read_data(table):

    masses = np.array((13, 15, 18, 20, 25, 30, 40))
    species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26])  # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe
    mass_indices = range(21,28)
    elements = np.array(['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe'])
    num_species = len(species)
    num_mass_bins = len(masses)

    # compute SN ejecta mass my summing tabulated element masses
    output_sn_mej = np.zeros(masses.size)
    for i in np.arange(2, table.size):
        for j in range(masses.size):
            output_sn_mej[j] += table[i][mass_indices[j]]

    # tabulated remnant masses
    mcut = np.array([table[1][21], table[1][22], table[1][23], table[1][24], table[1][25], table[1][26], table[1][27]])

    # total mass loss is difference between ZAMS mass and remnant mass 
    output_mass_ejected = masses - mcut

    # remaining ejected mass when subtracting away SN ejecta is from stellar winds
    output_wind_mej = output_mass_ejected - output_sn_mej

    hydrogen_list = ['H__1', 'H__2']
    helium_list = ['He_3', 'He_4']
    lithium_list = ['Li_6', 'Li_7']
    berillium_list = ['Be_9']
    boron_list = ['B_10', 'B_11']
    carbon_list = ['C_12', 'C_13']
    nitrogen_list = ['N_14', 'N_15']
    oxygen_list = ['O_16', 'O_17', 'O_18']
    fluorin_list = ['F_19']
    neon_list = ['Ne20', 'Ne21', 'Ne22']
    sodium_list = ['Na23']
    magnesium_list = ['Mg24', 'Mg25', 'Mg26']
    aluminium_list = ['Al27']
    silicon_list = ['Si28', 'Si29', 'Si30']
    phosphorus_list = ['P_31']
    sulfur_list = ['S_32', 'S_33', 'S_34', 'S_36']
    chlorine_list = ['Cl35', 'Cl37']
    argon_list = ['Ar36', 'Ar38', 'Ar40']
    potassium_list = ['K_39', 'K_41']
    calcium_list = ['K_40', 'Ca40', 'Ca42', 'Ca43', 'Ca44', 'Ca46', 'Ca48']
    scandium_list = ['Sc45']
    titanium_list = ['Ti46', 'Ti47', 'Ti48', 'Ti49', 'Ti50']
    vanadium_list = ['V_50', 'V_51']
    chromium_list = ['Cr50', 'Cr52', 'Cr53', 'Cr54']
    manganese_list = ['Mn55']
    iron_list = ['Fe54', 'Fe56', 'Fe57', 'Fe58']
    cobalt_list = ['Co59']
    nickel_list = ['Ni58', 'Ni60', 'Ni61', 'Ni62', 'Ni64']
    copper_list = ['Cu63', 'Cu65']
    zinc_list = ['Zn64', 'Zn66', 'Zn67', 'Zn68', 'Zn70']
    gallium_list = ['Ga69', 'Ga71']
    germanium_list = ['Ge70', 'Ge72', 'Ge73', 'Ge74']

    indexing = {}
    indexing['H'] = hydrogen_list
    indexing['He'] = helium_list
    indexing['Li'] = lithium_list
    indexing['Be'] = berillium_list
    indexing['B'] = boron_list
    indexing['C'] = carbon_list
    indexing['N'] = nitrogen_list
    indexing['O'] = oxygen_list
    indexing['F'] = fluorin_list
    indexing['Ne'] = neon_list
    indexing['Na'] = sodium_list
    indexing['Mg'] = magnesium_list
    indexing['Al'] = aluminium_list
    indexing['Si'] = silicon_list
    indexing['P'] = phosphorus_list
    indexing['S'] = sulfur_list
    indexing['Cl'] = chlorine_list
    indexing['Ar'] = argon_list
    indexing['K'] = potassium_list
    indexing['Ca'] = calcium_list
    indexing['Sc'] = scandium_list
    indexing['Ti'] = titanium_list
    indexing['V'] = vanadium_list
    indexing['Cr'] = chromium_list
    indexing['Mn'] = manganese_list
    indexing['Fe'] = iron_list
    indexing['Co'] = cobalt_list
    indexing['Ni'] = nickel_list
    indexing['Cu'] = copper_list
    indexing['Zn'] = zinc_list
    indexing['Ga'] = gallium_list
    indexing['Ge'] = germanium_list

    output_ccsn_mej = np.zeros((num_species,num_mass_bins))

    for i in range(2, len(table)):

        table_elem = table[i][0].decode("utf-8")

        for j in range(num_species):
            j_elem = indexing[elements[j]]
            for k in range(len(j_elem)):
                pick = j_elem[k] == table_elem
                if pick:
                    output_ccsn_mej[j, 0] += table[i][21]
                    output_ccsn_mej[j, 1] += table[i][22]
                    output_ccsn_mej[j, 2] += table[i][23]
                    output_ccsn_mej[j, 3] += table[i][24]
                    output_ccsn_mej[j, 4] += table[i][25]
                    output_ccsn_mej[j, 5] += table[i][26]
                    output_ccsn_mej[j, 6] += table[i][27]

    return output_wind_mej, output_ccsn_mej, output_mass_ejected

def make_CCSN_tables():

    species = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26]) # H, He, C, N, O, Ne, Mg, Si, S, Ca, Fe
    elements = np.array(['H','He','C','N','O','Ne','Mg','Si','S','Ca','Fe'])
    num_species = len(species)

    mass_bins = np.array((13, 15, 18, 20, 25, 30, 40))
    num_mass_bins = len(mass_bins)

    metallicities = [0.0000, 0.0010, 0.0040, 0.0080, 0.0200, 0.0500]
    metallicities_list = ['0.000', '0.001', '0.004', '0.008', '0.020', '0.050']
    metal_list_all = ['z=' + s for s in metallicities_list]
    Z_names = ['Z_' + s for s in metallicities_list]
    Z_bins = np.array(metallicities)
    num_Z_bins = len(Z_bins)

    # Write data to HDF5
    with h5py.File('./data/SNII.hdf5', 'w') as data_file:
        Header = data_file.create_group('Header')

        description = "Mass ejected in core-collapse SN (in units of solar mass) produced by Nomoto et al. (2013), compilation of tables from Kobayashi et al. (2006), Nomoto et al. (2006). "
        description += "Note that these tables provide the newly produced material in the cc-sn ejecta, as well as the total mass ejected through winds in the pre-sn phase."
        Header.attrs["Description"] = np.string_(description)

        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"] = np.string_(contact)

        date_int = int(datetime.today().strftime('%Y%m%d'))
        data_file.create_dataset('Date_string', data=np.array([date_int]))

        Reference = np.string_(
            ['Nomoto, K., et al., (2013) Annual Review of Astronomy and Astrophysics, vol. 51, issue 1, pp. 457-509'])
        data_file.create_dataset('Reference', data=Reference)

        masses_data = data_file.create_dataset('Masses', data=mass_bins)
        masses_data.attrs["Description"] = np.string_("Mass bins in units of Msolar")

        Z_data = data_file.create_dataset('Metallicities', data=Z_bins)
        Z_data.attrs["Description"] = np.string_("Metallicity bins")

        data_file.create_dataset('Number_of_metallicities', data=num_Z_bins)
        data_file.create_dataset('Number_of_masses', data=num_mass_bins)
        data_file.create_dataset('Number_of_species', data=np.array([11]))

        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Element_names = np.string_(['Hydrogen', 'Helium', 'Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium', 'Silicon', 'Sulphur',
                   'Calcium', 'Iron'])
        dt = h5py.string_dtype(encoding='ascii')
        data_file.create_dataset('Species_names', dtype=dt, data=Element_names)

        Data = data_file.create_group('Yields')

        for i, index in enumerate(metal_list_all):
            if i == 0:
                dt = np.dtype('a13,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8')
                table = np.loadtxt('./data/Nomoto2013/nomoto_2013_'+index+'0.dat', dtype=dt)

                # For Z=0, Mfinal = Minitial, no mass-loss through winds!
                wind_mej = np.zeros(num_mass_bins)
                mass_ejected = mass_bins - np.array([table[2][23], table[2][24], table[2][25], table[2][26], table[2][27], table[2][28], table[2][29]])

                ccsn_mej = np.zeros((num_species,num_mass_bins))
                for j in range(len(table)):

                    table_elem = table[j][0].decode("utf-8")
                    pick = np.where(elements == table_elem)[0]
                    if len(pick)>0:
                        ccsn_mej[pick,0] += table[j][23]
                        ccsn_mej[pick,1] += table[j][24]
                        ccsn_mej[pick,2] += table[j][25]
                        ccsn_mej[pick,3] += table[j][26]
                        ccsn_mej[pick,4] += table[j][27]
                        ccsn_mej[pick,5] += table[j][28]
                        ccsn_mej[pick,6] += table[j][29]

                data = Data.create_group(Z_names[i])
                data.create_dataset('Ejected_mass_in_ccsn', data=ccsn_mej)
                data.create_dataset('Ejected_mass_in_winds', data=wind_mej)
                data.create_dataset('Total_Mass_ejected', data=mass_ejected)

            else:
                dt = np.dtype('a13,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8')
                table = np.loadtxt('./data/Nomoto2013/nomoto_2013_' + index + '0.dat', dtype=dt)
                wind_mej, ccsn_mej, mass_ejected = read_data(table)

                data = Data.create_group(Z_names[i])
                data.create_dataset('Ejected_mass_in_ccsn', data=ccsn_mej)
                data.create_dataset('Ejected_mass_in_winds', data=wind_mej)
                data.create_dataset('Total_Mass_ejected', data=mass_ejected)


    # This option will rewrite the tables, expand them to include data
    # for stars in the mass range 6-100 Msun.
    apply_linear_extrapolation()
    plot_CCSN_yield_tables()

def apply_linear_extrapolation():

    Z_list = np.array(['00','01','04','08','20','50'])
    M_extrapolated = np.array([6, 7, 8, 9.5, 11, 13, 15, 18, 20, 25, 30, 40, 48, 57.5, 69.5, 83.5, 100])
    mej_ccsn_extrapolated = np.zeros((11, len(M_extrapolated)))
    mej_wind_extrapolated = np.zeros(len(M_extrapolated))
    mass_loss_ccnn_extrapolated = np.zeros(len(M_extrapolated))
    Z_bins = np.array([0.0000, 0.0010, 0.0040, 0.0080, 0.0200, 0.0500])

    # Write data to HDF5
    with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'w') as data_file:

        Header = data_file.create_group('Header')
        description = "Mass ejected in core-collapse SN (in units of solar mass) produced by Nomoto et al. (2013), compilation of tables from Kobayashi et al. (2006), Nomoto et al. (2006). "
        description += "Note that these tables provide the newly produced material in the cc-sn ejecta, as well as the total mass ejected through winds in the pre-sn phase. "
        description += "The yield tables have been modified to include additional data in the low/high mass end. A linear extrapolation method has been applied for this."
        Header.attrs["Description"]=np.string_(description)

        contact = "Dataset generated by Camila Correa (University of Amsterdam). Email: camila.correa@uva.nl,"
        contact += " website: camilacorrea.com"
        Header.attrs["Contact"]=np.string_(contact)

        #date_int = int(datetime.today().strftime('%Y%m%d'))
        date_int = 20220525
        data_file.create_dataset('Date_string', data=np.array([date_int]))
        
        Reference = np.string_(['Nomoto, K., et al., (2013) Annual Review of Astronomy and Astrophysics, vol. 51, issue 1, pp. 457-509'])
        data_file.create_dataset('Reference', data=Reference)

        mass_data = data_file.create_dataset('Masses', data=M_extrapolated)
        mass_data.attrs["Description"]=np.string_("Mass bins in units of Msolar")

        Z_data = data_file.create_dataset('Metallicities', data=Z_bins)
        Z_data.attrs["Description"]=np.string_("Metallicity bins")

        data_file.create_dataset('Number_of_metallicities', data=len(Z_bins))
        data_file.create_dataset('Number_of_masses', data=len(M_extrapolated))
        data_file.create_dataset('Number_of_species', data=np.array([11]))

        Z_names = ['Z_0.000', 'Z_0.001','Z_0.004','Z_0.008','Z_0.020','Z_0.050']
        var = np.array(Z_names, dtype='S')
        dt = h5py.special_dtype(vlen=str)
        data_file.create_dataset('Yield_names', dtype=dt, data=var)

        Z_names = ['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Sulphur','Calcium','Iron']
        Element_names = np.string_(Z_names)
        dt = h5py.string_dtype(encoding='ascii')
        data_file.create_dataset('Species_names',dtype=dt,data=Element_names)

        Data = data_file.create_group('Yields')

        for Zi in Z_list:

            # Read data
            with h5py.File('./data/SNII.hdf5', 'r') as read_data_file:
                Masses = read_data_file["Masses"][:]
                mej_ccsn = read_data_file["/Yields/Z_0.0"+Zi+"/Ejected_mass_in_ccsn"][:][:]
                mej_wind = read_data_file["/Yields/Z_0.0"+Zi+"/Ejected_mass_in_winds"][:]
                mass_loss = read_data_file["/Yields/Z_0.0"+Zi+"/Total_Mass_ejected"][:]

            f_interpolate_mej_wind = interpolate.interp1d(Masses, mej_wind)
            f_interpolate_mass_loss = interpolate.interp1d(Masses, mass_loss)

            lme_mej_wind_popt, _ = curve_fit(linear, Masses[0:4], mej_wind[0:4])
            hme_mej_wind_popt, _ = curve_fit(linear, Masses[-2:], mej_wind[-2:])

            lme_mass_loss_popt, _ = curve_fit(linear, Masses[0:2], mass_loss[0:2])
            hme_mass_loss_popt, _ = curve_fit(linear, Masses[-2:], mass_loss[-2:])

            for i, Mi in enumerate(M_extrapolated):

                if Mi < Masses[0]:
                    mej = linear(Mi, *lme_mej_wind_popt)
                    mej_wind_extrapolated[i] = np.max([0, mej])
                    mej = linear(Mi, *lme_mass_loss_popt)
                    mass_loss_ccnn_extrapolated[i] = np.max([0, mej])

                if (Mi >= Masses[0]) & (Mi <= Masses[-1]):
                    mej = f_interpolate_mej_wind(Mi)
                    mej_wind_extrapolated[i] = np.max([0, mej])
                    mej = f_interpolate_mass_loss(Mi)
                    mass_loss_ccnn_extrapolated[i] = np.max([0, mej])

                if Mi > Masses[-1]:
                    mej = linear(Mi, *hme_mej_wind_popt)
                    mej_wind_extrapolated[i] = np.max([0, mej])
                    mej = linear(Mi, *hme_mass_loss_popt)
                    mass_loss_ccnn_extrapolated[i] = np.max([0, mej])

            for elem in range(11):
                f_interpolate_mej_cc = interpolate.interp1d(Masses, mej_ccsn[elem,:])
                lme_mej_cc_popt, _ = curve_fit(linear, Masses[0:2], mej_ccsn[elem, 0:2])
                hme_mej_cc_popt, _ = curve_fit(linear, Masses[-2:], mej_ccsn[elem, -2:])

                for i, Mi in enumerate(M_extrapolated):

                    if Mi < Masses[0]:
                        mej = linear(Mi, *lme_mej_cc_popt)
                        mej_ccsn_extrapolated[elem, i] = np.max([0, mej])

                    if (Mi >= Masses[0]) & (Mi <= Masses[-1]):
                        mej = f_interpolate_mej_cc(Mi)
                        mej_ccsn_extrapolated[elem, i] = np.max([0, mej])

                    if Mi > Masses[-1]:
                        mej = linear(Mi, *hme_mej_cc_popt)
                        mej_ccsn_extrapolated[elem, i] = np.max([0, mej])

            # Some safety here in the calculate of total mass ejected.
            for i, Mi in enumerate(M_extrapolated):
                total = mej_wind_extrapolated[i] + np.sum(mej_ccsn_extrapolated[:,i], axis=0)

                if total > mass_loss_ccnn_extrapolated[i]:
                    mass_loss_ccnn_extrapolated[i] = total


            data_Zi = Data.create_group('Z_0.0'+Zi )
            data_Zi.create_dataset('Ejected_mass_in_ccsn', data=mej_ccsn_extrapolated)
            data_Zi.create_dataset('Ejected_mass_in_winds', data=mej_wind_extrapolated)
            data_Zi.create_dataset('Total_Mass_ejected', data=mass_loss_ccnn_extrapolated)




if __name__ == "__main__":

    make_CCSN_tables()

