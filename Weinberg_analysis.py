import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.integrate import quad, romberg
from scipy.integrate import cumtrapz, romb, simpson
from scipy.interpolate import interp1d
import h5py
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306) # Abbott et al. (2022)

mass_fraction = np.array([0.73738788833, #H
                          0.24924186942, #He
                          0.0023647215,  #C
                          0.0006928991,  #N
                          0.00573271036, #O
                          0.00125649278, #Ne
                          0.00070797838, #Mg
                          0.00066495154, #Si
                          0,
                          0,
                          0.00129199252]) #Fe

def lifetimes(m, Z, filename):
    """
    Interpolates stellar lifetimes based on mass and metallicity.

    Parameters:
    m (float): Stellar mass.
    Z (float): Metallicity.
    filename (str): Path to the HDF5 file containing the lifetime data.

    Returns:
    float: Interpolated lifetime in Gyr.
    """
    # Read data from HDF5 file
    with h5py.File(filename, 'r') as data_file:
        Masses = data_file["Masses"][:]
        lifetimes = data_file["Lifetimes"][:]
        Metallicities = data_file["Metallicities"][:]

    # Interpolate lifetimes for given mass and metallicity
    num_metals = len(Metallicities)
    MZ = np.zeros(num_metals)
    for i in range(num_metals):
        f_mass = interp1d(Masses, lifetimes[i, :], bounds_error=False, fill_value=(lifetimes[i, 0], lifetimes[i, -1]))
        MZ[i] = f_mass(m)

    f_metal = interp1d(Metallicities, MZ, bounds_error=False, fill_value=(MZ[0], MZ[-1]))
    result = f_metal(Z) / 1e9  # Convert to Gyr
    return result

def inverse_lifetime(Z, filename='./data/EAGLE_yieldtables/Lifetimes.hdf5'):
    """
    Finds the minimum mass with a lifetime shorter than the Hubble time for a given metallicity.

    Parameters:
    Z (float): Metallicity.
    filename (str): Path to the HDF5 file containing the lifetime data.

    Returns:
    float: Minimum mass with a lifetime shorter than the Hubble time.
    """
    Hubble_time = cosmo.age(0).value  # Gyr
    mass_range = np.arange(0.01, 1, 0.01)

    # Find the mass limit where the lifetime is less than the Hubble time
    for m in mass_range:
        t = lifetimes(m, Z, filename)
        if t <= Hubble_time:
            return m

    # If no mass is found within the range, return None or raise an error
    return None

def imf_lin(m):
    """
    Calculate the Initial Mass Function (IMF) in linear form.

    Parameters:
    m (float): Mass for which to calculate the IMF.

    Returns:
    float: Value of the IMF at mass m.
    """

    if m <= 1:
        log_m = np.log10(m)
        dm = log_m - np.log10(0.079)
        sigma = 0.69
        A = 0.852464
        xi = A * np.exp(-0.5 * dm**2 / sigma**2) / m

    else:
        A = 0.237912
        x = -2.3
        xi = A * m ** x

    xi /= np.log(10)
    return xi

def integrate_IMF_number(m_min, m_max, num_points=1000):
    """
    Integrate the Initial Mass Function (IMF) over a range of masses.

    Parameters:
    m_min (float): Minimum mass to integrate over.
    m_max (float): Maximum mass to integrate over.
    num_points (int): Number of points to use in the integration. Default is 1000.

    Returns:
    float: Integrated value of the IMF over the specified mass range.
    """
    # Generate logarithmically spaced mass values for better precision
    Masses = np.logspace(np.log10(m_min), np.log10(m_max), num_points)

    # Compute the IMF values for all masses
    imf_values = np.vectorize(imf_lin)(Masses)

    # Integrate using the composite Simpson's rule
    IMF_int = simpson(imf_values, x=Masses)

    return IMF_int

def integrate_IMF(m_min, m_max, num_points=1000):
    """
    Integrate the Initial Mass Function (IMF) over a range of masses.

    Parameters:
    m_min (float): Minimum mass to integrate over.
    m_max (float): Maximum mass to integrate over.
    num_points (int): Number of points to use in the integration. Default is 1000.

    Returns:
    float: Integrated value of the IMF over the specified mass range.
    """
    # Generate logarithmically spaced mass values for better precision
    Masses = np.logspace(np.log10(m_min), np.log10(m_max), num_points)

    # Compute the IMF values for all masses
    imf_values = np.vectorize(imf_lin)(Masses)

    # Integrate using the composite Simpson's rule
    IMF_int = simpson(imf_values * Masses, x=Masses)

    return IMF_int

def read_SNIa_tables(filename):

    with h5py.File(filename, 'r') as data_file:
        stellar_yields = data_file["Yield"][:]

    return stellar_yields

def calculate_SNIa_yields(filename, indx, total_mass_fraction, NSNIa):
    # Let's read tables
    stellar_yields = read_SNIa_tables(filename)

    if indx is not None:
        # Select elements: Hydrogen, Helium, Neon, Magnesium, Silicon, Sulfur, Calcium, Nickel, Iron
        stellar_yields = stellar_yields[indx]

    # Calculate the minimum mass limit based on the inverse lifetime
    mass_limit = inverse_lifetime(total_mass_fraction)

    # Integrate the Initial Mass Function (IMF) over the specified mass range
    IMF_int = integrate_IMF(0.1, 100)

    # Define the mass range and calculate the IMF values
    masses = np.arange(mass_limit, 100, 0.1)
    IMF = np.vectorize(imf_lin)(masses)

    # Calculate the integral of the IMF weighted by the number of SNIa per solar mass
    integral_mej = simpson(IMF * NSNIa * masses, x=masses)

    # Compute the final EAGLE yields normalized by the IMF integral
    yields = stellar_yields * integral_mej / IMF_int

    return yields

def read_SNIa_COLIBRE(total_mass_fraction):
    """
    Calculate the integrated yields from Type Ia supernovae for the COLIBRE model.

    Parameters:
    total_mass_fraction (float): The total mass fraction for which the yields are calculated.

    Returns:
    ndarray: The integrated and normalized stellar yields for COLIBRE.
    """
    # Load stellar yields from HDF5 file
    filename = './data/LeungNomoto2018/SNIa_W7LeungNomoto2018.hdf5'

    # Select elements?
    indx = None

    # Calculate the integral of the IMF weighted by the number of SNIa per solar mass
    NSNIa = 1.6e-3  # Msun^-1

    colibre_yields = calculate_SNIa_yields(filename, indx, total_mass_fraction, NSNIa)

    return colibre_yields

def integrate_num_stars(m_min, m_max):

    Masses = np.arange(m_min, m_max)
    imf_values = np.vectorize(imf_lin)(Masses)
    int = simpson(imf_values, x=Masses)
    return int

def read_SNII_COLIBRE(metallicity_flag, metallicity, total_mass_fraction, indx):

    #['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron'
    # indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])

    factor = metallicity / total_mass_fraction
    boost_factors = np.array([1, 1, 1.5, 1, 1, 1, 1.5, 1, 0, 0, 1])
    boost_factors = boost_factors[indx]

    # Write data to HDF5
    with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Ejected_mass_winds = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file["/Yields/Z_"+metallicity_flag+"/Ejected_mass_in_ccsn"][:][:]

    num_mass_bins = len(Masses)
    stellar_yields = np.zeros(num_mass_bins)

    for i in range(num_mass_bins):
        stellar_yields[i] = (Ejected_mass_ccsn[indx, i] + factor * mass_fraction[indx] * Ejected_mass_winds[i]) * boost_factors

    new_mass_range = np.arange(8, 40.1, 0.1)
    imf_range = np.vectorize(imf_lin)(new_mass_range)
    f = interp1d(Masses, stellar_yields, bounds_error=False, fill_value="extrapolate")
    stellar_yields_range = f(new_mass_range)
    yields = simpson(imf_range * stellar_yields_range, x=new_mass_range)
    IMF_int = integrate_IMF(0.1, 100)
    yields /= IMF_int # Mass fraction [no units]

    Rcc = simpson(imf_range, x=new_mass_range) / IMF_int # Number of CCSNe per Msun [Msun^-1]
    average_yield = yields / Rcc # IMF-averaged yields [Msun]

    return yields, average_yield

def calculate_yields_SNIa(indx):

    Z_array = np.array([0,0.001, 0.004, 0.008, 0.02, 0.05])
    Fe_SNIa_yields = []
    for Zi in Z_array:
        Fe_SNIa_yields = np.append(Fe_SNIa_yields, read_SNIa_COLIBRE(Zi)[indx])

    Z_extended = np.arange(-5, np.log10(0.05), 0.01)
    Z_extended = 10 ** Z_extended

    f = interp1d(Z_array, Fe_SNIa_yields)
    Fe_yields_extended = f(Z_extended)


    return Fe_SNIa_yields, Fe_yields_extended, Z_extended


def calculate_yields(indx):

    solar_metallicity = 0.0134
    Z_array = np.array([0,0.001, 0.004, 0.008, 0.02, 0.05])

    yield_Z0p000, avg_yield_Z0p000 = read_SNII_COLIBRE('0.000',0.000, solar_metallicity, indx)
    yield_Z0p001, avg_yield_Z0p001 = read_SNII_COLIBRE('0.001',0.001, solar_metallicity, indx)
    yield_Z0p004, avg_yield_Z0p004 = read_SNII_COLIBRE('0.004',0.004, solar_metallicity, indx)
    yield_Z0p008, avg_yield_Z0p008 = read_SNII_COLIBRE('0.008',0.008, solar_metallicity, indx)
    yield_Z0p020, avg_yield_Z0p020 = read_SNII_COLIBRE('0.020',0.020, solar_metallicity, indx)
    yield_Z0p050, avg_yield_Z0p050 = read_SNII_COLIBRE('0.050', 0.050, solar_metallicity, indx)

    yields = np.array([yield_Z0p000, yield_Z0p001, yield_Z0p004, yield_Z0p008, yield_Z0p020, yield_Z0p050])
    avg_yields = np.array([avg_yield_Z0p000, avg_yield_Z0p001, avg_yield_Z0p004, avg_yield_Z0p008, avg_yield_Z0p020, avg_yield_Z0p050])
    Z_extended = np.arange(-5, np.log10(0.05), 0.01)
    Z_extended = 10**Z_extended

    f = interp1d(Z_array, yields)
    yields_extended = f(Z_extended)

    return yields, avg_yields, yields_extended, Z_extended


def calculate_yield_ratios():

    H_yields, H_avg_yields, H_yields_extended, Z_extended = calculate_yields(0)
    Fe_yields, Fe_avg_yields, Fe_yields_extended, Z_extended = calculate_yields(10)
    Mg_yields, Mg_avg_yields, Mg_yields_extended, Z_extended = calculate_yields(6)
    O_yields, O_avg_yields, O_yields_extended, Z_extended = calculate_yields(4)
    Si_yields, Si_avg_yields, Si_yields_extended, Z_extended = calculate_yields(7)

    Fe_SNIa_yields, Fe_SNIa_yields_extended, Z_extended = calculate_yields_SNIa(8)

    print("What about elements ratio? e.g. [Mg/Fe]=log10(mg/fe / Xmg/Xfe)")
    Xmg = 6.71e-4 ## Magg et al. (2022)
    Xfe = 13.7e-4
    Xsi = 8.75e-4
    XO = 73.3e-4
    # Xmg = 0.00070797838
    # Xfe = 0.00129199252
    # XO = 0.00573271036
    XH = 0.73738788833
    # Xsi = 0.00066495154

    norm_MgFe = np.log10(Xmg / Xfe)
    norm_OFe = np.log10(XO / Xfe)
    norm_SiFe = np.log10(Xsi / Xfe)
    norm_FeH = np.log10(Xfe / XH)

    print("It depends on metallicity: (Z=0, 0.001, 0.004, 0.008, 0.02)")
    print("Fe")
    ratios = np.log10(Fe_yields / Xfe)
    print(ratios)
    print(Fe_avg_yields)

    print('====')
    print("[Mg/Fe], Mg")
    ratios = np.log10(Mg_yields / Fe_yields) - norm_MgFe
    print(ratios)

    ratios = np.log10(Mg_yields / Xmg)
    # ratios = np.log10(Mg_yields / (Fe_yields + Fe_SNIa_yields)) - norm_MgFe
    print(ratios)
    print(Mg_avg_yields)

    print('====')
    print("[O/Fe], O")
    ratios = np.log10(O_yields / Fe_yields) - norm_OFe
    print(ratios)
    ratios = np.log10(O_yields / XO)
    # ratios = np.log10(O_yields / (Fe_yields + Fe_SNIa_yields))  - norm_OFe
    print(ratios)
    print(O_avg_yields)

    # print("[Si/Fe]=")
    # ratios = np.log10(Si_yields / Fe_yields) - norm_SiFe
    # print(ratios)
    # # ratios = np.log10(Si_yields / (Fe_yields + Fe_SNIa_yields)) - norm_SiFe
    # # print(ratios)
    # print('===')

    # factor_Fe = Z_extended / solar_metallicity
    # FeH_sun = np.log10(Xfe / XH)
    # FeH = np.log10(Xfe * factor_Fe / XH) - FeH_sun
    # ratio_all = np.log10(mg_yields_extended / iron_yields_extended) - norm
    # for i in range(num_bins):
    #     print(ratio_all[i], Z_extended[i], FeH[i])

    # with h5py.File('./data/Mean_CCSN_yields.hdf5', 'w') as data_file:
    #
    #     Header = data_file.create_group('Header')
    #     # description = "Add details here"
    #     # Header.attrs["Description"] = np.string_(description)
    #
    #     contact = "Dataset generated by Camila Correa (CEA Paris-Saclay). Email: camila.correa@cea.fr,"
    #     contact += " website: camilacorrea.com"
    #     Header.attrs["Contact"] = np.string_(contact)
    #
    #     data_file.create_dataset('FeH', data=FeH)
    #     data_file.create_dataset('Z', data=Z_extended)
    #     data_file.create_dataset('MgFe', data=ratio_all)


if __name__ == "__main__":

    # calculate_yield_ratios()

    # num_massive = integrate_num_stars(8, 100)
    num_all = integrate_num_stars(0.01, 100)
    #print(num_massive)
    print(num_all)

    # value = num_massive / num_all
    # print("Fraction of massive stars that explode as CCSN")
    # print(value)
    #
    IMF_all = integrate_IMF(0.01, 100)
    print(IMF_all)
    # num_massive = integrate_num_stars(8, 100)
    # value = num_massive / IMF_all
    # print("Number of massive stars per unit mass of star formation:")
    # print(value)

    print("Number of stars per unit mass of stellar particle:")
    value = num_all / IMF_all
    print(value)
    part_mass = 1e6
    print(value * 1e6)