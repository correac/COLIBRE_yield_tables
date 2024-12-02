import h5py
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from scipy.integrate import quad, romberg
from scipy.integrate import cumtrapz, romb, simpson
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as tck
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
                          0.00129199252]) #Fe

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


def lifetimes(m, Z, filename='./data/EAGLE_yieldtables/Lifetimes.hdf5'):
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


def read_AGB_Yields(filename, metallicity_flag):
    # indx == ['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron']
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
    with h5py.File(filename, 'r') as data_file:
        Masses = data_file["Masses"][:]
        Yield = data_file[f"/Yields/Z_{metallicity_flag}/Yield"][indx, :]
        Ejected_mass = data_file[f"/Yields/Z_{metallicity_flag}/Ejected_mass"][:]
        Total_metals = data_file[f"/Yields/Z_{metallicity_flag}/Total_Metals"][:]
    return Masses, Yield, Ejected_mass, Total_metals


def compute_AGB_stellar_yields(Masses, Yield, Ejected_mass, Total_metals, factor, mass_fraction, total_mass_fraction):
    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    stellar_yields = np.zeros((num_elements, num_mass_bins))
    total_yields = np.zeros(num_mass_bins)

    for i in range(num_mass_bins):
        stellar_yields[:, i] = Yield[:, i] + factor * mass_fraction * Ejected_mass[i]
        total_yields[i] = Total_metals[i] + factor * total_mass_fraction * Ejected_mass[i]

    return stellar_yields, total_yields


def interpolate_yields(Masses, stellar_yields, mass_range, elements=9):
    num_mass_bins = len(mass_range)
    new_imf_range = np.zeros(num_mass_bins)
    new_stellar_yields_range = np.zeros(num_mass_bins)
    interpolated_yields = np.zeros(elements)

    for i in range(elements):
        f = interp1d(Masses, stellar_yields[i, :], bounds_error=False, fill_value="extrapolate")
        for j in range(num_mass_bins):
            new_imf_range[j] = imf_lin(mass_range[j])
            new_stellar_yields_range[j] = f(mass_range[j])

        interpolated_yields[i] = simpson(new_imf_range * new_stellar_yields_range, x=mass_range)

    return interpolated_yields


def read_AGB_COLIBRE(metallicity_flag, metallicity, total_mass_fraction, mass_fraction):
    factor = metallicity / total_mass_fraction
    Masses, Yield, Ejected_mass, Total_metals = read_AGB_Yields('./data/AGB.hdf5', metallicity_flag)
    stellar_yields, _ = compute_AGB_stellar_yields(Masses, Yield, Ejected_mass, Total_metals, factor, mass_fraction,
                                               total_mass_fraction)

    minimum_mass = inverse_lifetime(total_mass_fraction)
    new_mass_range = np.arange(minimum_mass, 8, 0.1)
    colibre = interpolate_yields(Masses, stellar_yields, new_mass_range)

    return colibre


def read_AGB_EAGLE(metallicity_flag, metallicity, total_mass_fraction, mass_fraction):
    factor = metallicity / total_mass_fraction
    Masses, Yield, Ejected_mass, Total_metals = read_AGB_Yields('./data/EAGLE_yieldtables/AGB.hdf5', metallicity_flag)
    stellar_yields, _ = compute_AGB_stellar_yields(Masses, Yield, Ejected_mass, Total_metals, factor, mass_fraction,
                                               total_mass_fraction)

    minimum_mass = inverse_lifetime(total_mass_fraction)
    new_mass_range = np.arange(minimum_mass, 6, 0.1)
    eagle = interpolate_yields(Masses, stellar_yields, new_mass_range)

    return eagle

def plot_AGB(total_mass_fraction):

    IMF_int = integrate_IMF(0.1, 100)

    colibre_Z0p14 = read_AGB_COLIBRE('0.014',0.014, total_mass_fraction, mass_fraction)
    colibre_Z0p3 = read_AGB_COLIBRE('0.03',0.03, total_mass_fraction, mass_fraction)
    Z_colibre = np.array([0.014, 0.03])

    eagle_Z0p02 = read_AGB_EAGLE('0.019',0.019, total_mass_fraction, mass_fraction)
    eagle_Z0p008 = read_AGB_EAGLE('0.008',0.008, total_mass_fraction, mass_fraction)
    Z_eagle = np.array([0.008,0.019])

    # Interpolate and normalize data
    colibre = interpolate_and_normalize(Z_colibre, colibre_Z0p14, colibre_Z0p3, total_mass_fraction, IMF_int)
    eagle = interpolate_and_normalize(Z_eagle, eagle_Z0p008, eagle_Z0p02, total_mass_fraction, IMF_int)

    tng = np.array([0.2, 0.07, 0.0012, 0.0008, 0.0015, 0.00045, 0.00015, 0.00018, 0.0003])
    illustris = np.array([0.17, 6e-2, 0.0012, 6.5e-4, 1.3e-3, 4e-4, 1.2e-4, 1.6e-4, 2.5e-4])

    plot_bar_chart(colibre, eagle, tng, illustris, '===AGB Ratios===')


def read_SNII_COLIBRE(metallicity_flag, metallicity, total_mass_fraction):
    # Indices for the elements
    # indx == ['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron']
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
    factor = metallicity / total_mass_fraction

    # Read data from HDF5 file
    with h5py.File('./data/SNII_linear_extrapolation.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Ejected_mass_winds = data_file[f"/Yields/Z_{metallicity_flag}/Ejected_mass_in_winds"][:]
        Ejected_mass_ccsn = data_file[f"/Yields/Z_{metallicity_flag}/Ejected_mass_in_ccsn"][:]

    # Initialize arrays
    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    stellar_yields = np.zeros((num_elements, num_mass_bins))
    boost_factors = np.array([1, 1, 1.5, 1, 1, 1, 1.5, 1, 1])

    # Calculate stellar yields
    for i in range(num_mass_bins):
        stellar_yields[:, i] = (Ejected_mass_ccsn[indx, i] + factor * mass_fraction * Ejected_mass_winds[i]) * boost_factors

    # Interpolation and integration
    new_mass_range = np.arange(8, 40.1, 0.1)
    colibre = calculate_yields(Masses, stellar_yields, new_mass_range)

    return colibre

def read_SNII_EAGLE(metallicity_flag, metallicity, total_mass_fraction):
    # Indices for the elements
    # indx == ['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron']
    indx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
    factor = metallicity / total_mass_fraction

    # Read data from HDF5 file
    with h5py.File('./data/EAGLE_yieldtables/SNII.hdf5', 'r') as data_file:
        Masses = data_file["Masses"][:]
        Yield = data_file[f"/Yields/Z_{metallicity_flag}/Yield"][:]
        Ejected_mass = data_file[f"/Yields/Z_{metallicity_flag}/Ejected_mass"][:]

    # Initialize arrays
    num_mass_bins = len(Masses)
    num_elements = len(mass_fraction)
    stellar_yields = np.zeros((num_elements, num_mass_bins))
    boost_factors = np.array([1, 1, 0.5, 1, 1, 1.0, 2.0, 1.0, 0.5])

    # Calculate stellar yields
    for i in range(num_mass_bins):
        stellar_yields[:, i] = (Yield[indx, i] + factor * mass_fraction * Ejected_mass[i]) * boost_factors

    # Interpolation and integration
    new_mass_range = np.arange(6, 100, 0.1)
    eagle = calculate_yields(Masses, stellar_yields, new_mass_range)

    return eagle


def calculate_yields(Masses, stellar_yields, mass_range):
    """
    Calculate interpolated and integrated stellar yields over a given mass range.

    Parameters:
    Masses (ndarray): The original mass bins.
    stellar_yields (ndarray): The stellar yields corresponding to the original mass bins.
    new_mass_range (ndarray): The new mass range to interpolate and integrate yields over.

    Returns:
    yields (ndarray): The integrated yields for each element over the new mass range.
    """
    num_elements = stellar_yields.shape[0]

    imf_range = np.vectorize(imf_lin)(mass_range)
    yields = np.zeros(num_elements)

    for i in range(num_elements):
        f = interp1d(Masses, stellar_yields[i, :], bounds_error=False, fill_value="extrapolate")
        stellar_yields_range = f(mass_range)
        yields[i] = simpson(imf_range * stellar_yields_range, x=mass_range)

    return yields


def interpolate_and_normalize(Z, yields_low, yields_high, total_mass_fraction, IMF_int):
    """
    Interpolate yields between two metallicities and normalize by the integrated IMF.

    Parameters:
    Z (ndarray): The array of metallicity values corresponding to yields_low and yields_high.
    yields_low (ndarray): The yields at the lower metallicity.
    yields_high (ndarray): The yields at the higher metallicity.
    total_mass_fraction (float): The target metallicity for interpolation.
    IMF_int (float): The integrated Initial Mass Function for normalization.

    Returns:
    interpolated_yields (ndarray): The interpolated and normalized yields.
    """
    num_bins = len(yields_low)
    interpolated_yields = np.zeros(num_bins)

    # Interpolate and normalize yields for each bin
    for i in range(num_bins):
        f = interp1d(Z, [yields_low[i], yields_high[i]], kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolated_yields[i] = f(total_mass_fraction) / IMF_int

    return interpolated_yields

def plot_SNII(total_mass_fraction):

    IMF_int = integrate_IMF(0.1, 100)
    Z = np.array([0.008,0.02])

    # Read data for different metallicities
    colibre_Z0p02 = read_SNII_COLIBRE('0.020', 0.02, total_mass_fraction)
    colibre_Z0p008 = read_SNII_COLIBRE('0.008', 0.008, total_mass_fraction)
    eagle_Z0p02 = read_SNII_EAGLE('0.02', 0.02, total_mass_fraction)
    eagle_Z0p008 = read_SNII_EAGLE('0.008', 0.008, total_mass_fraction)

    # Interpolate and normalize data
    colibre = interpolate_and_normalize(Z, colibre_Z0p008, colibre_Z0p02, total_mass_fraction, IMF_int)
    eagle = interpolate_and_normalize(Z, eagle_Z0p008, eagle_Z0p02, total_mass_fraction, IMF_int)

    # Data from TNG and Illustris
    tng = np.array([8e-2, 6.5e-2, 2.5e-3, 5e-4, 1.5e-2, 5e-3, 1.5e-3, 1.1e-3, 8e-4])
    illustris = np.array([1.1e-1, 8e-2, 8.5e-3, 9e-4, 1.7e-2, 2.1e-3, 5e-4, 1.8e-3, 1.5e-3])

    # Plotting
    plot_bar_chart(colibre, eagle, tng, illustris, '===CCSN Ratios===')


def plot_bar_chart(colibre, eagle, tng, illustris, label):
    elements = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']
    color_list = ['lightskyblue', 'steelblue', 'khaki', 'salmon']
    bar_width = 0.25
    index = np.arange(9) * 1.5

    plt.bar(index, colibre, bar_width, color=color_list[0], label='COLIBRE', edgecolor='black', linewidth=0.2)
    plt.bar(index + bar_width, eagle, bar_width, color=color_list[1], label='EAGLE', edgecolor='black', linewidth=0.2)
    plt.bar(index + 2 * bar_width, tng, bar_width, color=color_list[2], label='TNG', edgecolor='black', linewidth=0.2)
    plt.bar(index + 3 * bar_width, illustris, bar_width, color=color_list[3], label='Illustris', edgecolor='black', linewidth=0.2)

    # Print ratios for comparison
    print(label)
    for i, elem in enumerate(elements):
        print(f'{elem}: EAGLE/COLIBRE: {eagle[i]/colibre[i]:.2f} ({colibre[i]/eagle[i]:.2f}), '
              f'TNG/COLIBRE: {tng[i]/colibre[i]:.2f} ({colibre[i]/tng[i]:.2f}), '
              f'Illustris/COLIBRE: {illustris[i]/colibre[i]:.2f} ({colibre[i]/illustris[i]:.2f})')

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

def read_SNIa_EAGLE(total_mass_fraction):
    """
    Calculate the integrated yields from Type Ia supernovae for the EAGLE model.

    Parameters:
    total_mass_fraction (float): The total mass fraction for which the yields are calculated.

    Returns:
    ndarray: The integrated and normalized stellar yields for EAGLE.
    """

    filename = './data/EAGLE_yieldtables/SNIa.hdf5'

    # Select elements: Hydrogen, Helium, Neon, Magnesium, Silicon, Sulfur, Calcium, Nickel, Iron
    indx = np.array([0, 1, 5, 6, 7, 9, 11, 13, 25])

    # Calculate the integral of the IMF weighted by the number of SNIa per solar mass
    NSNIa = 2e-3  # Msun^-1

    eagle_yields = calculate_SNIa_yields(filename, indx, total_mass_fraction, NSNIa)

    return eagle_yields

def read_SNIa_COLIBRE(total_mass_fraction):
    """
    Calculate the integrated yields from Type Ia supernovae for the COLIBRE model.

    Parameters:
    total_mass_fraction (float): The total mass fraction for which the yields are calculated.

    Returns:
    ndarray: The integrated and normalized stellar yields for COLIBRE.
    """
    # Load stellar yields from HDF5 file
    filename = './data/SNIa.hdf5'

    # Select elements?
    indx = None

    # Calculate the integral of the IMF weighted by the number of SNIa per solar mass
    NSNIa = 1.6e-3  # Msun^-1

    colibre_yields = calculate_SNIa_yields(filename, indx, total_mass_fraction, NSNIa)

    return colibre_yields

def plot_SNIa(solar_metallicity):

    colibre = read_SNIa_COLIBRE(solar_metallicity)
    eagle = read_SNIa_EAGLE(solar_metallicity)

    tng = np.array([1e-8, 1e-8, 6e-5, 1e-8, 2e-4, 6e-6, 1.5e-5, 2e-4, 1e-3])
    illustris = np.array([1e-8, 1e-8, 3e-5, 1e-8, 8e-5, 2e-6, 8e-6, 9e-5, 4.5e-4])

    # Plotting
    plot_bar_chart(colibre, eagle, tng, illustris, '===SNIa Ratios===')


def make_comparison():

    solar_metallicity = 0.014

    # Plot parameters
    params = {
        "font.size": 13,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (10, 3.8),
        "figure.subplot.left": 0.07,
        "figure.subplot.right": 0.99,
        "figure.subplot.bottom": 0.1,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.03,
        "figure.subplot.hspace": 0.03,
        "lines.markersize": 3,
        "lines.linewidth": 1,
        "figure.max_open_warning": 0,
        "axes.axisbelow": True,
    }
    rcParams.update(params)

    ######
    plt.figure()

    ax = plt.subplot(1, 3, 1)
    ax.grid(True)
    plt.grid(which='major', linestyle='-',linewidth=0.3)
    plt.grid(which='minor', linestyle=':',linewidth=0.3)

    plot_AGB(solar_metallicity)
    index = np.arange(9) * 1.5
    bar_width = 0.25

    plt.ylabel('Returned Stellar Mass Fraction')
    plt.xticks(index + bar_width + 0.11, ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe'))

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.35, 0.96, 'AGB yields', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.axis([-0.5, 13.5, 1e-6, 0.3])
    plt.yscale('log')
    plt.legend(loc=[0.65, 0.72], labelspacing=0.2, handlelength=0.8,
               handletextpad=0.3, frameon=False, columnspacing=0.4,
               ncol=1, fontsize=12)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    ########################
    ########################
    ax = plt.subplot(1, 3, 2)
    ax.grid(True)
    plt.grid(which='major', linestyle='-',linewidth=0.3)
    plt.grid(which='minor', linestyle=':',linewidth=0.3)
    plot_SNII(solar_metallicity)

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.35, 0.96, 'CCSN yields', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.xticks(index + bar_width + 0.11, ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe'))
    plt.axis([-0.5, 13.5, 1e-6, 0.3])
    plt.yscale('log')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_yaxis().set_ticklabels([])

    ########################
    ########################
    ax = plt.subplot(1, 3, 3)
    ax.grid(True)
    plt.grid(which='major', linestyle='-',linewidth=0.3)
    plt.grid(which='minor', linestyle=':',linewidth=0.3)
    plot_SNIa(solar_metallicity)

    props = dict(boxstyle='round', fc='white', ec='black', alpha=1)
    ax.text(0.35, 0.96, 'SNIa yields', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.xticks(index + bar_width + 0.11, ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe'))
    plt.axis([-0.5, 13.5, 1e-6, 0.3])
    plt.yscale('log')
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    ax.get_yaxis().set_ticklabels([])

    plt.savefig('./figures/Comparison_literature_all.png', dpi=300)


if __name__ == "__main__":

    make_comparison()
