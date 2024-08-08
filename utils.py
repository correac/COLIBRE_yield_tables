import numpy as np
from scipy import interpolate

def interpolate_data(x_data, y_data, x_bins):
    """
    Interpolates and extrapolates data over specified bins.

    Parameters:
    x_data (ndarray): The x-values of the original data points.
    y_data (ndarray): The y-values of the original data points.
    x_bins (ndarray): The x-values at which interpolation is required.

    Returns:
    ndarray: Interpolated (and extrapolated) y-values at the specified x_bins.
    """

    # Create interpolation function with extrapolation
    f = interpolate.interp1d(x_data, y_data, kind='linear', fill_value='extrapolate')

    # Interpolate/extrapolate data
    fx = f(x_bins)

    # Handle extrapolation manually (this will overwrite the automatic extrapolation)
    x_min, x_max = np.min(x_data), np.max(x_data)

    fx[x_bins < x_min] = y_data[0] * x_bins[x_bins < x_min] / x_min
    fx[x_bins > x_max] = y_data[-1] * x_bins[x_bins > x_max] / x_max

    return fx
