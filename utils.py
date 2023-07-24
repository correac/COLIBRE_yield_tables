import numpy as np
from scipy import interpolate

def interpolate_data(x_data, y_data, x_bins):

    x_min = np.min(x_data)
    x_max = np.max(x_data)
    f = interpolate.interp1d(x_data, y_data, kind='linear', fill_value='extrapolate')
    fx = np.zeros(len(x_bins))

    for j, x in enumerate(x_bins):

        if (x >= x_min) & (x <= x_max):
            fx[j] = f(x)
        elif (x > x_max):
            fx[j] = y_data[-1] * x / x_max
        elif (x < x_min):
            fx[j] = y_data[0] * x / x_min

    return fx