import numpy as np
from scipy import interpolate
from sklearn.metrics import r2_score


def interpolate_and_extrapolate(y, num_points=10, interpol_step=0.1, n_forecasts=1, k=3, only_extrapolated_pts=False):
    """
    Returns the spline interpolation of y at a specific step as well as some extrapolated points. 

    --- parameters
    - y: time series to be interpolated.
    - num_points: number of points of the time series to consider in the interpolation.
    - interpol_step: step between the interpolated values. interpol_step=1 simply gives the last num_points values of y and the n_forecasts interpolated points.
    - n_forecasts: number of points of y to be extrapolated.  
    - k: The degree of the spline fit.
    - only_extrapolated_pts: if True, only returns the extrapolated points on the same scale as the time series.
    """
    x = np.arange(num_points)
    y = y[-num_points:]
    spl = interpolate.splrep(x, y, k=k, s=0)
    x_new = np.arange(num_points + n_forecasts - 1 + interpol_step, step=interpol_step)
    interpolation = interpolate.splev(x_new, spl)
    if only_extrapolated_pts:
        return interpolation[::int(1 / interpol_step)][-n_forecasts:]
    else:
        return interpolation


def extrapolate_parameters(alpha, sigma, num_points=10, interpol_step=1, n_forecasts=1, k=3):
    """
    Returns the extrapolated values of alpha and sigma.
    Shape of alpha: (n_forecasts, p). Shape of sigma: (n_forecasts)

    --- parameters
    - alpha: time series of the alpha coefficient.
    - sigma: time series of the sigma coefficient.
    - all other parameters are used in the interpolate_and_extrapolate function.
    """
    alpha_extrapolated = np.empty((n_forecasts, alpha.shape[1]))
    sigma_extrapolated = interpolate_and_extrapolate(sigma, num_points, interpol_step, n_forecasts, k, True)
    for i in range(alpha.shape[1]):
        alpha_extrapolated[:, i] = interpolate_and_extrapolate(alpha[:, i], num_points, interpol_step, n_forecasts, k, True)
    return alpha_extrapolated, sigma_extrapolated


def plot_interpolation(y, y_interpolation, ax, num_points=10, interpol_step=0.1, n_forecasts=1):
    """
    Plots the interpolation of the time series y and the true series on the same graph.
    """
    x = np.linspace(0, len(y)-1, num=len(y))
    ax.scatter(x, y, label='true points')

    x_interpolation = np.arange(x[int(len(y) - (n_forecasts + num_points)) + 1], x[-1] + n_forecasts + interpol_step, step=interpol_step)
    ax.plot(x_interpolation, y_interpolation, color='red', linestyle='--', label='spline interpolation')
    ax.legend()


def select_interpolation_order(y):
    r2_dict = dict()
    test_mask = np.random.choice([False, True], size=(y.shape[0],), p=[0.9, 0.1])
    x = np.arange(y.shape[0])
    x_train = x[~test_mask]
    x_test = x[test_mask]
    y_train = y[~test_mask]
    y_test = y[test_mask]

    for k in range(1, 6):
        spl = interpolate.splrep(x_train, y_train, k=k)
        interpolation = interpolate.splev(x, spl)
        r2_dict[k] = r2_score(y_test, interpolation[x_test])
    
    return r2_dict

