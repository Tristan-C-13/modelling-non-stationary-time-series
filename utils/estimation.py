import numpy as np
import scipy.linalg

from .kernels import Kernel


def estimate_autocovariance(X, t_0, k, kernel, bandwidth):
    """
    Returns an estimate of the autocovariance of X at t_0 and lag k.
    Can be multidimensional if X represents several realizations.

    --- parameters
    - X: array representing one (or several) realizations of the process
    - t_0: point where estimate the covariance
    - k: lag
    - kernel: localizing kernel used in the estimation
    - bandwidth: bandwidth of the kernel
    """
    T = X.shape[0]
    return np.sum(
        (
            kernel((t_0 - (np.arange(T-k) + k / 2)) / (bandwidth * T)).repeat(X.shape[1]).reshape(X[k:, :].shape) 
            * X[:T-k, :] 
            * X[k:, :]
        ), axis=0) / (bandwidth * T)


def estimate_yw_coef(c_list):
    """
    Estimates the Yule-Walker estimates for an AR(p) model. Supports multi-dimensional time series (for Monte Carlo simulations).

    --- parameters
    - c_list: autocovariance sequence ((c_0, ..., c_p), ...)
    """
    yw_coeff = np.empty(shape=c_list.shape) # (p+1, n_realizations)

    for i in range(c_list.shape[1]):
        gamma = c_list[1:, i]
        Gamma = scipy.linalg.toeplitz(c=c_list[:-1, i], r=c_list[:-1, i])
        alpha_hat = -np.dot(scipy.linalg.inv(Gamma), gamma) 
        sigma_hat = np.sqrt(c_list[0, i] + np.dot(gamma, alpha_hat))
        yw_coeff[:, i] = np.concatenate((alpha_hat, [sigma_hat]), axis=0)
    
    return yw_coeff


def estimate_parameters_tvAR_p(time_series: np.ndarray, p: int, u_list: np.ndarray, kernel: Kernel, bandwidth: float):
    """
    Returns the Yule-Walker estimates of a tvAR(p) model. Supports multi-dimensional time series for Monte-Carlo simulations.

    --- parameters
    - time_series: time series with shape (T, n_realizations)
    - p: order of the tvAR(p) model
    - u_list: list of points between 0 and 1 where to evaluate the parameters.
    - kernel: kernel used for the autocovariance approximation
    - bandwidth: bandwidth used in the non-parametric estimation
    """
    T = time_series.shape[0]
    time_series = time_series.reshape(T, -1) # make sure time series shape is (T, n_realizations)
    estimates = np.empty(shape=(u_list.shape[0], p + 1, time_series.shape[1])) # (alpha_1, ..., alpha_p, sigma) for each time series at each point u

    for i, u_0 in enumerate(u_list):
        # define the window
        t_0 = int(u_0 * T)
        start = max(0, int(t_0 - bandwidth * T / 2))
        end = min(T-1, int(t_0 + bandwidth * T / 2))
        time_series_window = time_series[start:(end+1), :]

        # estimate the covariance and Yule-Walker estimates
        c_list = np.array([estimate_autocovariance(time_series_window, t_0 - start, k, kernel, bandwidth) for k in range(p+1)]).reshape((p+1, -1))
        estimates[i, :, :] = estimate_yw_coef(c_list)

    return estimates


def forecast_future_values_tvAR_p(alpha_forecasts, time_series):
    # returns (n_forecasts,)
    n_forecasts, p = alpha_forecasts.shape
    forecasts = np.empty((n_forecasts,))
    p_last_values = time_series[-p:]
    for i in range(n_forecasts):
        x_star = -np.sum(p_last_values * np.flip(alpha_forecasts[i, :]), axis=0) # alpha needs to be flipped to have (X_{t-p} * alpha_p, ...) and not (X_{t-p} * alpha_1, ...)
        forecasts[i] = x_star
        p_last_values = np.concatenate([p_last_values[1:], [x_star]]) 
    return forecasts


def multistep_forecast_tvAR_1(alpha, time_series, n_forecasts):
    """
    In expectation: X_t = -alpha X_{t-1} ==> X_{t+k} = (-alpha)^k * X_t
    """
    return [time_series[-k] * (-alpha) ** k for k in range(1, n_forecasts + 1)]
