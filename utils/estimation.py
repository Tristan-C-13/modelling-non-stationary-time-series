import numpy as np
import scipy.linalg

from .kernels import Kernel


def estimate_local_autocovariance(X: np.ndarray, t_0: int, k: int, kernel: Kernel, bandwidth: float) -> np.ndarray:
    """
    Returns an estimate of the autocovariance of X at t_0 and lag k.
    Can be multidimensional if X represents several realisations.

    --- parameters
    - X: array representing one (or several) realisations of the process
    - t_0: point where to estimate the covariance
    - k: lag
    - kernel: localizing kernel used in the estimation
    - bandwidth: bandwidth of the kernel
    """
    T = X.shape[0]
    return np.sum(
        (
            kernel(-(t_0 - (np.arange(T-k) + k/2)) / (bandwidth * T)).repeat(X.shape[1]).reshape(X[k:, :].shape) 
            * X[:T-k, :] 
            * X[k:, :]
        ), axis=0) / (bandwidth * T)

def estimate_local_mean(X: np.ndarray, t_0: int, kernel: Kernel, bandwidth: float) -> np.ndarray:
    """
    Returns an estimate of the mean of X at t_0 and lag k.
    Can be multidimensional if X represents several realisations.

    --- parameters
    - X: array representing one (or several) realisations of the process
    - t_0: point where to estimate the mean
    - kernel: localizing kernel used in the estimation
    - bandwidth: bandwidth of the kernel
    """
    T = X.shape[0]
    return np.sum(
        (
            kernel(-(t_0 - np.arange(T)) / (bandwidth * T)).repeat(X.shape[1]).reshape(X.shape)
            * X 
        ), axis=0) / (bandwidth * T)


def estimate_yw_coef(c_list: np.ndarray) -> np.ndarray:
    """
    Estimates the Yule-Walker estimates for an AR(p) model. Supports multi-dimensional time series (for Monte Carlo simulations).

    --- parameters
    - c_list: autocovariance sequence ((c_0, ..., c_p), ...)
    """
    yw_coeff = np.empty(shape=c_list.shape) # (p+1, n_realisations)

    for i in range(c_list.shape[1]):
        gamma = c_list[1:, i]
        Gamma = scipy.linalg.toeplitz(c=c_list[:-1, i], r=c_list[:-1, i])
        alpha_hat = -np.dot(scipy.linalg.inv(Gamma), gamma) 
        sigma_hat = np.sqrt(np.abs(c_list[0, i] + np.dot(gamma, alpha_hat)))   # should always be positive but just in case
        yw_coeff[:, i] = np.concatenate((alpha_hat, [sigma_hat]), axis=0)
    
    return yw_coeff


def estimate_parameters_tvAR_p(time_series: np.ndarray, p: int, u_list: np.ndarray, kernel: Kernel, bandwidth: float) -> np.ndarray:
    """
    Returns the Yule-Walker estimates of a tvAR(p) model. Supports multi-dimensional time series for Monte-Carlo simulations.

    --- parameters
    - time_series: time series with shape (T, n_realisations)
    - p: order of the tvAR(p) model
    - u_list: list of points between 0 and 1 where to evaluate the parameters.
    - kernel: kernel used for the autocovariance approximation
    - bandwidth: bandwidth used in the non-parametric estimation
    """
    T = time_series.shape[0]
    time_series = time_series.reshape(T, -1) # make sure time series shape is (T, n_realisations)
    estimates = np.empty(shape=(u_list.shape[0], p + 1, time_series.shape[1])) # (alpha_1, ..., alpha_p, sigma) for each time series at each point u

    for i, u_0 in enumerate(u_list):
        t_0 = int(u_0 * T)
        c_list = np.array([estimate_local_autocovariance(
            time_series, t_0, k, kernel, bandwidth) for k in range(p+1)]).reshape((p+1, -1))
        estimates[i, :, :] = estimate_yw_coef(c_list)

    return estimates


def forecast_future_values_tvAR_p(alpha_forecasts: np.ndarray, time_series: np.ndarray) -> np.ndarray:
    """
    Returns the (multi-step) forecasts of a tvAR(p) process given alpha. Return is of shape (n_forecasts,)
    If the number of required forecasts is greater than 1, the previous forecast will be used to make the next one.

    --- parameters
    - alpha_forecasts: alpha coefficients of the tvAR(p) model. (alpha_1, ..., alpha_p) shape (n_forecasts, p).
    - time_series: time series to be forecasted. Only the last p values are used.
    """
    n_forecasts, p = alpha_forecasts.shape
    forecasts = np.empty((n_forecasts,))
    p_last_values = time_series[-p:]
    for i in range(n_forecasts):
        x_star = -np.sum(p_last_values * np.flip(alpha_forecasts[i, :]), axis=0) # alpha needs to be flipped to have (X_{t-p} * alpha_p, ...) and not (X_{t-p} * alpha_1, ...)
        forecasts[i] = x_star
        p_last_values = np.concatenate([p_last_values[1:], [x_star]]) 
    return forecasts


def multistep_forecast_tvAR_1(alpha: float, time_series: np.ndarray, n_forecasts: int) -> list:
    """
    In expectation: X_t = -alpha X_{t-1} ==> X_{t+k} = (-alpha)^k * X_t
    """
    return [time_series[-k] * (-alpha) ** k for k in range(1, n_forecasts + 1)]
