import numpy as np

from .simulate import realization_tvAR1


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


def estimate_parameters_tvAR1(T, n_approximations, alpha_fun, sigma_fun, kernel, bandwidth, u_list):
    """
    Estimates the curves alpha and sigma of n_approximations realizations of a tvAR(1) process.
    
    --- parameters
    - T: length of the process
    - u_list: list of points between 0 and 1 where to evaluate the parameters.
    - n_approximations: number of realizations of the stationary process used for the MC approximation.
    """
    alpha_hat = np.empty((len(u_list), n_approximations))
    sigma_hat = np.empty((len(u_list), n_approximations))

    for i, u_0 in enumerate(u_list):
        t_0 = int(u_0 * T)
        start = max(0, int(t_0 - bandwidth * T / 2))
        end = min(T-1, int(t_0 + bandwidth * T / 2))

        # generate n_approximations realizations of X
        epsilon = np.random.normal(0, 1, size=(T, n_approximations))
        X = realization_tvAR1(np.zeros(n_approximations), T, alpha_fun, sigma_fun, epsilon)
        X_window = X[start:(end+1), :]

        # estimate alpha_hat and sigma_hat by taking the average of the approximations
        c_1 = estimate_autocovariance(X_window, t_0 - start, 1, kernel, bandwidth)
        c_0 = estimate_autocovariance(X_window, t_0 - start, 0, kernel, bandwidth)
        alpha_hat[i] = (-c_1 / c_0)
        sigma_hat[i] = np.sqrt(c_0 - c_1 * c_1 / c_0) # = sqrt(c_0 + alpha * c_1)

    return alpha_hat, sigma_hat