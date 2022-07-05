import numpy as np


def realization_tvAR1(X_0, epsilon, alpha_fun, sigma_fun):
    """
    Returns a (multidimensional) realization of a tvAR(1) process.

    --- parameters
    - X_0: initial value 
    - alpha_fun: alpha in the tvAR(1) expression. Defined over [0, 1]
    - sigma_fun: sigma in the tvAR(1) expression. Defined over [0, 1]
    - epsilon: noise generating the process. Can be multidimensional.
    """
    T = epsilon.shape[0]
    epsilon = epsilon.reshape(epsilon.shape[0], -1)
    X = np.empty(shape=(T, epsilon.shape[1]))
    X[0, :] = X_0
    
    for t in range(1, T):
        alpha_t = alpha_fun(t / T)
        sigma_t = sigma_fun(t / T)
        X[t, :] = X[t-1, :] * (-alpha_t) + sigma_t * epsilon[t, :]
        
    return X