import numpy as np


def simulate_tvAR_p(p, X_0, epsilon, alpha_fun_list, sigma_fun):
    """
    Returns a (multidimensional) realisation of a tvAR(p) process.

    --- parameters
    - p: order of the model
    - X_0: initial value(s) 
    - epsilon: noise generating the process. Can be multidimensional.
    - alpha_fun_list: list containing the alpha coefficients in the tvAR(p) expression. All defined over [0, 1]. (alpha_1, ..., alpha_p)
    - sigma_fun: sigma in the tvAR(p) expression. Defined over [0, 1].
    """
    if not isinstance(X_0, np.ndarray):
        X_0 = np.array(X_0)
    if not isinstance(alpha_fun_list, list):
        alpha_fun_list = [alpha_fun_list]
    
    epsilon = epsilon.reshape(epsilon.shape[0], -1)
    X_0 = X_0.reshape(p, -1)
    T, n_realisations = epsilon.shape
    assert X_0.shape == (p, n_realisations)

    X = np.empty(shape=epsilon.shape)
    X[:p, :] = X_0

    for t in range(p, T):
        alpha_t_list = [alpha_fun(t / T) for alpha_fun in alpha_fun_list]
        sigma_t= sigma_fun(t / T)
        X[t, :] = np.sum(-X[t-p:t, :] * np.flip(alpha_t_list), axis=0) + sigma_t * epsilon[t, :]
        
    return X