import numpy as np


def simulate_tvAR_p(p, X_0, epsilon, alpha_fun_list, sigma_fun):
    """
    Returns a (multidimensional) realization of a tvAR(p) process.

    --- parameters
    - p: order of the model
    - X_0: initial value 
    - epsilon: noise generating the process. Can be multidimensional.
    - alpha_fun_list: list containing the alpha coefficients in the tvAR(p) expression. All defined over [0, 1].
    - sigma_fun: sigma in the tvAR(p) expression. Defined over [0, 1].
    """
    if not isinstance(X_0, np.ndarray):
        X_0 = np.array(X_0)
    if not isinstance(alpha_fun_list, list):
        alpha_fun_list = [alpha_fun_list]
    
    epsilon = epsilon.reshape(epsilon.shape[0], -1)
    X_0 = X_0.reshape(p, -1)
    T, n_realizations = epsilon.shape
    assert X_0.shape == (p, n_realizations)

    X = np.empty(shape=epsilon.shape)
    X[:p, :] = X_0

    for t in range(p, T):
        alpha_t_list = [alpha_fun(t / T) for alpha_fun in alpha_fun_list]
        sigma_t= sigma_fun(t / T)
        X[t, :] = np.sum(-X[t-p:t, :] * alpha_t_list, axis=0) + sigma_t * epsilon[t, :]
        
    return X



    
if __name__ == '__main__':
    def alpha(t):
        return -0.8 * np.cos(1.5 - np.cos(4 * np.pi * t))

    def sigma(t):
        return np.cos(t * np.pi / 2 + np.exp(t)) ** 2


    # Parameters
    T = 500
    u_list = np.linspace(0, 1, 100, endpoint=False)
    n_realizations = 2
    np.random.seed(1234)
    epsilon = np.random.normal(0, 1, size=T)
    X = simulate_tvAR_p(1, 0, epsilon, alpha, sigma)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(X)
    plt.show()