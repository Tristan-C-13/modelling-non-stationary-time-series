import numpy as np
import matplotlib.pyplot as plt
from kernels import Kernel


def realization_tvAR1(X_0, T, alpha_fun, sigma_fun, epsilon):
    """
    Returns a (multidimensional) realization of a tvAR(1) process.

    --- parameters
    - X_0: initial value 
    - alpha_fun: alpha in the tvAR(1) expression. Defined over [0, 1]
    - sigma_fun: sigma in the tvAR(1) expression. Defined over [0, 1]
    - epsilon: noise generating the process. Can be multidimensional.
    """
    assert epsilon.shape[0] == T
    epsilon = epsilon.reshape(epsilon.shape[0], -1)
    X = np.empty(shape=(T, epsilon.shape[1]))
    X[0, :] = X_0
    
    for t in range(1, T):
        alpha_t = alpha_fun(t / T)
        sigma_t = sigma_fun(t / T)
        X[t, :] = X[t-1, :] * (-alpha_t) + sigma_t * epsilon[t, :]
        
    return X


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


def make_row_plot(alpha_fun, sigma_fun, alpha_hat, sigma_hat, u_list, subfig, n_realizations):
    axs = subfig.subplots(1, 4)

    # different realizations
    for i in range(n_realizations):
        axs[0].plot(u_list, alpha_hat[:, i], alpha=0.5)
        axs[1].plot(u_list, sigma_hat[:, i], alpha=0.5)
    
    # averages
    axs[2].plot(u_list, alpha_hat.mean(axis=1), label=f"Approximation")
    axs[3].plot(u_list, sigma_hat.mean(axis=1), label=f"Approximation")

    # true curves
    for i in range(0, 3, 2):
        axs[i].plot(u_list, alpha_fun(u_list), color="black", label="True curve")
        axs[i+1].plot(u_list, sigma_fun(u_list), color="black", label="True curve")

    # titles and legends
    axs[0].set_title(f"{n_realizations} realizations of " + r"$\hat{\alpha}$", fontsize=10)
    axs[1].set_title(f"{n_realizations} realizations of " + r"$\hat{\sigma}$", fontsize=10)
    axs[2].set_title(f"Mean over {alpha_hat.shape[1]} approximations of " + r"$\hat{\alpha}$", fontsize=10)
    axs[3].set_title(f"Mean over {sigma_hat.shape[1]} approximations of " + r"$\hat{\sigma}$", fontsize=10)


def make_row_plot_alpha(alpha_fun, alpha_hat, u_list, subfig, n_realizations):
    axs = subfig.subplots(1, 2)

    # different realizations
    for i in range(n_realizations):
        axs[0].plot(u_list, alpha_hat[:, i], alpha=0.5)
    # averages
    axs[1].plot(u_list, alpha_hat.mean(axis=1), label=f"Approximation")
    # true curves
    for ax in axs.flat:
        ax.plot(u_list, alpha_fun(u_list), color="black")
    # titles and legends
    axs[0].set_title(f"{n_realizations} realizations of " + r"$\hat{\alpha}$", fontsize=10)
    axs[1].set_title(f"Mean over {alpha_hat.shape[1]} approximations of " + r"$\hat{\alpha}$", fontsize=10)

def make_row_plot_sigma(sigma_fun, sigma_hat, u_list, subfig, n_realizations):
    axs = subfig.subplots(1, 2)

    # different realizations
    for i in range(n_realizations):
        axs[0].plot(u_list, sigma_hat[:, i], alpha=0.5)
    # averages
    axs[1].plot(u_list, sigma_hat.mean(axis=1), label=f"Approximation")
    # true curves
    for ax in axs.flat:
        ax.plot(u_list, sigma_fun(u_list), color="black")
    # titles and legends
    axs[0].set_title(f"{n_realizations} realizations of " + r"$\hat{\sigma}$", fontsize=10)
    axs[1].set_title(f"Mean over {sigma_hat.shape[1]} approximations of " + r"$\hat{\sigma}$", fontsize=10)





if __name__ == '__main__':
    def alpha(t):
        return -0.8 * np.cos(1.5 - np.cos(4 * np.pi * t))

    def sigma(t):
        return np.cos(t * np.pi / 2 + np.exp(t)) ** 2


    T_list = [100, 1000, 10000]
    u_list = np.linspace(0, 1, 100, endpoint=False)

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Convergence of the estimators as " + r"$T \rightarrow \infty$", fontsize=13)
    subfigs = fig.subfigures(nrows=len(T_list), ncols=1)
    # fig2, axs2 = plt.subplots(len(T_list), 1)

    for i in range(len(T_list)):
        T = T_list[i]
        bandwidth = 1 / (T ** (1/5))
        print(f"{T=}")
        print(f"{bandwidth=}")

        # Estimate coefficients
        alpha_hat, sigma_hat = estimate_parameters_tvAR1(T, 100, alpha, sigma, Kernel("uniform"), bandwidth, u_list)

        # Plot 
        subfigs[i].suptitle(f"{T=}", fontweight="bold")
        make_row_plot(alpha, sigma, alpha_hat, sigma_hat, u_list, subfigs[i], 5)
        # make_row_plot_alpha(alpha, alpha_hat, u_list, subfigs[i], 5)
        



        # epsilon = np.random.normal(0, 1, size=T)
        # X = realization_tvAR1(0, T, alpha, sigma, epsilon)

        # axs2[i].plot(np.linspace(0, 1, T), X)
        # axs2[i].vlines(0.4, ymin=-2, ymax=2, color='red')
        # axs2[i].vlines(max(0, 0.4 - bandwidth / 2), ymin=-2, ymax=2, color='red', linestyle='--')
        # axs2[i].vlines(min(1, 0.4 + bandwidth / 2), ymin=-2, ymax=2, color='red', linestyle='--')


    # legend
    lines = subfigs[0].axes[-1].get_lines()
    fig.legend(lines, ("Approximation", "True curve"), loc=(0.75, 0.96), ncol=2)
    
    # plt.savefig("convergence-yw-estimators.pdf")
    plt.show()
    print("end")

