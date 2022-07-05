import numpy as np
import matplotlib.pyplot as plt

from utils.kernels import Kernel
from utils.estimation import estimate_parameters_tvAR_p
from utils.simulate import realization_tvAR1


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


    # Parameters
    T_list = [100, 1000, 10000]
    u_list = np.linspace(0, 1, 100, endpoint=False)
    n_realizations = 200


    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Convergence of the estimators as " + r"$T \rightarrow \infty$", fontsize=13)
    subfigs = fig.subfigures(nrows=len(T_list), ncols=1)
    fig_alpha = plt.figure(constrained_layout=True)
    fig_alpha.suptitle("Convergence of " + r"$\hat{\alpha}$" + " as " + r"$T \rightarrow \infty$", fontsize=13)
    subfigs_alpha = fig_alpha.subfigures(nrows=len(T_list), ncols=1)
    fig_sigma = plt.figure(constrained_layout=True)
    fig_sigma.suptitle("Convergence of " + r"$\hat{\sigma}$" + " as " + r"$T \rightarrow \infty$", fontsize=13)
    subfigs_sigma = fig_sigma.subfigures(nrows=len(T_list), ncols=1)

    for i, T in enumerate(T_list):
        bandwidth = 1 / (T ** (1/5))
        print(f"{T=}")
        print(f"{bandwidth=}")

        # Estimate coefficients
        epsilon = np.random.normal(0, 1, size=(T, n_realizations))
        X = realization_tvAR1(np.zeros(n_realizations), epsilon, alpha, sigma)
        yw_estimates = estimate_parameters_tvAR_p(X, 1, u_list, Kernel("epanechnikov"), bandwidth)
        alpha_hat = yw_estimates[:, 0]
        sigma_hat = yw_estimates[:, 1]

        # Plot 
        subfigs[i].suptitle(f"{T=}", fontweight="bold")
        subfigs_alpha[i].suptitle(f"{T=}", fontweight="bold")
        subfigs_sigma[i].suptitle(f"{T=}", fontweight="bold")
        make_row_plot(alpha, sigma, alpha_hat, sigma_hat, u_list, subfigs[i], 5)
        make_row_plot_alpha(alpha, alpha_hat, u_list, subfigs_alpha[i], 5)
        make_row_plot_sigma(sigma, sigma_hat, u_list, subfigs_sigma[i], 5)


    # legend
    lines = subfigs[0].axes[-1].get_lines()
    fig.legend(lines, ("Approximation", "True curve"), loc=(0.75, 0.96), ncol=2)
    fig_alpha.legend(lines, ("Approximation", "True curve"), loc=(0.75, 0.96), ncol=2)
    fig_sigma.legend(lines, ("Approximation", "True curve"), loc=(0.75, 0.96), ncol=2)
    
    # plt.savefig("convergence-yw-estimators.pdf")
    plt.show()
