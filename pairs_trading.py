import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

from utils.kernels import Kernel
from utils.estimation import estimate_parameters_tvAR_p

sns.set_style("whitegrid")


if __name__ == '__main__':
    ## DATA & SPREAD
    data = yf.download("BTC-USD ETH-USD", period="6mo", interval="1h")
    data = data['Close']
    data = data.dropna()
    data['spread'] = data['BTC-USD'] - data['ETH-USD']

    # data = yf.download("PEP COKE", period="6mo", interval="1h")
    # data = data['Close']
    # data = data.dropna()
    # data['spread'] = data['PEP'] - data['COKE']
    
    # data = yf.download("DPZ PZZA", period="6mo", interval="1h")
    # data = data['Close']
    # data = data.dropna()
    # data['spread'] = data['DPZ'] - data['PZZA']


    ## MODELLING
    u_list = np.linspace(0, 1, 100, endpoint=False)
    time_series = data['spread'].to_numpy()
    T = time_series.shape[0]
    b_T = T ** (-1/6)
    print(f"{T=}")
    print(f"{b_T=}")

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=1, u_list=u_list, kernel=Kernel("epanechnikov"), bandwidth=b_T)
    alpha_hat = theta_hat[:, :-1, :]
    sigma_hat = theta_hat[:, -1, :]

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(time_series, label="spread")
    axs[1].plot(alpha_hat[:, 0], label="alpha")
    axs[1].plot(sigma_hat, label="sigma")
    for ax in axs.flat:
        ax.legend()
    plt.show()


