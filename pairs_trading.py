import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

from utils.kernels import Kernel
from utils.estimation import estimate_parameters_tvAR_p
from utils.simulate import realization_tvAR1

sns.set_style("whitegrid")




if __name__ == '__main__':
    # data = yf.download("BTC-USD ETH-USD", period="6mo", interval="1h")
    # data = data['Close']
    # data = data.dropna()
    # data['spread'] = data['BTC-USD'] - data['ETH-USD']
    # # data = data.pct_change()

    # sns.lineplot(data=data)
    # plt.show()


    ## MODELLING
    def alpha(t):
        return -0.8 * np.cos(1.5 - np.cos(4 * np.pi * t))

    def sigma(t):
        return np.cos(t * np.pi / 2 + np.exp(t)) ** 2

    epsilon = np.random.normal(0, 1, size=(10000 , 5))
    time_series = realization_tvAR1(0, epsilon, alpha, sigma)



    u_list = np.linspace(0, 1, 100, endpoint=False)
    # time_series = data['spread'].to_numpy().reshape(-1, 1)
    T = time_series.shape[0]
    b_T = T ** (-1/5)

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=1, u_list=u_list, kernel=Kernel("epanechnikov"), bandwidth=b_T)
    plt.plot(theta_hat[:, 0])
    plt.plot(theta_hat[:, 1])
    plt.show()


