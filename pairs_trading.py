import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf

from utils.kernels import Kernel
from utils.estimation import estimate_parameters_tvAR_p

sns.set_style("whitegrid")


if __name__ == '__main__':
    # DATA & SPREAD
    
    # BTC-USD / ETH-USD
    data = yf.download("BTC-USD ETH-USD", period="1y", interval="1h")
    data = data['Close']
    data = data.dropna()
    data = np.log(1 + data.pct_change()) # log returns
    data = data.dropna()
    data['spread'] = data['BTC-USD'] - data['ETH-USD']
    # data['ratio'] = data['BTC-USD'] / data['ETH-USD']

    # PEP / COKE
    # data = yf.download("PEP COKE", period="6mo", interval="1h")
    # data = data['Close']
    # data = data.dropna()
    # data['spread'] = data['PEP'] - data['COKE']
    
    # DPZ / PZZA
    # data = yf.download("DPZ PZZA", period="6mo", interval="1h")
    # data = data['Close']
    # data = data.dropna()
    # data['spread'] = data['DPZ'] - data['PZZA']

    # MS / GS
    # data = yf.download("MS GS", period="6mo", interval="1h")
    # data = data['Close']
    # data = data.dropna()
    # data['spread'] = data['MS'] - data['GS']



    ## MODELLING
    u_list = np.linspace(0, 1, 100, endpoint=False)
    time_series = data['spread'].to_numpy()
    T = time_series.shape[0]
    b_T = T ** (-1/5)
    p = 1
    print(f"{T=}")
    print(f"{b_T=}")

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=p, u_list=u_list, kernel=Kernel("epanechnikov"), bandwidth=b_T)
    alpha_hat = theta_hat[:, :-1, :]
    sigma_hat = theta_hat[:, -1, :]

    ## PLOT
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(3, 1)
    # time series
    sns.lineplot(data=data, ax=subfigs[0].subplots(1, 1))
    # alpha
    axs = subfigs[1].subplots(1, p) if p > 1 else [subfigs[1].subplots(1, p)]
    for i in range(p):
        axs[i].plot(u_list, alpha_hat[:, i, :])
        axs[i].set_title(f"alpha_{i+1}")
    # sigma
    ax = subfigs[2].subplots(1, 1)
    ax.plot(u_list, sigma_hat)
    ax.set_title("sigma")

    fig2, ax2 = plt.subplots()
    pacf, conf_int = pacf(time_series, nlags=10, alpha=0.05)
    plot_pacf(time_series, ax2, lags=np.arange(10), alpha=0.05)
    
    plt.show()