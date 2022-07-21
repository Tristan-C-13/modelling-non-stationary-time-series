import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from utils.kernels import Kernel
from utils.estimation import estimate_parameters_tvAR_p, estimate_future_values_tvAR_p
from utils.interpolation import interpolate_and_extrapolate, plot_interpolation, extrapolate_parameters

sns.set_style("whitegrid")


if __name__ == '__main__':
    ## DATA & SPREAD
    # BTC-USD / ETH-USD
    data = yf.download("BTC-USD ETH-USD", period="1y", interval="1h") # period="1y", interval="1h"
    data = data['Close']
    data = data.dropna()
    data = np.log(1 + data.pct_change()) # log returns
    data = data.dropna()
    data['spread'] = data['BTC-USD'] - data['ETH-USD']

    ## MODELLING
    u_list = np.linspace(0, 1, 100, endpoint=False)
    time_series = data['spread'].to_numpy()
    T = time_series.shape[0]
    b_T = T ** (-1/5)
    p = 2
    print(f"{T=}")
    print(f"{b_T=}")

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=p, u_list=u_list, kernel=Kernel("epanechnikov"), bandwidth=b_T)
    alpha_hat = theta_hat[:, :-1, :].squeeze(axis=2)
    sigma_hat = theta_hat[:, -1, :].squeeze()

    ## PLOT
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(3, 1)
    # time series
    sns.lineplot(data=data, ax=subfigs[0].subplots(1, 1))
    # alpha
    axs = subfigs[1].subplots(1, p) if p > 1 else [subfigs[1].subplots(1, p)]
    for i in range(p):
        axs[i].plot(u_list, alpha_hat[:, i])
        axs[i].set_title(f"alpha_{i+1}")
    # sigma
    ax = subfigs[2].subplots(1, 1)
    ax.plot(u_list, sigma_hat)
    ax.set_title("sigma")


    # acf and pacf
    fig2, axs2 = plt.subplots(1, 2)
    pacf_, conf_int = pacf(time_series, nlags=10, alpha=0.05)
    plot_pacf(time_series, method='ywm', ax=axs2[0], lags=np.arange(10), alpha=0.05, markersize=3)
    plot_acf(time_series, ax=axs2[1], lags=np.arange(10), alpha=0.05, markersize=3)


    # interpolation
    fig, ax = plt.subplots()
    alpha_interpolation = interpolate_and_extrapolate(alpha_hat[:, 0])
    plot_interpolation(alpha_hat[:, 0], alpha_interpolation, ax)
    # forecast time series
    alpha_forecasts, sigma_forecasts = extrapolate_parameters(alpha_hat, sigma_hat, num_points=10, interpol_step=1, n_forecasts=1) # shape (n_forecasts, alpha_hat.shape[1])
    preds = estimate_future_values_tvAR_p(p, alpha_forecasts, sigma_forecasts, time_series)
    # plot forecast
    fig, ax = plt.subplots()
    ax.plot(np.arange(19, 20+len(preds)), np.concatenate([[time_series[-1]], preds]), color='red', label='prediction')
    ax.plot(np.arange(20), time_series[-20:], label='time series')
    ax.set_title("last 20 points of the time series + forecast")
    ax.legend()
    
    plt.show()