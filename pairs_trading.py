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


def download_and_prepare_data(symbol1:str, symbol2:str, **kwargs) -> pd.DataFrame:
    """
    Returns a pd.DataFrame containing the log returns of two securities as well as their spread.

    --- parameters
    - symbol1, symbol2: symbol of the financial security.
    - **kwargs: parameters used in yf.download.
    """
    data_df = yf.download(f"{symbol1} {symbol2}", **kwargs) 
    data_df = data_df['Close']
    data_df = data_df.dropna()
    data_df = np.log(1 + data_df.pct_change()) # log returns
    data_df = data_df.dropna()
    data_df['spread'] = data_df['BTC-USD'] - data_df['ETH-USD']
    return data_df


if __name__ == '__main__':
    # DATA & SPREAD: BTC-USD / ETH-USD
    data = download_and_prepare_data("BTC-USD", "ETH-USD", period="1y", interval="1h")

    # MODELLING
    u_list = np.linspace(0, 1, 100, endpoint=False)
    time_series = data['spread'].to_numpy()
    T = time_series.shape[0]
    b_T = T ** (-1/5)
    p = 1
    print(f"{T=}")
    print(f"{b_T=}")

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=p, u_list=u_list, kernel=Kernel("epanechnikov"), bandwidth=b_T)
    alpha_hat = theta_hat[:, :-1, :].squeeze(axis=2)
    sigma_hat = theta_hat[:, -1, :].squeeze()

    # PLOT DATA TIME SERIES + COEFFICIENTS TIME SERIES
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

    # PLOT ACF / PACF
    fig2, axs2 = plt.subplots(1, 2)
    pacf_, conf_int = pacf(time_series, nlags=10, alpha=0.05)
    plot_pacf(time_series, method='ywm', ax=axs2[0], lags=np.arange(10), alpha=0.05, markersize=3)
    plot_acf(time_series, ax=axs2[1], lags=np.arange(10), alpha=0.05, markersize=3)

    # PLOT Interpolation for one coefficient
    fig, ax = plt.subplots()
    alpha_interpolation = interpolate_and_extrapolate(alpha_hat[:, 0], k=2)
    plot_interpolation(alpha_hat[:, 0], alpha_interpolation, ax)
    
    # PLOT forecast of the time series
    fig, ax = plt.subplots()
    alpha_extrapolated, sigma_extrapolated = extrapolate_parameters(alpha_hat, sigma_hat, num_points=10, interpol_step=1, n_forecasts=1, k=3) # shape (n_forecasts, alpha_hat.shape[1])
    preds = estimate_future_values_tvAR_p(p, alpha_extrapolated, sigma_extrapolated, time_series)
    ax.plot(np.arange(19, 20+len(preds)), np.concatenate([[time_series[-1]], preds]), color='red', label='prediction')
    ax.plot(np.arange(20), time_series[-20:], label='time series')
    ax.set_title("last 20 points of the time series + forecast")
    ax.legend()
    
    plt.show()