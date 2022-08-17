import numpy as np
import pandas as pd
import scipy
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from utils.data_processing import get_dates_str, load_spread_btc_eth
from utils.kernels import Kernel
from utils.estimation import estimate_parameters_tvAR_p, forecast_future_values_tvAR_p 
from utils.interpolation import interpolate_and_extrapolate, plot_interpolation, extrapolate_parameters
from utils.trading import get_actions_and_forecasts, launch_trading_simulation1, launch_trading_simulation2, launch_trading_simulation3

sns.set_style("whitegrid")


##################
# BACKTESTS
##################
def plot_rolling_forecasts(time_series, n_forecasts=50, n_last_points=80, p=1, k=3):
    # parameters
    u_list = np.linspace(0, 1, 100, endpoint=False)
    T = time_series.shape[0] - n_forecasts
    b_T = T ** (-1/5)

    # define windows of time series. 1 time series has length T and is associated with 1 forecast
    windows = np.lib.stride_tricks.sliding_window_view(time_series, T)[:-1]
    ts_forecasts = np.empty(windows.shape[0])
    ts_forecasts_var = np.empty(windows.shape[0])
    ts_alpha_extrapolated = np.empty((windows.shape[0], p))
    ts_moving_std = np.empty(windows.shape[0])
    ts_moving_mean = np.empty(windows.shape[0])

    for i, time_series_i in enumerate(windows):
        theta_hat = estimate_parameters_tvAR_p(
            time_series=time_series_i, p=p, u_list=u_list, kernel=Kernel("epanechnikov"), bandwidth=b_T)
        alpha_hat = theta_hat[:, :-1, :].squeeze(axis=2)
        sigma_hat = theta_hat[:, -1, :].squeeze()

        alpha_extrapolated, sigma_extrapolated = extrapolate_parameters(
            alpha_hat, sigma_hat, num_points=10, interpol_step=1, n_forecasts=1, k=k)
        ts_forecasts[i] = forecast_future_values_tvAR_p(
            alpha_extrapolated, time_series_i)
        ts_forecasts_var[i] = sigma_extrapolated
        ts_alpha_extrapolated[i, :] = alpha_extrapolated
        ts_moving_std[i] = time_series_i[-24:].std()
        ts_moving_mean[i] = time_series_i[-24:].mean()

    print(scipy.stats.spearmanr(time_series[-n_forecasts:], ts_forecasts))

    # plot true time series + forecasts
    x = np.arange(n_last_points)
    fig, ax = plt.subplots()
    ax.plot(x, time_series[-n_last_points:], label="time series")
    ax.plot(x[-n_forecasts:], ts_forecasts, color='red', label='forecast')
    ax.plot(x[-n_forecasts:], ts_moving_std,
            color='black', label='moving std')
    ax.plot(x[-n_forecasts:], -ts_moving_std, color='black')
    # ax.fill_between(x[-n_forecasts:], -np.abs(ts_forecasts_var), np.abs(ts_forecasts_var), alpha=0.5, color="grey", label=r"$\pm \sigma^2$")
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(ts_alpha_extrapolated)
    ax.set_title("Extrapolation of alpha_1")

def plot_rolling_entries(time_series, spread_time_series, n_forecasts=50, n_last_points=80, p=1, k=3):
    (actions, forecasts, z_list) = get_actions_and_forecasts(time_series=time_series, n_forecasts=n_forecasts, p=p, k=k)
    print(scipy.stats.spearmanr(time_series[-n_forecasts:], forecasts))
    
    # Plot figure
    ## time series and forecasts
    fig, ax = plt.subplots()
    x = np.arange(n_last_points)
    ax.plot(x, time_series[-n_last_points:])
    ax.plot(x[-n_forecasts:], forecasts, color='red', label='forecasts')
    ax.vlines(x[-n_forecasts], time_series[-n_last_points:].min(),
              time_series[-n_last_points:].max(), color='red', linestyle='--', label='forecasts begin here')
    ax.scatter(x[-n_forecasts + np.argwhere(actions == 'LONG')] - 1,
               time_series[-n_forecasts + np.argwhere(actions == 'LONG') - 1], marker='^', color='green', label='Open LONG')
    ax.scatter(x[-n_forecasts + np.argwhere(actions == 'SHORT') - 1],
               time_series[-n_forecasts + np.argwhere(actions == 'SHORT') - 1], marker='v', color='black', label='Open SHORT')
    ax.set_title(f"Trade Signals ({k=}, {p=})")
    ax.legend()
    ## spread time series
    fig, ax = plt.subplots()
    ax.plot(x, spread_time_series[-n_last_points:])
    ax.scatter(x[-n_forecasts + np.argwhere(actions == 'LONG')] - 1,
               spread_time_series[-n_forecasts + np.argwhere(actions == 'LONG') - 1], marker='^', color='green', label='Open LONG')
    ax.scatter(x[-n_forecasts + np.argwhere(actions == 'SHORT') - 1],
               spread_time_series[-n_forecasts + np.argwhere(actions == 'SHORT') - 1], marker='v', color='black', label='Open SHORT')
    ## z
    fig, ax = plt.subplots()
    ax.plot(x[-n_forecasts:], z_list, color='green', label='z')


##################
# OTHER
##################
def make_general_plots(time_series, p=1, k=3):
    # MODELLING
    u_list = np.linspace(0, 1, 100, endpoint=False)
    T = time_series.shape[0]
    b_T = 0.1 * T ** (-1/5)
    print(f"{T=}")
    print(f"{b_T=}")

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=p, u_list=u_list, kernel=Kernel("epanechnikov"), bandwidth=b_T)
    alpha_hat = theta_hat[:, :-1, :].squeeze(axis=2)
    sigma_hat = theta_hat[:, -1, :].squeeze()

    # PLOT DATA TIME SERIES + COEFFICIENTS TIME SERIES
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(3, 1)
    # spread time series
    ax = subfigs[0].subplots(1, 1)
    ax.plot(time_series)
    ax.set_title("Spread")
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
    alpha_interpolation = interpolate_and_extrapolate(alpha_hat[:, 0], k=k)
    plot_interpolation(alpha_hat[:, 0], alpha_interpolation, ax)
    
    # PLOT forecast of the time series
    fig, ax = plt.subplots()
    alpha_extrapolated, sigma_extrapolated = extrapolate_parameters(alpha_hat, sigma_hat, num_points=10, interpol_step=1, n_forecasts=1, k=3) # shape (n_forecasts, alpha_hat.shape[1])
    preds = forecast_future_values_tvAR_p(alpha_extrapolated, time_series)
    ax.plot(np.arange(19, 20+len(preds)), np.concatenate([[time_series[-1]], preds]), color='red', label='prediction')
    ax.plot(np.arange(20), time_series[-20:], label='time series')
    ax.set_title("last 20 points of the time series + forecast")
    ax.legend()


# ========================================================================================


if __name__ == '__main__':
    # Parameters
    p = 1       # order of the tvAR(p)
    k = 3       # order of the spline interpolation
    T = 10_000  # length of each time series

    # DATA & SPREAD: BTC-USD / ETH-USD
    n_simulations = 500
    # start, end = get_dates_str(n_simulations + T - 1, '2022-08-14')
    start, end = get_dates_str(n_simulations + T - 1, '2021-12-31')
    data_df = load_spread_btc_eth(start, end)
    time_series = data_df['spread_log_returns'].to_numpy()
    spread_time_series = data_df['spread'].to_numpy()

    # PLOTS & ANALYSES  
    # make_general_plots(time_series, p=p, k=k)
    # plot_rolling_forecasts(time_series, p=p, k=k, n_forecasts=200, n_last_points=220)
    # plot_rolling_entries(time_series, spread_time_series, p=p, k=k, n_forecasts=200, n_last_points=220)

    
    # TRADING SIMULATION
    portfolio_1 = launch_trading_simulation1(data_df, T, p, k, 1, 'strat1-1000')
    portfolio_2 = launch_trading_simulation2(data_df, T, p, k, 1, 'strat2-1000')
    portfolio_3 = launch_trading_simulation3(data_df, T, p, k, 1, 0.25, 'strat3-1000')

    with (
        open('data/trading_simulations/strat1-1000.pickle', 'rb') as f1,
        open('data/trading_simulations/strat2-1000.pickle', 'rb') as f2,
        open('data/trading_simulations/strat3-1000.pickle', 'rb') as f3,
    ):
        portfolio_1 = pickle.load(f1)
        portfolio_2 = pickle.load(f2)
        portfolio_3 = pickle.load(f3)

    fig, axs = plt.subplots(3, 1)
    portfolio_1.plot_pnl_entries(axs[0])
    portfolio_2.plot_pnl_entries(axs[1])
    portfolio_3.plot_pnl_entries(axs[2])

    plt.show()