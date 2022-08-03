import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
import datetime as dt
import logging
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from utils.kernels import Kernel
from utils.estimation import estimate_parameters_tvAR_p, forecast_future_values_tvAR_p
from utils.interpolation import interpolate_and_extrapolate, plot_interpolation, extrapolate_parameters, select_interpolation_order
from utils.trading import Portfolio

sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO)


##################
# DATA PREPARATION
##################
def get_dates_str(num_hours, end=None):
    """
    Returns two string dates separated by num_hours hours. Note, there will be **approximately** num_hours between the two dates.

    --- parameters
    - num_hours: number of hours between the two dates.
    - end: end date. If end is None, then it is today's date.
    """
    if end is None:
        end = dt.date.today()
    else:
        end = dt.datetime.strptime(end, "%Y-%m-%d").date()
    start = end - dt.timedelta(hours=num_hours)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def download_and_prepare_data(symbol1:str, symbol2:str, **kwargs) -> pd.DataFrame:
    """
    Returns a pd.DataFrame containing the log returns of two securities as well as their spread.

    --- parameters
    - symbol1, symbol2: symbol of the financial securities.
    - **kwargs: parameters used in yf.download.
    """
    data_df = yf.download(f"{symbol1} {symbol2}", **kwargs) 
    data_df = data_df['Close']
    data_df = data_df.dropna()
    prices_df = data_df.copy()
    prices_df.columns = ['btc_close', 'eth_close']
    data_df = np.log(1 + data_df.pct_change()) # log returns
    data_df = pd.concat([data_df, prices_df], axis=1)
    data_df = data_df.dropna()
    data_df['spread'] = data_df['BTC-USD'] - data_df['ETH-USD']

    # logging info
    start_datetime = data_df.index[0]
    end_datetime = data_df.index[-1]
    logging.info(f"Data downloaded: from {start_datetime} to {end_datetime}")
    return data_df


##################
# TRADES / SIGNALS
##################
def check_entry_trade(time_series, p=1, kernel_str="epanechnikov", k=3):
    """
    For a given time series, returns the forecasted entry trade action. It is one of three possibilities:
    * 'SHORT': sell A, buy B
    * 'LONG'; buy A, sell B
    * None: no signal found.

    --- parameters
    Same parameters that are used in the estimation and extrapolation of the coefficients for a tvAR(p) process.
    """
    u_list = np.linspace(0, 1, 100, endpoint=False)
    T = time_series.shape[0]
    b_T = T ** (-1/5)

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=p, u_list=u_list, kernel=Kernel(kernel_str), bandwidth=b_T)
    alpha_hat = theta_hat[:, :-1, :].squeeze(axis=2)
    sigma_hat = theta_hat[:, -1, :].squeeze()

    alpha_extrapolated, sigma_extrapolated = extrapolate_parameters(alpha_hat, sigma_hat, num_points=10, interpol_step=1, n_forecasts=1, k=k)
    x_star = forecast_future_values_tvAR_p(alpha_extrapolated, time_series)
    
    moving_std = time_series[-24:].std()
    moving_mean = time_series[-24:].mean()
    z = (x_star - moving_mean) / moving_std

    if z > 1:
        return ('SHORT', x_star, z)  # sell A, buy B
    elif z < -1:
        return ('LONG', x_star, z)   # buy A, sell B
    else:
        return (None, x_star, z)


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

def plot_rolling_entries(time_series, n_forecasts=50, n_last_points=80, p=1, k=3):
    # Define windows of time series. 1 time series has length T and is associated with 1 forecast
    T = time_series.shape[0] - n_forecasts
    windows = np.lib.stride_tricks.sliding_window_view(time_series, T)[:-1]
    actions = np.empty(windows.shape[0], dtype=object)
    z_list = np.empty(windows.shape[0], dtype=object)
    forecasts = np.empty(windows.shape[0])
    for i, time_series_i in enumerate(windows):
        actions[i], forecasts[i], z_list[i] = check_entry_trade(time_series_i, p, k=k)
    
    # Plot figure
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

    fig, ax = plt.subplots()
    ax.plot(x[-n_forecasts:], z_list, color='green', label='z')

##################
# OTHER
##################
def make_general_plots(data):
    # MODELLING
    time_series = data['spread'].to_numpy()
    u_list = np.linspace(0, 1, 100, endpoint=False)
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
    alpha_interpolation = interpolate_and_extrapolate(alpha_hat[:, 0], k=3)
    plot_interpolation(alpha_hat[:, 0], alpha_interpolation, ax)
    
    # PLOT forecast of the time series
    fig, ax = plt.subplots()
    alpha_extrapolated, sigma_extrapolated = extrapolate_parameters(alpha_hat, sigma_hat, num_points=10, interpol_step=1, n_forecasts=1, k=3) # shape (n_forecasts, alpha_hat.shape[1])
    preds = forecast_future_values_tvAR_p(alpha_extrapolated, time_series)
    ax.plot(np.arange(19, 20+len(preds)), np.concatenate([[time_series[-1]], preds]), color='red', label='prediction')
    ax.plot(np.arange(20), time_series[-20:], label='time series')
    ax.set_title("last 20 points of the time series + forecast")
    ax.legend()




def launch_trading_simulation(n_days=2000, p=1, k=3):
    """
    --- parameters
    - n_days: Number of days of simulated trading
    - p: Order of the tvAR(p) model
    - k: Order of the spline interpolation
    """
    # Data & Spread: BTC-USD / ETH-USD
    start, end = get_dates_str(10000 + n_days) 
    data_df = pd.read_csv("data/spread.csv")

    # data_df = download_and_prepare_data("BTC-USD", "ETH-USD", start=start, end=end, interval="1h")
    spread_time_series = data_df['spread'].to_numpy()

    # Initialize the portfolio: 0 BTC and 0 ETH
    # portfolio = Portfolio(dt.datetime.fromtimestamp(data_df.index[0].timestamp()).strftime("%Y-%m-%d %H:%M:%S"))
    portfolio = Portfolio(data_df.index[0])

    T = spread_time_series.shape[0] - n_days
    for i in range(n_days):
        date = dt.datetime.fromtimestamp(data_df.index[i].timestamp()).strftime("%Y-%m-%d %H:%M:%S")
        time_series_i = spread_time_series[i : T+i]
        btc_close = data_df.loc[data_df.index[T+i-1], 'btc_close']
        eth_close = data_df.loc[data_df.index[T+i-1], 'eth_close']

        # Close previous positions
        portfolio.close_positions(btc_close, eth_close, date)

        # Forecast and get trade signal
        action, _, _ = check_entry_trade(time_series_i, p, k=k)

        # Pass an order if there is a trade signal
        if action is not None:
            side_btc = 'BUY' if action == 'LONG' else 'SELL'
            side_eth = 'BUY' if action == 'SHORT' else 'SELL'
            portfolio.insert_order('BTC-USD', side=side_btc, price=btc_close, volume=1)
            portfolio.insert_order('ETH-USD', side=side_eth, price=eth_close, volume=1)
            portfolio.pnl_dict[date] = portfolio.pnl

    pnl_dict = portfolio.get_pnl_dict()
    print(portfolio.pnl)
    # print(pd.Series(pnl_dict))



if __name__ == '__main__':
    # Parameters
    p = 1
    k = 3

    # DATA & SPREAD: BTC-USD / ETH-USD
    start, end = get_dates_str(10000 + 50) 
    print(f"start: {start}", f"end: {end}", sep='\n')
    # data_df = download_and_prepare_data("BTC-USD", "ETH-USD", start=start, end=end, interval="1h")
    # data_df.to_csv("data/spread.csv", index=False)
    data_df = pd.read_csv("data/spread.csv")
    spread_time_series = data_df['spread'].to_numpy()

    # PLOTS & ANALYSES  
    # make_general_plots(data_df)
    # plot_rolling_forecasts(spread_time_series, p=p, k=k, n_forecasts=200, n_last_points=220)
    # plot_rolling_entries(spread_time_series, p=p, k=k, n_forecasts=200, n_last_points=220)
    
    
    # TRADING SIMULATION
    launch_trading_simulation(p=p, k=k)

    plt.show()



