import numpy as np
import pandas as pd

from .data_processing import get_dates_str
from .estimation import estimate_parameters_tvAR_p, forecast_future_values_tvAR_p, estimate_local_mean, estimate_local_autocovariance
from .kernels import Kernel
from .interpolation import extrapolate_parameters

class Portfolio:
    def __init__(self, start_date) -> None:
        self.cash = 0     # initial cash
        self.crypto = 0   # no initial position
        self.pnl = self.cash + self.crypto
        self.pnl_dict = {start_date: self.pnl}
        self.positions = {'BTC-USD': 0, 'ETH-USD': 0}

    def insert_order(self, instrument_id:str, side:str, price:float, volume:float) -> None:
        assert side in ['BUY', 'SELL']
        sign = -1 if side == 'SELL' else 1
        self.positions[instrument_id] += sign * volume
        self.cash -= sign * volume * price
        self.crypto += sign * volume * price
        # self.pnl -= sign * volume * price

    def close_positions(self, btc_close:float, eth_close:float, date) -> None:
        # self.update_pnl(btc_close, eth_close, date)
        self.cash += self.crypto
        self.crypto = 0
        self.positions = {key: 0 for key in self.positions.keys()}

    def get_positions_value(self, btc_close:float, eth_close:float) -> float:
        return self.positions['BTC-USD'] * btc_close + self.positions['ETH-USD'] * eth_close

    def update_crypto(self, btc_close: float, eth_close: float) -> None:
        self.crypto = self.get_positions_value(btc_close, eth_close)

    def update_pnl(self) -> None:
        # self.pnl += self.get_positions_value(btc_close, eth_close)
        # self.pnl_dict[date] = self.pnl
        self.pnl = self.crypto + self.cash

    def get_pnl_dict(self) -> list[float]:
        return self.pnl_dict


##################
# TRADES / SIGNALS
##################
def get_forecast_zscore(time_series, p=1, k=3, kernel_str="epanechnikov"):
    """
    For a given time series, returns the forecasted value and the associated z-score, used to determine if a trade is entered or not.

    --- parameters
    Same parameters that are used in the estimation and extrapolation of the coefficients for a tvAR(p) process.
    """
    u_list = np.linspace(0, 1, 100, endpoint=False)
    T = time_series.shape[0]
    b_T = 0.1 * T ** (-1/5)

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=p, u_list=u_list, kernel=Kernel(kernel_str), bandwidth=b_T)
    alpha_hat = theta_hat[:, :-1, :].squeeze(axis=2)
    sigma_hat = theta_hat[:, -1, :].squeeze()

    alpha_extrapolated, sigma_extrapolated = extrapolate_parameters(alpha_hat, sigma_hat, num_points=10, interpol_step=1, n_forecasts=1, k=k)
    x_star = forecast_future_values_tvAR_p(alpha_extrapolated, time_series)
    
    # Localized version of the sample mean / std on the last 36 hours. The local point is chosen in the middle of the interval
    localized_mean = estimate_local_mean(time_series.reshape(-1, 1), T, Kernel("one-sided epanechnikov (L)"), b_T)   
    localized_std = np.sqrt(estimate_local_autocovariance(time_series.reshape(-1, 1), T, 0, Kernel("one-sided epanechnikov (L)"), b_T))
    z = (x_star - localized_mean) / localized_std

    return (x_star, z)


def check_entry_trade(z: float, threshold: float):
    """
    Determine the entry trade action. It is one of three possibilities:
    * 'SHORT': sell A, buy B
    * 'LONG'; buy A, sell B
    * None: no signal found

    --- parameters
    - z: value of the z-score
    - threshold: threshold to enter a trade
    """
    if z > threshold:
        return 'SHORT'  # sell A, buy B
    elif z < -threshold:
        return 'LONG'   # buy A, sell B
    else:
        return None


def get_actions_and_forecasts(time_series, threshold=1, n_forecasts=50, p=1, k=3):
    """
    Returns a tuple of arrays:
    * actions: whether to enter a traded or not.
    * forecasts: forecasted values of the time series.
    * z_list: localized z-score associated with the actions.
    """
    # Define windows of time series. 1 time series has length T and is associated with 1 forecast
    T = time_series.shape[0] - n_forecasts
    windows = np.lib.stride_tricks.sliding_window_view(time_series, T)[:-1]
    actions = np.empty(n_forecasts, dtype=object)
    z_list = np.empty(n_forecasts)
    forecasts = np.empty(n_forecasts)
    for i, time_series_i in enumerate(windows):
        forecast, z_score = get_forecast_zscore(time_series_i, p, k)
        action = check_entry_trade(z_score, threshold)
        actions[i] = action
        forecasts[i] = forecast
        z_list[i] = z_score
    return (actions, forecasts, z_list)


##################
# METRICS
##################
def hit_ratio(time_series_spread_log_returns, forecasts, actions):
    """
    Returns the ratio (number of time the forecast had the right direction) / (number of forecasts)
    time_series.shape[0] >= forecasts.shape[0] + 1
    """
    n_forecasts = forecasts.shape[0]
    time_series_spread_log_returns = time_series_spread_log_returns[-n_forecasts:]
    idx_long = np.argwhere(actions == 'LONG')
    idx_short = np.argwhere(actions == 'SHORT')
    n_short = forecasts[idx_short].shape[0]
    n_long = forecasts[idx_long].shape[0]

    ratio_short = np.mean(np.sign(forecasts[idx_short]) == np.sign(time_series_spread_log_returns[idx_short]))
    ratio_long = np.mean(np.sign(forecasts[idx_long]) == np.sign(time_series_spread_log_returns[idx_long]))
    ratio = (n_short * ratio_short + n_long * ratio_long) / (n_short + n_long)    
    return {'ratio': ratio, 'short ratio': ratio_short, 'long ratio': ratio_long}


########################
# BACKTESTS / STRATEGIES    
########################
def launch_trading_simulation1(n_hours=2000, p=1, k=3):
    """
    Trading simulations on n_hours. 
    Enter a position if abs(z_score) > 1 and unwind it the next hour.

    --- parameters
    - n_hours: Number of hours of simulated trading
    - p: Order of the tvAR(p) model
    - k: Order of the spline interpolation
    """
    # Data & Spread: BTC-USD / ETH-USD
    start, end = get_dates_str(10000 + n_hours) 
    data_df = pd.read_csv("data/data.csv", index_col='datetime', parse_dates=True)

    # data_df = download_and_prepare_data("BTC-USD", "ETH-USD", start=start, end=end, interval="1h")
    spread_time_series = data_df['spread_log_returns'].to_numpy()

    # Initialize the portfolio: 0 BTC and 0 ETH
    portfolio = Portfolio(data_df.index[0])

    T = spread_time_series.shape[0] - n_hours
    for i in range(n_hours):
        date = data_df.index[i]
        time_series_i = spread_time_series[i : T+i]
        btc_close = data_df.loc[data_df.index[T+i-1], 'btc_close']
        eth_close = data_df.loc[data_df.index[T+i-1], 'eth_close']

        # Update crypto value
        portfolio.update_crypto(btc_close, eth_close)
        # Close previous positions
        portfolio.close_positions(btc_close, eth_close, date)
        # Forecast and get trade signal
        _, z_score = get_forecast_zscore(time_series_i, p, k)
        action = check_entry_trade(z_score, 1)
        # Pass an order if there is a trade signal
        if action is not None:
            side_btc = 'BUY' if action == 'LONG' else 'SELL'
            side_eth = 'BUY' if action == 'SHORT' else 'SELL'
            portfolio.insert_order('BTC-USD', side=side_btc, price=btc_close, volume=1)
            portfolio.insert_order('ETH-USD', side=side_eth, price=eth_close, volume=1)
        # Update P&L and save it 
        portfolio.update_pnl()
        portfolio.pnl_dict[date] = portfolio.pnl

        # Verbose
        if i % 50 == 0:
            print(f"step {i+1} / {n_hours}")

    # Close the final positions at the end of the trading period
    portfolio.close_positions(btc_close, eth_close, date)

    # Save result
    pnl_dict = portfolio.get_pnl_dict()
    pnl_series = pd.Series(pnl_dict)
    pnl_series.to_csv('../data/pnl_series2.csv', index_label='datetime')