import numpy as np

from .estimation import estimate_parameters_tvAR_p, forecast_future_values_tvAR_p, estimate_local_mean, estimate_local_autocovariance
from .kernels import Kernel
from .interpolation import extrapolate_parameters

class Portfolio:
    def __init__(self, start_date) -> None:
        self.pnl = 0
        self.pnl_dict = {start_date: self.pnl}
        self.positions = {'BTC-USD': 0, 'ETH-USD': 0}

    def insert_order(self, instrument_id:str, side:str, price:float, volume:float) -> None:
        assert side in ['BUY', 'SELL']
        sign = -1 if side == 'SELL' else 1
        self.positions[instrument_id] += sign * volume
        self.pnl -= sign * volume * price

    def close_positions(self, btc_close:float, eth_close:float, date:str) -> None:
        self.update_pnl(btc_close, eth_close, date)
        self.positions = {key: 0 for key in self.positions.keys()}

    def get_positions_value(self, btc_close:float, eth_close:float) -> float:
        return self.positions['BTC-USD'] * btc_close + self.positions['ETH-USD'] * eth_close

    def update_pnl(self, btc_close:float, eth_close:float, date:str) -> None:
        self.pnl += self.get_positions_value(btc_close, eth_close)
        self.pnl_dict[date] = self.pnl

    def get_last_pnl(self) -> float:
        return self.pnl

    def get_pnl_dict(self) -> list[float]:
        return self.pnl_dict


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
    b_T = 0.1 * T ** (-1/5)

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=p, u_list=u_list, kernel=Kernel(kernel_str), bandwidth=b_T)
    alpha_hat = theta_hat[:, :-1, :].squeeze(axis=2)
    sigma_hat = theta_hat[:, -1, :].squeeze()

    alpha_extrapolated, sigma_extrapolated = extrapolate_parameters(alpha_hat, sigma_hat, num_points=10, interpol_step=1, n_forecasts=1, k=k)
    x_star = forecast_future_values_tvAR_p(alpha_extrapolated, time_series)
    
    # moving_std = time_series[-24:].std()
    # moving_mean = time_series[-24:].mean()
    # z = (x_star - moving_mean) / moving_std

    # Localized version of the sample mean / std on the last 36 hours. The local point is chosen in the middle of the interval
    localized_mean = estimate_local_mean(time_series.reshape(-1, 1), T, Kernel("one-sided epanechnikov (L)"), b_T)   
    localized_std = np.sqrt(estimate_local_autocovariance(time_series.reshape(-1, 1), T, 0, Kernel("one-sided epanechnikov (L)"), b_T))
    z = (x_star - localized_mean) / localized_std

    if z > 1:
        return ('SHORT', x_star, z)  # sell A, buy B
    elif z < -1:
        return ('LONG', x_star, z)   # buy A, sell B
    else:
        return (None, x_star, z)


def get_actions_and_forecasts(time_series, n_forecasts=50, p=1, k=3):
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
        actions[i], forecasts[i], z_list[i] = check_entry_trade(time_series_i, p, k=k)
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
