from xmlrpc.client import Boolean
import numpy as np
import pandas as pd

import logging
import pickle
from abc import ABC, abstractclassmethod
from dataclasses import dataclass

from .data_processing import reflect_time_series
from .estimation import estimate_parameters_tvAR_p, forecast_future_values_tvAR_p, estimate_local_mean, estimate_local_autocovariance
from .kernels import Kernel
from .interpolation import extrapolate_parameters

class Portfolio:
    """Object saving all data from a trading simulation and used to pass orders."""

    def __init__(self, start_date, end_date) -> None:
        self.positions = {'BTC-USD': 0, 'ETH-USD': 0}  
        self.cash = 0     # initial cash
        self.crypto = 0   # no initial position
        self.pnl = self.cash + self.crypto
        # History 
        index = pd.date_range(start=start_date, end=end_date, freq='1H')
        self.history = pd.DataFrame(
            index=index,
            data={
                'pnl': np.nan,
                'trade_actions': [[] for _ in range(len(index))],
                'forecast': np.nan,
                'true_value': np.nan, # to be initialized by hand
                'z_score': np.nan
            }
        )

    def insert_order(self, instrument_id: str, side: str, price: float, volume: float) -> None:
        assert side in ['BUY', 'SELL']
        sign = -1 if side == 'SELL' else 1
        self.positions[instrument_id] += sign * volume
        self.cash -= sign * volume * price
        self.crypto += sign * volume * price
        self.update_pnl()

    def close_positions(self, date: str) -> None:
        if self.hold_positions:
            self.add_action(date, 'CLOSE')
            self.cash += self.crypto
            self.crypto = 0
            self.positions = {key: 0 for key in self.positions.keys()}

    def get_positions_value(self, btc_close: float, eth_close: float) -> float:
        return self.positions['BTC-USD'] * btc_close + self.positions['ETH-USD'] * eth_close

    def update_crypto(self, btc_close: float, eth_close: float) -> None:
        self.crypto = self.get_positions_value(btc_close, eth_close)
        self.update_pnl()

    def update_pnl(self) -> None:
        self.pnl = self.crypto + self.cash

    def update_history(self, date: str, action: str, forecast: float, z_score: float) -> None:
        self.history.at[date, 'pnl'] = self.pnl
        if action is not None:
            self.history.at[date, 'trade_actions'].append(action)
        self.history.at[date, 'forecast'] = forecast
        self.history.at[date, 'z_score'] = z_score

    def add_action(self, date: str, action: str) -> None:
        self.history.at[date, 'trade_actions'].append(action)

    @property
    def hold_positions(self):
        return any(self.positions.values())

    @property
    def holding_position(self):
        if not self.hold_positions:
            return None
        else:
            if self.positions['BTC-USD'] < 0 and self.positions['ETH-USD'] > 0:
                return 'SHORT'
            elif self.positions['BTC-USD'] > 0 and self.positions['ETH-USD'] < 0:
                return 'LONG'
            else:
                return None


    def plot_pnl_entries(self, ax, show_actions=True):
        pnl_series = self.history['pnl'].to_numpy()
        min_ = pnl_series.min()
        max_ = pnl_series.max()
        marker_ = {'LONG': 'v', 'SHORT': '^'}
        color_ = {'LONG': 'black', 'SHORT': 'green'}
        ax.plot(pnl_series, label='P&L')

        if show_actions:
            for i, action in enumerate(self.history['trade_actions'].to_numpy()):
                if action:
                    for a in action:
                        if a in ['SHORT', 'LONG']:
                            ax.scatter(i, min_, marker=marker_[a], color=color_[a])
                        else:
                            ax.scatter(i, max_, marker='x', color='red')

        ax.legend();


##################
# TRADES / SIGNALS
##################
def get_forecast_zscore(time_series, p=1, k=3, kernel_str="epanechnikov", reflect_time_series_bool=False):
    """
    For a given time series, returns the forecasted value and the associated z-score, used to determine if a trade is entered or not.

    --- parameters
    Same parameters that are used in the estimation and extrapolation of the coefficients for a tvAR(p) process.
    """
    u_list = np.linspace(0, 1, 100, endpoint=False)
    T = time_series.shape[0]
    b_T = 0.1 * T ** (-1/5)

    theta_hat = estimate_parameters_tvAR_p(time_series=time_series, p=p, u_list=u_list, kernel=Kernel(kernel_str), bandwidth=b_T, reflect_ts=reflect_time_series_bool)
    alpha_hat = theta_hat[:, :-1, :].squeeze(axis=2)
    sigma_hat = theta_hat[:, -1, :].squeeze()

    alpha_extrapolated, sigma_extrapolated = extrapolate_parameters(alpha_hat, sigma_hat, num_points=10, interpol_step=1, n_forecasts=1, k=k)
    x_star = forecast_future_values_tvAR_p(alpha_extrapolated, time_series)
    
    # Localized version of the z-score computed using the reflected time series
    right = estimate_local_mean(time_series.reshape(-1, 1), T, Kernel("one-sided epanechnikov (L)"), b_T)   
    new_time_series = reflect_time_series(time_series, right_point=right)
    localized_mean = estimate_local_mean(new_time_series.reshape(-1, 1), 2*T, Kernel("epanechnikov"), b_T)   
    localized_std = np.sqrt(estimate_local_autocovariance(new_time_series.reshape(-1, 1), 2*T, 0, Kernel("one-sided epanechnikov (L)"), b_T)) 
    # localized_mean = estimate_local_mean(time_series.reshape(-1, 1), T, Kernel("one-sided epanechnikov (L)"), b_T)   
    # localized_std = np.sqrt(estimate_local_autocovariance(time_series.reshape(-1, 1), T, 0, Kernel("one-sided epanechnikov (L)"), b_T))
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


def check_exit_trade(z: float, holding_position: str, threshold: float):
    """
    Determine the exit trade action. It is one of two possibilities:
    * 'CLOSE': close position
    * None: no signal found, hold the position

    --- parameters
    - z: value of the z-score
    - holding_position: 'SHORT', 'LONG' or None
    - threshold: threshold to exit a trade
    """
    assert holding_position in [None, 'SHORT', 'LONG']
    if (holding_position == 'SHORT' and z < -threshold) or (holding_position == 'LONG' and z > threshold):
        return 'CLOSE'
    else:
        return None


def get_actions_and_forecasts(time_series, threshold=1, n_forecasts=50, p=1, k=3, reflect_time_series=False):
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
        forecast, z_score = get_forecast_zscore(time_series_i, p, k, reflect_time_series_bool=reflect_time_series)
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
    Returns the ratio (number of time the forecast had the right direction) / (number of forecasts) everytime a trade signal is detected.
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
@dataclass
class TradingStrategy(ABC):
    """
    Generic class containing the logic of a trading strategy.
    
    --- parameters
    - data_df: dataframe with the data
    - T: length of each time series for fitting the model
    - p: Order of the tvAR(p) model
    - k: Order of the spline interpolation
    - entry_threshold: threshold to enter a trade
    - exit_threshold: threshold to close a position
    - filename: name of the csv file where the time series of the P&L is saved
    """

    data_df: pd.DataFrame
    T: int = 10_000
    p: int = 1
    k: int = 3
    entry_threshold: float = 1
    exit_threshold: float = 0.25
    filename: str = ""

    @abstractclassmethod
    def can_enter_trade(self, action: str, portfolio: Portfolio) -> bool:
        """Checks if all requirements are met to enter a new trade."""
        ...

    @abstractclassmethod
    def can_exit_trade(self, z_score: float, action: str, portfolio: Portfolio) -> bool:
        """Checks if all requirements are met to close current positions."""
        ...

    def simulate_trading(self, reflected_time_series=False, verbose=True) -> Portfolio:
        # Data & Spread: BTC-USD / ETH-USD
        assert self.data_df.shape[0] > self.T, f"The time series needs to have at least {self.T} rows."
        n_hours = self.data_df.shape[0] - self.T # number of hours of simulation
        start = self.data_df.index[self.T]
        end = self.data_df.index[-1]
        logging.info(f"Simulation launched from {start} to {end} ({n_hours} hours)")
        entry_actions = np.empty((n_hours,), dtype=object)

        # Initialize the portfolio: 0 BTC and 0 ETH
        portfolio = Portfolio(start, end)
        portfolio.history['true_value'] = self.data_df['spread_log_returns'].iloc[-n_hours:]
        portfolio.reflected_time_series = reflected_time_series

        for i in range(n_hours):
            date = self.data_df.index[self.T+i]
            time_series_i = self.data_df.loc[self.data_df.index[i : self.T+i], 'spread_log_returns'].to_numpy()
            btc_close = self.data_df.loc[self.data_df.index[self.T+i-1], 'btc_close']
            eth_close = self.data_df.loc[self.data_df.index[self.T+i-1], 'eth_close']

            # Update crypto value
            portfolio.update_crypto(btc_close, eth_close)
            # Get entry trade signal
            forecast, z_score = get_forecast_zscore(time_series_i, self.p, self.k, reflect_time_series_bool=reflected_time_series)
            action = check_entry_trade(z_score, self.entry_threshold)
            # Close previous positions
            if self.can_exit_trade(z_score, action, portfolio):
                portfolio.close_positions(date)
            # Open new position
            if self.can_enter_trade(action, portfolio):
                side_btc = 'BUY' if action == 'LONG' else 'SELL'
                side_eth = 'BUY' if action == 'SHORT' else 'SELL'
                portfolio.insert_order('BTC-USD', side=side_btc, price=btc_close, volume=1)
                portfolio.insert_order('ETH-USD', side=side_eth, price=eth_close, volume=1)
                entry_actions[i] = action
            # Update history 
            portfolio.update_history(date, action, forecast, z_score)
            # Verbose
            if verbose and (i+1) % 100 == 0:
                print(f"step {i+1} / {n_hours}")

        # Close the final positions at the end of the trading period
        portfolio.close_positions(date)

        # Compute hit ratio
        hit_ratio_ = hit_ratio(self.data_df['spread_log_returns'].to_numpy(), portfolio.history['forecast'].to_numpy(), entry_actions)
        portfolio.hit_ratio = hit_ratio_
        logging.info(f"Ratios: {hit_ratio_}")

        # Save results
        if self.filename:
            with open(f'./data/trading_simulations/{self.filename}.pickle', 'wb') as f:
                pickle.dump(portfolio, f, pickle.HIGHEST_PROTOCOL)

        return portfolio


class Strategy1(TradingStrategy):
    """Enter a position if abs(z_score) > entry_threshold and unwind it the next hour."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def can_enter_trade(self, action: str, portfolio: Portfolio) -> bool:
        """Enter a trade if a signal is detected and no positions are being held"""
        return (not portfolio.hold_positions) and (action is not None)

    def can_exit_trade(self, z_score: float, action: str, portfolio: Portfolio) -> bool:
        """Always close a position at time t+1."""
        return True


class Strategy2(TradingStrategy):
    """Enter a position if abs(z_score) > entry_threshold and unwind it when the opposite signal comes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def can_enter_trade(self, action, portfolio):
        """Enter a trade if a signal is detected and no positions are being held."""
        return (not portfolio.hold_positions) and (action is not None)

    def can_exit_trade(self, z_score, action, portfolio):
        """Close positions when the action is the opposite of the last action."""
        return (action is not None) and (action != portfolio.holding_position)


class Strategy3(TradingStrategy):
    """Enter a position if abs(z_score) > entry_threshold and unwind it when the spread returns close to the mean (or goes beyond in the other direction)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def can_enter_trade(self, action, portfolio):
        """Enter a trade if a signal is detected and no positions are being held."""
        return (not portfolio.hold_positions) and (action is not None)

    def can_exit_trade(self, z_score, action, portfolio):
        """Close positions when the the spread returns close to the mean (or goes beyond in the other direction)."""
        exit_action = check_exit_trade(z_score, portfolio.holding_position, self.exit_threshold)
        return exit_action == 'CLOSE'
