import numpy as np
import pandas as pd

import datetime as dt
import logging

import yfinance as yf

logging.basicConfig(level=logging.INFO)


def get_dates_str(num_hours:int, end:str = None) -> tuple[str, str]:
    """
    Returns two string dates separated by num_hours hours. Note, there will be **approximately** num_hours between the two dates.

    --- parameters
    - num_hours: number of hours between the two dates.
    - end: end date in format YYYY-MM-DD. If end is None, then it is today's date.
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
    # Download historical data
    data_df = yf.download(f"{symbol1} {symbol2}", **kwargs)['Close'] 
    data_df = data_df.dropna()
    # Keep raw prices in memory
    prices_df = data_df.copy()
    prices_df = prices_df.rename(columns={'BTC-USD': 'btc_close', 'ETH-USD': 'eth_close'})
    # Compute log returns
    data_df = np.log(1 + data_df.pct_change())
    data_df = data_df.rename(columns={'BTC-USD': 'btc_log_returns', 'ETH-USD': 'eth_log_returns'})
    # Concatenate raw prices and log returns an compute the spreads
    data_df = pd.concat([prices_df, data_df], axis=1)
    data_df = data_df.dropna()
    data_df['spread'] = data_df['btc_close'] - data_df['eth_close']
    data_df['spread_log'] = np.log(data_df['btc_close'] / data_df['eth_close'])
    data_df['spread_log_returns'] = data_df['btc_log_returns'] - data_df['eth_log_returns']
    # Format the index
    data_df.index = data_df.index.strftime("%Y-%m-%d %H:%M:%S")

    # logging info
    start_datetime = data_df.index[0]
    end_datetime = data_df.index[-1]
    logging.info(f"Data downloaded: from {start_datetime} to {end_datetime}")
    return data_df


#######################
# Reducing edge effects
#######################
def reflect_time_series(time_series: np.ndarray, left_point: tuple[float] = None, right_point: tuple[float] = None) -> np.ndarray:
    new_time_series = time_series.copy()
    if right_point is not None:
        time_series_right = np.flip(2 * right_point[1] - time_series)
        new_time_series = np.concatenate([new_time_series, time_series_right])
    if left_point is not None:
        time_series_left = np.flip(2 * left_point[1] - time_series)
        new_time_series = np.concatenate([time_series_left, new_time_series])
    return new_time_series
    