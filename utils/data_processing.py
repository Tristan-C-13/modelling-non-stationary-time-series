import numpy as np
import pandas as pd

import datetime as dt
import logging
import os

import yfinance as yf

logging.basicConfig(level=logging.INFO)


def get_dates_str(num_hours: int, end: str = None) -> tuple[str, str]:
    """
    Returns two string dates separated by num_hours hours. Note, there will be **approximately** num_hours between the two dates.

    --- parameters
    - num_hours: number of hours between the two dates.
    - end: end date in format YYYY-MM-DD. If end is None, then it is today's date.
    """
    if end is None:
        end = dt.datetime.today()
    else:
        end = dt.datetime.strptime(end, "%Y-%m-%d")
    start = end - dt.timedelta(hours=num_hours)
    return start.strftime("%Y-%m-%d %H:%M"), end.strftime("%Y-%m-%d %H:%M")


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


def load_spread_btc_eth(start: str = None, end: str = None) -> pd.DataFrame:
    """
    Returns a pd.DataFrame containing close prices of ETH/USDT and BTC/USDT as well as their spread (raw spread + spread of log returns)

    --- parameters
    - start (str): start date in format YYYY-MM-DD
    - end (str): end date in format YYYY-MM-DD
    """
    # Load csv if it already exists:
    if os.path.exists('data/data-binance.csv'):
        data_df = pd.read_csv('data/data-binance.csv', parse_dates=True, dayfirst=False, index_col='date')
    else:
        # Load historical data
        btc_df = pd.read_csv('data/Binance_BTCUSDT_1h.csv', sep=';', parse_dates=True,
                             index_col='date', usecols=['date', 'close'], dayfirst=True)
        btc_df = btc_df.rename(columns={'close': 'btc_close'})
        btc_df = btc_df.sort_index(ascending=True) # Reverse to have recent values at the end
        eth_df = pd.read_csv('data/Binance_ETHUSDT_1h.csv', sep=';', parse_dates=True,
                             index_col='date', usecols=['date', 'close'], dayfirst=True)
        eth_df = eth_df.rename(columns={'close': 'eth_close'})
        eth_df = eth_df.sort_index(ascending=True)
        
        # Compute log returns
        btc_df['btc_log_returns'] = np.log(1 + btc_df['btc_close'].pct_change())
        eth_df['eth_log_returns'] = np.log(1 + eth_df['eth_close'].pct_change())

        # Compute spreads
        data_df = pd.concat([btc_df, eth_df], axis=1)
        data_df = data_df.dropna()
        data_df = data_df.reindex(pd.date_range(data_df.index.min(), data_df.index.max(), freq='1H'), method='ffill')
        data_df['spread'] = data_df['btc_close'] - data_df['eth_close']
        data_df['spread_log_returns'] = data_df['btc_log_returns'] - data_df['eth_log_returns']

        # Save
        data_df.to_csv("data/data-binance.csv", index=True, index_label='date')
        logging.info("Data successfully saved.")

    # Slice dataframe
    if start is not None:
        data_df = data_df.loc[start:, :]
    if end is not None:
        data_df = data_df.loc[:end, :]

    # Logging info
    start_datetime = data_df.index[0]
    end_datetime = data_df.index[-1]
    logging.info(f"Data loaded from {start_datetime} to {end_datetime}")

    return data_df


#######################
# Reducing edge effects
#######################
def reflect_time_series(time_series: np.ndarray, left_point: float = None, right_point: float = None) -> np.ndarray:
    """
    Reflects a time series according to the geometrical method described in Peter Hall (1991).
    """
    new_time_series = time_series.copy()
    if right_point is not None:
        time_series_right = np.flip(2 * right_point - time_series)
        new_time_series = np.concatenate([new_time_series, time_series_right])
    if left_point is not None:
        time_series_left = np.flip(2 * left_point - time_series)
        new_time_series = np.concatenate([time_series_left, new_time_series])
    return new_time_series
    
def convert_u_list(u_list: np.ndarray) -> np.ndarray:
    """
    Converts u_list which is a partition between 0 and 1 to a partition between 1/3 and 2/3.
    This function is used after  reflecting the time series.
    """
    return (u_list + 1) / 3 