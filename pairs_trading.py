import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_processing import get_dates_str, load_spread_btc_eth, reflect_time_series
from utils.trading import Strategy1, Strategy2, Strategy3
from utils.graphics import plot_rolling_entries, plot_rolling_forecasts, make_general_plots

sns.set_style("whitegrid")



if __name__ == '__main__':
    # Parameters
    p = 1       # order of the tvAR(p)
    k = 3       # order of the spline interpolation
    T = 10_000  # length of each time series

    # DATA & SPREAD: BTC-USD / ETH-USD
    n_simulations = 5000
    start, end = get_dates_str(n_simulations + T - 1, '2022-08-14')
    data_df = load_spread_btc_eth(start, end)
    time_series = data_df['spread_log_returns'].to_numpy()
    spread_time_series = data_df['spread'].to_numpy()
    
    # TRADING SIMULATION
    # strat3 = Strategy3(data_df, T, p, k, 1, 0.25, filename="")
    # portfolio_3 = strat3.simulate_trading(reflected_time_series=False)
    # print(portfolio_3.hit_ratio)


    # TUNING ENTRY THRESHOLD
    z_entry_list = np.linspace(1, 3, 20)
    for z_entry in z_entry_list:
        p1 = Strategy1(data_df, T, p, k, z_entry, filename=f"strat1_n=5000_z={z_entry}").simulate_trading(verbose=False)
        print(f"1) {z_entry}: {p1.hit_ratio['ratio']}")
        p2 = Strategy2(data_df, T, p, k, z_entry, filename=f"strat2_n=5000_z={z_entry}").simulate_trading(verbose=False)
        print(f"2) {z_entry}: {p2.hit_ratio['ratio']}")
        p3 = Strategy3(data_df, T, p, k, z_entry, 0.25, filename=f"strat3_n=5000_z={z_entry}").simulate_trading(verbose=False)
        print(f"3) {z_entry}: {p3.hit_ratio['ratio']}")

    # plt.show()