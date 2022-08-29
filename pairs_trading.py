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
    # data_df = load_spread_btc_eth(start, end)
    data_df = load_spread_btc_eth(end=end)


    # TUNING ENTRY THRESHOLD
    data_df_tune = data_df.loc[:start, :]
    z_entry_list = np.linspace(1, 3, 20) # grid

    ratio_df = pd.DataFrame(index=z_entry_list, columns=['strat1', 'strat2', 'strat3'], dtype=float)
    for z_entry in z_entry_list:
        p1 = Strategy1(data_df_tune, T, p, k, z_entry, filename=f"strat1_n=5000_z={z_entry}").simulate_trading(verbose=True)
        print(f"1) {z_entry}: {p1.hit_ratio['ratio']}")
        p2 = Strategy2(data_df_tune, T, p, k, z_entry, filename=f"strat2_n=5000_z={z_entry}").simulate_trading(verbose=True)
        print(f"2) {z_entry}: {p2.hit_ratio['ratio']}")
        p3 = Strategy3(data_df_tune, T, p, k, z_entry, 0.25, filename=f"strat3_n=5000_z={z_entry}").simulate_trading(verbose=True)
        print(f"3) {z_entry}: {p3.hit_ratio['ratio']}")

        ratio_df.loc[z_entry, :] = [p1.hit_ratio['ratio'], p2.hit_ratio['ratio'], p3.hit_ratio['ratio']]
    
    z_entry_star_series = ratio_df.idxmax(axis=0, skipna=True)
    print(z_entry_star_series)


    # TRADING SIMULATION
    data_df_simulation = data_df[start:, :]

    z_entry_star3 = z_entry_star_series.at['strat3']
    # strat3 = Strategy3(data_df_simulation, T, p, k, z_entry_star3, 0.25, filename=f"strat3_n=5000_z={z_entry_star3}-final")
    # portfolio_3 = strat3.simulate_trading(reflected_time_series=False)
    # print(portfolio_3.hit_ratio)
    # plt.show()