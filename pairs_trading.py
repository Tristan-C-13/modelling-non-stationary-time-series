import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import pickle
import logging

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
    data_df = load_spread_btc_eth(end=end)


    # TUNING ENTRY THRESHOLD
    logging.info("TUNING")
    data_df_tune = data_df.loc['2019-01-13':start, :]
    z_entry_list = np.linspace(1, 2.5, 15) # grid

    ratio_df = pd.DataFrame(index=z_entry_list, columns=['strat1', 'strat2', 'strat3'], dtype=float)
    for z_entry in z_entry_list:
        p1 = Strategy1(data_df_tune, T, p, k, z_entry, filename=f"strat1_n=5000_z={z_entry}-tune").simulate_trading(verbose=True)
        print(f"1) {z_entry}: {p1.hit_ratio['ratio']}")
        p2 = Strategy2(data_df_tune, T, p, k, z_entry, filename=f"strat2_n=5000_z={z_entry}-tune").simulate_trading(verbose=True)
        print(f"2) {z_entry}: {p2.hit_ratio['ratio']}")
        p3 = Strategy3(data_df_tune, T, p, k, z_entry, 0.25, filename=f"strat3_n=5000_z={z_entry}-tune").simulate_trading(verbose=True)
        print(f"3) {z_entry}: {p3.hit_ratio['ratio']}")

        ratio_df.loc[z_entry, :] = [p1.hit_ratio['ratio'], p2.hit_ratio['ratio'], p3.hit_ratio['ratio']]
    
    z_entry_star_series = ratio_df.idxmax(axis=0, skipna=True)
    ratio_df.to_csv('ratio_df.csv')


    # TRADING SIMULATION
    logging.info("SIMULATIONS")
    data_df_simulation = data_df.loc[start:, :]

    z_entry_star1 = z_entry_star_series.at['strat1']   # 1.321429
    strat1 = Strategy1(data_df_simulation, T, p, k, z_entry_star1, filename=f"strat1_n=5000-final-25000dollars2")
    portfolio_1 = strat1.simulate_trading(reflected_time_series=False)
    print(portfolio_1.hit_ratio)

    z_entry_star2 = z_entry_star_series.at['strat2']   # 1.428571
    strat2 = Strategy2(data_df_simulation, T, p, k, z_entry_star2, filename=f"strat2_n=5000-final-25000dollars2")
    portfolio_2 = strat2.simulate_trading(reflected_time_series=False)
    print(portfolio_2.hit_ratio)

    z_entry_star3 = z_entry_star_series.at['strat3']   # 1.214286	
    strat3 = Strategy3(data_df_simulation, T, p, k, z_entry_star3, 0.75, filename=f"strat3_n=5000-final-25000dollars2")
    portfolio_3 = strat3.simulate_trading(reflected_time_series=False)
    print(portfolio_3.hit_ratio)
    

    # PLOT P&L
    fig, axs = plt.subplots(3, 1)
    portfolio_1.plot_pnl_entries(axs[0])
    portfolio_2.plot_pnl_entries(axs[1])
    portfolio_3.plot_pnl_entries(axs[2])
    plt.show()