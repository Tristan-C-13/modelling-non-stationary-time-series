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
    n_simulations = 1000
    start, end = get_dates_str(n_simulations + T - 1, '2022-08-14')
    data_df = load_spread_btc_eth(start, end)
    time_series = data_df['spread_log_returns'].to_numpy()
    spread_time_series = data_df['spread'].to_numpy()

    # PLOTS & ANALYSES  
    # make_general_plots(time_series, p=p, k=k)
    # plot_rolling_forecasts(time_series, p=p, k=k, n_forecasts=200, n_last_points=220)
    # plot_rolling_entries(time_series, spread_time_series, p=p, k=k, n_forecasts=200, n_last_points=220)

    
    # TRADING SIMULATION
    # strat3 = Strategy3(data_df, T, p, k, 1, 0.25, filename="")
    # portfolio_3 = strat3.simulate_trading(reflected_time_series=False)
    # print(portfolio_3.hit_ratio)

    strat3 = Strategy3(data_df, T, p, k, 1, 0.25, filename="")
    portfolio_3 = strat3.simulate_trading(reflected_time_series=True)
    print(portfolio_3.hit_ratio)

    # with (
    #     open('data/trading_simulations/strat1-1000.pickle', 'rb') as f1,
    #     open('data/trading_simulations/strat2-1000.pickle', 'rb') as f2,
    #     open('data/trading_simulations/strat3-1000.pickle', 'rb') as f3,
    # ):
    #     portfolio_1 = pickle.load(f1)
    #     portfolio_2 = pickle.load(f2)
    #     portfolio_3 = pickle.load(f3)

    # fig, axs = plt.subplots(3, 1)
    # portfolio_1.plot_pnl_entries(axs[0])
    # portfolio_2.plot_pnl_entries(axs[1])
    # portfolio_3.plot_pnl_entries(axs[2])

    # fig, axs = plt.subplots(3, 3)
    # for i in range(1, 4):
    #     for z in range(3):
    #         with open(f'data/trading_simulations/strat{i}_n=5000_z{z}.pickle', 'rb') as f:
    #             portfolio = pickle.load(f)
    #         portfolio.plot_pnl_entries(axs[i-1, z], z != 0)
    #         print(f"Strategy {i}, {z=}, hit_ratio: {portfolio.hit_ratio}")
    #         axs[i-1, z].set_title(f"Strategy {i}, {z=}")


    # TUNING ENTRY THRESHOLD
    # z_entry_list = np.linspace(1, 2, 5)
    # for z_entry in z_entry_list:
    #     p1 = Strategy1(data_df, T, p, k, z_entry, filename="").simulate_trading(verbose=False)
    #     print(f"1) {z_entry}: {p1.hit_ratio['ratio']}")
    #     p2 = Strategy2(data_df, T, p, k, z_entry, filename="").simulate_trading(verbose=False)
    #     print(f"2) {z_entry}: {p2.hit_ratio['ratio']}")
    #     p3 = Strategy3(data_df, T, p, k, z_entry, 0.25, filename="").simulate_trading(verbose=False)
    #     print(f"3) {z_entry}: {p3.hit_ratio['ratio']}")

    plt.show()