import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_processing import get_dates_str, load_spread_btc_eth
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
    strat1 = Strategy1(data_df, T, p, k, 1, 'strat1-1000')
    portfolio_1 = strat1.simulate_trading()

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

        
    plt.show()