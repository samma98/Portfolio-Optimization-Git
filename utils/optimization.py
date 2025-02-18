import numpy as np
import pandas as pd

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

def simulate_random_portfolios(price_df, 
                               risk_free_rate=0.03, 
                               simulations=2000, 
                               trading_days=252):
    """
    (Optional) Generate random portfolios if you want a "cloud" of points.

    """
    daily_log_returns = (price_df / price_df.shift(1)).apply(np.log).dropna()
    mu = daily_log_returns.mean() * trading_days
    cov = daily_log_returns.cov() * trading_days
    print("Price DF columns:", price_df.columns)
    print("Shape:", price_df.shape)

    n_assets = len(price_df.columns)
    results = []

    for _ in range(simulations):
        w = np.random.random(n_assets)
        w /= np.sum(w)  # normalize to sum=1

        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        sharpe = (port_ret - risk_free_rate) / port_vol

        results.append((port_ret, port_vol, sharpe, w))

    df = pd.DataFrame(results, columns=["Returns", "Volatility", "Sharpe_ratio", "Weights"])
    return df


def compute_efficient_frontier(price_df, points=1000, risk_free_rate=0.03):
    """
    Use PyPortfolioOpt to compute a Markowitz efficient frontier by
    iterating over possible target returns. 

    """
    daily_log_returns = (price_df / price_df.shift(1)).apply(np.log).dropna()
    mu = daily_log_returns.mean() * 252
    S = daily_log_returns.cov() * 252
    print("Covariance shape1:", S.shape)

    min_r = mu.min()
    max_r = mu.max()

    rets = np.linspace(min_r, max_r, points)
    frontier_data = []
    for target_return in rets:
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 1.0))
        try:
            
            ef.efficient_return(target_return, market_neutral=False)
            ret, vol, _ = ef.portfolio_performance(verbose=False)

        
            w_dict = ef.clean_weights()
            # e.g., {"AAPL": 0.25, "GOOG": 0.35, "MSFT": 0.40, ...}
            print("Computed Weights:", w_dict)

            frontier_data.append((ret, vol, w_dict))
        except:
            # if no feasible solution
            pass

    frontier_df = pd.DataFrame(frontier_data, columns=["Returns","Volatility","Weights"])
    return frontier_df


def compute_max_sharpe_and_min_vol(price_df, risk_free_rate=0.03):
    """
    Compute the single portfolio with max Sharpe ratio
    and the single portfolio with min volatility, using PyPortfolioOpt.
    
    """
    daily_log_returns = (price_df / price_df.shift(1)).apply(np.log).dropna()
    mu = daily_log_returns.mean() * 252
    S = daily_log_returns.cov() * 252
    print("Covariance shape2:", S.shape)

    ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 1.0))
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    ret_ms, vol_ms, _ = ef.portfolio_performance()

    ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 1.0))
    ef.min_volatility()
    ret_mv, vol_mv, _ = ef.portfolio_performance()

    return (ret_ms, vol_ms), (ret_mv, vol_mv)
