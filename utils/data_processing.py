import pandas as pd
import numpy as np
import yfinance as yf

def fetch_data_from_yahoo(tickers, start_date, end_date, interval='1d'):
    """
    Fetch historical stock data from Yahoo Finance using yfinance.

    Parameters:
    - tickers (list): list of ticker symbols as strings (e.g., ['AAPL', 'GOOG'])
    - start_date (str or datetime)
    - end_date (str or datetime)
    
    """
    data = yf.download(tickers, start=start_date, end=end_date, interval=interval, group_by="ticker", auto_adjust=False)
    
    if len(tickers) == 1:
        data.columns = pd.MultiIndex.from_product([data.columns, [tickers[0]]])
    
    return data


def clean_close_data(raw_df):
    """
    Extract a DataFrame of close prices
    """

    price_levels = raw_df.columns.get_level_values("Price").unique()

    if "Close" in price_levels:
        
        close_df = raw_df.xs("Close", level="Price", axis=1)
    elif "Adj Close" in price_levels:
        # If there's no 'Close' but 'Adj Close' is present
        close_df = raw_df.xs("Adj Close", level="Price", axis=1)
    else:
        # Neither 'Close' nor 'Adj Close' is in the second-level columns
        raise ValueError(
            f"No 'Close' or 'Adj Close' found in the second-level columns: {list(price_levels)}"
        )

    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame()

    return close_df


def compute_log_returns(price_df):
    """
    Compute log returns of each column in the DataFrame.
    """
    # log return: ln(P_t / P_(t-1))
    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns.dropna()
