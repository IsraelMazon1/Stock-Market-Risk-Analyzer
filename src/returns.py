from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes log returns: r_t = ln(P_t / P_{t-1})
    """
    if prices is None or prices.empty:
        raise ValueError("prices is empty.")
    prices = prices.astype(float)
    rets = np.log(prices / prices.shift(1))
    rets = rets.dropna(how="all")
    return rets


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}
    """
    if prices is None or prices.empty:
        raise ValueError("prices is empty.")
    prices = prices.astype(float)
    rets = prices.pct_change().dropna(how="all")
    return rets


def annualized_return(log_returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Annualized mean of daily log returns.
    """
    if log_returns is None or log_returns.empty:
        raise ValueError("log_returns is empty.")
    return log_returns.mean() * trading_days


def annualized_volatility(log_returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Annualized volatility of daily log returns.
    """
    if log_returns is None or log_returns.empty:
        raise ValueError("log_returns is empty.")
    return log_returns.std(ddof=1) * np.sqrt(trading_days)


def covariance_matrix(log_returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Annualized covariance matrix of log returns.
    """
    if log_returns is None or log_returns.empty:
        raise ValueError("log_returns is empty.")
    return log_returns.cov() * trading_days
