from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    s = weights.sum()
    if s == 0:
        raise ValueError("Weights sum to zero.")
    return weights / s


def portfolio_expected_return(annual_returns: pd.Series, weights: np.ndarray) -> float:
    """
    annual_returns: Series indexed by ticker
    weights: array aligned to annual_returns index order
    """
    w = normalize_weights(weights)
    mu = annual_returns.values.astype(float)
    if mu.shape[0] != w.shape[0]:
        raise ValueError("weights length must match number of assets.")
    return float(np.dot(mu, w))


def portfolio_volatility(cov_matrix_annual: pd.DataFrame, weights: np.ndarray) -> float:
    """
    Computes sqrt(w^T * Sigma * w)
    """
    w = normalize_weights(weights)
    Sigma = cov_matrix_annual.values.astype(float)
    if Sigma.shape[0] != w.shape[0] or Sigma.shape[1] != w.shape[0]:
        raise ValueError("weights length must match covariance matrix size.")
    var = float(w.T @ Sigma @ w)
    return float(np.sqrt(var))


def portfolio_daily_returns(log_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Returns a Series of portfolio daily log returns (weighted sum across assets).
    """
    w = normalize_weights(weights)
    if log_returns.shape[1] != w.shape[0]:
        raise ValueError("weights length must match number of assets.")

    # weighted sum of columns
    portfolio_returns = log_returns @ w
    portfolio_returns.name = "portfolio_log_return"
    return portfolio_returns

