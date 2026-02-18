from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def sharpe_ratio(
    annual_return: float,
    annual_volatility: float,
    risk_free_rate_annual: float = 0.0,
) -> float:
    """
    Sharpe = (Rp - Rf) / sigma_p
    """
    if annual_volatility <= 0:
        raise ValueError("annual_volatility must be > 0")
    return float((annual_return - risk_free_rate_annual) / annual_volatility)


def var_parametric_normal(
    daily_returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """
    Parametric (Normal) 1-day VaR in *currency units* (positive number = loss threshold).
    VaR = - (mu + z * sigma) * V, where z is the lower-tail quantile (e.g. 5% for 95% VaR).
    """
    if daily_returns is None or daily_returns.empty:
        raise ValueError("daily_returns is empty.")
    mu = float(daily_returns.mean())
    sigma = float(daily_returns.std(ddof=1))
    if sigma <= 0:
        raise ValueError("Return volatility is zero; VaR not meaningful.")

    alpha = 1.0 - confidence_level
    z = float(norm.ppf(alpha))  # negative
    var_return = -(mu + z * sigma)  # positive loss threshold
    return float(var_return * portfolio_value)


def var_historical(
    daily_returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """
    Historical 1-day VaR using empirical quantile.
    VaR = - quantile(alpha) * V
    """
    if daily_returns is None or daily_returns.empty:
        raise ValueError("daily_returns is empty.")
    alpha = 1.0 - confidence_level
    q = float(daily_returns.quantile(alpha))  # typically negative
    return float(-q * portfolio_value)


def cvar_historical(
    daily_returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """
    Historical CVaR (Expected Shortfall): average loss beyond VaR threshold.
    """
    if daily_returns is None or daily_returns.empty:
        raise ValueError("daily_returns is empty.")
    alpha = 1.0 - confidence_level
    threshold = float(daily_returns.quantile(alpha))
    tail = daily_returns[daily_returns <= threshold]
    if tail.empty:
        raise RuntimeError("No tail observations found for CVaR.")
    # loss is negative returns -> positive loss
    return float((-tail.mean()) * portfolio_value)
