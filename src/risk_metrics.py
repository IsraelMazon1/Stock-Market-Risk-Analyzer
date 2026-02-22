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
    Parametric (Normal) 1-day VaR in currency units (positive loss threshold).
    VaR = - (mu + z * sigma) * V, where z is lower-tail quantile (e.g. 5% for 95% VaR).
    """
    if daily_returns is None or daily_returns.empty:
        raise ValueError("daily_returns is empty.")
    mu = float(daily_returns.mean())
    sigma = float(daily_returns.std(ddof=1))
    if sigma <= 0:
        raise ValueError("Return volatility is zero; VaR not meaningful.")

    alpha = 1.0 - confidence_level
    z = float(norm.ppf(alpha))  # negative
    var_return = -(mu + z * sigma)  # positive loss threshold in return terms
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
    return float((-tail.mean()) * portfolio_value)


# ----------------------------
# Essential upgrades for DS:
# Rolling VaR + Backtesting + Diagnostics
# ----------------------------

def rolling_var_parametric_normal(
    daily_returns: pd.Series,
    confidence_level: float = 0.95,
    window: int = 60,
) -> pd.Series:
    """
    Rolling parametric VaR (return units, positive numbers).

    For each day t, fit Normal(mu_t, sigma_t) from the previous `window` returns,
    then compute VaR_t = -(mu_t + z * sigma_t).

    Returns Series aligned to input index (NaN for first `window-1` points).
    """
    if daily_returns is None or daily_returns.empty:
        raise ValueError("daily_returns is empty.")
    if window < 10:
        raise ValueError("window should be >= 10 for stable estimates.")

    alpha = 1.0 - confidence_level
    z = float(norm.ppf(alpha))  # negative

    mu = daily_returns.rolling(window).mean()
    sigma = daily_returns.rolling(window).std(ddof=1)

    var_return = -(mu + z * sigma)  # positive loss threshold in return terms
    return var_return


def rolling_var_historical(
    daily_returns: pd.Series,
    confidence_level: float = 0.95,
    window: int = 60,
) -> pd.Series:
    """
    Rolling historical VaR (return units, positive numbers).

    VaR_t = - quantile_alpha(returns_{t-window:t})
    """
    if daily_returns is None or daily_returns.empty:
        raise ValueError("daily_returns is empty.")
    if window < 10:
        raise ValueError("window should be >= 10 for stable estimates.")

    alpha = 1.0 - confidence_level
    q = daily_returns.rolling(window).quantile(alpha)  # negative in losses
    return -q  # positive


def var_backtest(
    daily_returns: pd.Series,
    var_series_return_units: pd.Series,
    confidence_level: float = 0.95,
) -> dict:
    """
    Backtest VaR: compare realized losses vs VaR threshold.

    Convention here:
      - daily_returns: daily portfolio log returns
      - var_series_return_units: positive VaR values in return units (e.g. 0.028)
      - Violation occurs if return < -VaR (i.e. loss exceeds VaR threshold)

    Returns summary stats: observed violation rate vs expected alpha.
    """
    if daily_returns is None or daily_returns.empty:
        raise ValueError("daily_returns is empty.")
    if var_series_return_units is None or var_series_return_units.empty:
        raise ValueError("var_series_return_units is empty.")

    aligned = pd.concat([daily_returns, var_series_return_units], axis=1).dropna()
    aligned.columns = ["ret", "var"]

    # violation if return is less than negative VaR threshold
    violations = aligned["ret"] < -aligned["var"]

    n = int(len(violations))
    x = int(violations.sum())
    observed = x / n if n else float("nan")

    alpha = 1.0 - confidence_level  # expected violation probability

    return {
        "n_obs": n,
        "n_violations": x,
        "expected_violation_rate": alpha,
        "observed_violation_rate": observed,
        "difference": observed - alpha,
    }


def distribution_diagnostics(daily_returns: pd.Series) -> dict:
    """
    Returns skewness and (excess) kurtosis for diagnostics.
    pandas Series.kurt() returns excess kurtosis by default (0 = normal).
    """
    if daily_returns is None or daily_returns.empty:
        raise ValueError("daily_returns is empty.")
    return {
        "mean": float(daily_returns.mean()),
        "std": float(daily_returns.std(ddof=1)),
        "skew": float(daily_returns.skew()),
        "excess_kurtosis": float(daily_returns.kurt()),
    }


def normal_pdf_overlay(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Utility for plotting: normal pdf values at x for overlay on histogram.
    """
    if std <= 0:
        raise ValueError("std must be > 0")
    return norm.pdf(x, loc=mean, scale=std)