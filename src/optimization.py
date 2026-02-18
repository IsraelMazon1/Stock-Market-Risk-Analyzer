from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def minimize_volatility(cov_matrix: pd.DataFrame):
    """
    Computes minimum variance portfolio weights.
    """

    n = len(cov_matrix)
    init_guess = np.ones(n) / n
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def portfolio_variance(w):
        return w.T @ cov_matrix.values @ w

    result = minimize(
        portfolio_variance,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError("Optimization failed.")

    return result.x


def efficient_frontier(
    annual_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    points: int = 50,
):
    """
    Generates efficient frontier curve.
    """

    results = []
    n = len(annual_returns)

    for _ in range(points):
        weights = np.random.random(n)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, annual_returns.values)
        portfolio_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)

        results.append((portfolio_vol, portfolio_return))

    return np.array(results)
