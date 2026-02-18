from __future__ import annotations
import numpy as np


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float = 1.0,
    steps: int = 252,
    simulations: int = 1000,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Simulates price paths using Geometric Brownian Motion.

    Returns:
        np.ndarray shape (steps + 1, simulations)
    """

    if S0 <= 0:
        raise ValueError("Initial price S0 must be positive.")
    if sigma < 0:
        raise ValueError("Volatility must be non-negative.")

    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / steps
    price_paths = np.zeros((steps + 1, simulations))
    price_paths[0] = S0

    for t in range(1, steps + 1):
        z = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return price_paths
