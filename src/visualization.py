from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_correlation_matrix(corr: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_monte_carlo(price_paths: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.plot(price_paths)
    plt.title("Monte Carlo Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


def plot_efficient_frontier(frontier: np.ndarray):
    plt.figure(figsize=(8, 6))
    plt.scatter(frontier[:, 0], frontier[:, 1], alpha=0.6)
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.title("Efficient Frontier")
    plt.tight_layout()
    plt.show()


def plot_var_distribution(daily_returns: pd.Series, var_value: float):
    plt.figure(figsize=(8, 6))
    sns.histplot(daily_returns, bins=50, kde=True)
    plt.axvline(-var_value, color="red", linestyle="--", label="VaR")
    plt.title("Return Distribution with VaR")
    plt.legend()
    plt.tight_layout()
    plt.show()
