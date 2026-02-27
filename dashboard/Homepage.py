import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import DEFAULT_CONFIG
from src.data_loader import fetch_price_data
from src.returns import compute_log_returns, rolling_annualized_volatility
from src.portfolio import portfolio_daily_returns
from src.risk_metrics import (
    var_parametric_normal,
    var_historical,
    cvar_historical,
    rolling_var_parametric_normal,
    rolling_var_historical,
    var_backtest,
    distribution_diagnostics,
    normal_pdf_overlay,
)

def save_results_to_session(**kwargs):
    st.session_state["risk_app_results"] = kwargs

st.set_page_config(page_title="Stock Risk Analyzer", layout="wide")
st.title("Stock Risk Analyzer Dashboard")

# ---- Inputs ----
tickers_str = st.text_input("Enter tickers (comma-separated):", "AAPL,MSFT,AMZN")
confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)

portfolio_value = st.number_input(
    "Portfolio Value ($)",
    min_value=1.0,
    value=100000.0,
    step=1000.0,
)

rolling_window = st.slider(
    "Rolling window (days) for time-varying risk",
    min_value=20,
    max_value=252,
    value=60,
    step=5,
)

ticker_list = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
if len(ticker_list) == 0:
    st.warning("Please enter at least one ticker (e.g., AAPL,MSFT,AMZN).")
    st.stop()

# ---- Data ----
try:
    prices = fetch_price_data(
        tickers=ticker_list,
        start_date=DEFAULT_CONFIG.start_date,
        end_date=DEFAULT_CONFIG.end_date,
        price_field=DEFAULT_CONFIG.price_field,
        cache_dir=DEFAULT_CONFIG.data_raw_dir,
        use_cache=True,
    ).prices
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

if prices.empty:
    st.error("No price data returned. Check tickers and date range.")
    st.stop()
# After fetching prices
prices = prices.sort_index()

# Option A (strict): keep only dates where ALL tickers have prices
prices = prices.dropna(how="any")

# (Optional) If you want to be less strict:
# prices = prices.ffill().dropna(how="any")

log_returns = compute_log_returns(prices)
log_returns = log_returns.dropna(how="any")

# ---- Weights (user-adjustable, auto-normalized) ----
st.subheader("Portfolio Weights")

default_w = np.ones(len(ticker_list)) / len(ticker_list)
weights_key = f"weights::{','.join(ticker_list)}"

weights = []
cols = st.columns(min(len(ticker_list), 4))

for i, ticker in enumerate(ticker_list):
    with cols[i % len(cols)]:
        w = st.slider(
            f"{ticker}",
            min_value=0.0,
            max_value=1.0,
            value=float(default_w[i]),
            step=0.01,
            key=f"{weights_key}::{ticker}",
        )
        weights.append(w)

weights = np.array(weights, dtype=float)

if weights.sum() == 0:
    st.error("Total weight cannot be 0. Increase at least one ticker weight.")
    st.stop()

# Normalize to sum to 1
weights = weights / weights.sum()

st.caption("Normalized weights (auto-scaled to sum to 1):")
w_cols = st.columns(min(len(ticker_list), 4))
for i, (ticker, w) in enumerate(zip(ticker_list, weights)):
    with w_cols[i % len(w_cols)]:
        st.write(f"**{ticker}:** {w:.2%}")

portfolio_returns = portfolio_daily_returns(log_returns, weights)

# ---- Point-in-time risk metrics ----
var_norm_dollars = var_parametric_normal(portfolio_returns, confidence, portfolio_value)
var_hist_dollars = var_historical(portfolio_returns, confidence, portfolio_value)
cvar_hist_dollars = cvar_historical(portfolio_returns, confidence, portfolio_value)

col1, col2, col3 = st.columns(3)
col1.metric(f"1-Day VaR (Normal) @ {int(confidence*100)}%", f"${var_norm_dollars:,.2f}")
col2.metric(f"1-Day VaR (Historical) @ {int(confidence*100)}%", f"${var_hist_dollars:,.2f}")
col3.metric(f"1-Day CVaR (Historical) @ {int(confidence*100)}%", f"${cvar_hist_dollars:,.2f}")

st.caption(
    "VaR is a threshold loss (e.g., 95% VaR means ~5% chance of losing more). "
    "CVaR/Expected Shortfall is the average loss in the worst 5% of days."
)

# ---- Rolling risk ----
st.subheader("Time-Varying Risk (Rolling Metrics)")

roll_vol = rolling_annualized_volatility(
    portfolio_returns,
    window=rolling_window,
    trading_days=252
)

roll_var_norm = rolling_var_parametric_normal(
    portfolio_returns,
    confidence,
    window=rolling_window
)
roll_var_hist = rolling_var_historical(
    portfolio_returns,
    confidence,
    window=rolling_window
)

# Convert rolling VaR from return-units to dollars
roll_var_norm_dollars = roll_var_norm * portfolio_value
roll_var_hist_dollars = roll_var_hist * portfolio_value

rolling_df = pd.DataFrame(
    {
        "Rolling Annualized Vol": roll_vol,
        "Rolling VaR Normal ($)": roll_var_norm_dollars,
        "Rolling VaR Historical ($)": roll_var_hist_dollars,
    }
)

st.line_chart(rolling_df)

# ---- Backtesting ----
st.subheader("VaR Backtesting")

bt_norm = var_backtest(portfolio_returns, roll_var_norm, confidence)
bt_hist = var_backtest(portfolio_returns, roll_var_hist, confidence)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Obs (after window)", f"{bt_norm['n_obs']}")
c2.metric("Expected violation rate", f"{bt_norm['expected_violation_rate']:.2%}")
c3.metric("Observed (Normal VaR)", f"{bt_norm['observed_violation_rate']:.2%}")
c4.metric("Observed (Hist VaR)", f"{bt_hist['observed_violation_rate']:.2%}")

st.write(
    f"- Normal VaR violations: **{bt_norm['n_violations']} / {bt_norm['n_obs']}** "
    f"(diff vs expected: **{bt_norm['difference']:+.2%}**)\n"
    f"- Historical VaR violations: **{bt_hist['n_violations']} / {bt_hist['n_obs']}** "
    f"(diff vs expected: **{bt_hist['difference']:+.2%}**)"
)

# ---- Distribution diagnostics ----
st.subheader("Return Distribution Diagnostics")

diag = distribution_diagnostics(portfolio_returns)
st.write(
    f"- Mean: **{diag['mean']:.6f}**\n"
    f"- Std dev: **{diag['std']:.6f}**\n"
    f"- Skewness: **{diag['skew']:.3f}** (negative = heavier left tail)\n"
    f"- Excess kurtosis: **{diag['excess_kurtosis']:.3f}** (0 ≈ normal; >0 = fat tails)"
)

save_results_to_session(
    tickers=ticker_list,
    weights=weights,
    confidence=confidence,
    portfolio_value=portfolio_value,
    rolling_window=rolling_window,

    # data needed for insight generation
    portfolio_returns=portfolio_returns,
    log_returns_assets=log_returns,  # <-- NOTE: change this to your asset log returns df
    var_norm_dollars=var_norm_dollars,
    var_hist_dollars=var_hist_dollars,
    cvar_hist_dollars=cvar_hist_dollars,
    bt_norm=bt_norm,
    bt_hist=bt_hist,
    diag=diag,
)
# ---- Charts: prices + histogram + normal overlay ----
left, right = st.columns([2, 1])

with left:
    st.subheader("Adjusted Close Prices")
    st.line_chart(prices)

with right:
    st.subheader("Portfolio Daily Log Returns (Histogram + Normal Overlay)")

    r = portfolio_returns.dropna().values
    mu, sd = diag["mean"], diag["std"]

    fig, ax = plt.subplots()
    ax.hist(r, bins=50, density=True)  # density=True so pdf overlay matches scale

    x = np.linspace(r.min(), r.max(), 300)
    y = normal_pdf_overlay(x, mu, sd)
    ax.plot(x, y)  # overlay normal pdf

    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Density")
    st.pyplot(fig, clear_figure=True)