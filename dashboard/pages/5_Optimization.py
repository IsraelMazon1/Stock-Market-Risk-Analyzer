from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.optimization import minimize_volatility, efficient_frontier
from src.returns import annualized_return, covariance_matrix
from src.portfolio import portfolio_expected_return, portfolio_volatility, portfolio_daily_returns
from src.risk_metrics import var_historical, cvar_historical

st.set_page_config(page_title="Portfolio Optimization", layout="wide")
st.title("Portfolio Optimization")

# ---- Guard: require session state ----
if "risk_app_results" not in st.session_state:
    st.warning("No portfolio data found. Please configure your portfolio on the **Homepage** first.")
    st.stop()

bundle = st.session_state["risk_app_results"]
tickers = bundle["tickers"]
weights = bundle["weights"]
confidence = bundle["confidence"]
portfolio_value = bundle["portfolio_value"]
log_returns_assets = bundle["log_returns_assets"]
var_hist_dollars = bundle["var_hist_dollars"]
cvar_hist_dollars = bundle["cvar_hist_dollars"]

# ---- Sidebar controls ----
st.sidebar.header("Frontier Settings")
frontier_points = st.sidebar.slider(
    "Frontier sample points",
    min_value=200,
    max_value=2000,
    value=1000,
    step=100,
)

# ---- Computation ----
annual_returns = annualized_return(log_returns_assets)   # pd.Series
cov_mat = covariance_matrix(log_returns_assets)          # pd.DataFrame

# Efficient frontier (random sampling, seeded for reproducibility)
np.random.seed(42)
frontier = efficient_frontier(annual_returns, cov_mat, points=frontier_points)
# frontier[:, 0] = vols, frontier[:, 1] = returns

# Sharpe ratio for coloring (assume rf=0)
sharpe = frontier[:, 1] / (frontier[:, 0] + 1e-10)

# Current portfolio stats
cur_ret = portfolio_expected_return(annual_returns, weights)
cur_vol = portfolio_volatility(cov_mat, weights)

# Min-vol portfolio
with st.spinner("Running optimization…"):
    opt_weights = minimize_volatility(cov_mat)

opt_ret = portfolio_expected_return(annual_returns, opt_weights)
opt_vol = portfolio_volatility(cov_mat, opt_weights)

# Risk metrics for min-vol portfolio
opt_daily_returns = portfolio_daily_returns(log_returns_assets, opt_weights)
opt_var = var_historical(opt_daily_returns, confidence, portfolio_value)
opt_cvar = cvar_historical(opt_daily_returns, confidence, portfolio_value)

# ---- Efficient frontier scatter ----
st.subheader("Efficient Frontier")

fig, ax = plt.subplots(figsize=(10, 6))

sc = ax.scatter(
    frontier[:, 0], frontier[:, 1],
    c=sharpe, cmap="viridis", alpha=0.6, s=8, label="Frontier portfolios"
)
plt.colorbar(sc, ax=ax, label="Sharpe ratio (rf=0)")

# Current portfolio
ax.scatter(cur_vol, cur_ret, marker="*", s=300, color="red", zorder=5, label="Current portfolio")

# Min-vol portfolio
ax.scatter(opt_vol, opt_ret, marker="*", s=300, color="green", zorder=5, label="Min-vol portfolio")

ax.set_xlabel("Annualized volatility")
ax.set_ylabel("Annualized expected return")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax.set_title("Portfolio Efficient Frontier")
ax.legend()
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

# ---- Portfolio comparison metrics ----
st.subheader("Current vs. Min-Vol Portfolio")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Expected Return",
    f"{opt_ret:.2%}",
    delta=f"{opt_ret - cur_ret:+.2%}",
    help="Annualized expected return of the min-vol portfolio vs current",
)
m2.metric(
    "Annualized Vol",
    f"{opt_vol:.2%}",
    delta=f"{opt_vol - cur_vol:+.2%}",
    delta_color="inverse",
    help="Lower is better for volatility",
)
m3.metric(
    f"VaR @ {int(confidence*100)}% (Historical)",
    f"${opt_var:,.0f}",
    delta=f"${opt_var - var_hist_dollars:+,.0f}",
    delta_color="inverse",
    help="1-day historical VaR in dollars (min-vol portfolio vs current)",
)
m4.metric(
    f"CVaR @ {int(confidence*100)}% (Historical)",
    f"${opt_cvar:,.0f}",
    delta=f"${opt_cvar - cvar_hist_dollars:+,.0f}",
    delta_color="inverse",
    help="1-day historical CVaR in dollars (min-vol portfolio vs current)",
)

st.caption("Metrics shown for **min-vol** portfolio. Delta = min-vol minus current.")

# ---- Weight comparison table & bar chart ----
st.subheader("Weight Comparison")

weight_df = pd.DataFrame(
    {"Current": weights, "Min-Vol": opt_weights},
    index=tickers,
)

st.bar_chart(weight_df)

# Also display as a formatted table
fmt_df = weight_df.copy()
fmt_df["Current"] = fmt_df["Current"].map("{:.2%}".format)
fmt_df["Min-Vol"] = fmt_df["Min-Vol"].map("{:.2%}".format)
fmt_df.index.name = "Ticker"
st.dataframe(fmt_df, use_container_width=True)
