from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.monte_carlo import simulate_gbm
from src.returns import annualized_return, annualized_volatility

st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")
st.title("Monte Carlo Simulation")

# ---- Guard: require session state ----
if "risk_app_results" not in st.session_state:
    st.warning("No portfolio data found. Please configure your portfolio on the **Homepage** first.")
    st.stop()

bundle = st.session_state["risk_app_results"]
portfolio_returns = bundle["portfolio_returns"]
portfolio_value = bundle["portfolio_value"]
confidence = bundle["confidence"]

# ---- Sidebar controls ----
st.sidebar.header("Simulation Settings")

n_simulations = st.sidebar.select_slider(
    "Number of simulations",
    options=[500, 1000, 5000],
    value=1000,
)

horizon_days = st.sidebar.selectbox(
    "Horizon",
    options=[21, 63, 126, 252],
    format_func=lambda d: {21: "1 month (21 days)", 63: "3 months (63 days)",
                            126: "6 months (126 days)", 252: "1 year (252 days)"}[d],
    index=1,
)

fix_seed = st.sidebar.checkbox("Fix random seed (reproducible)", value=True)
seed = 42 if fix_seed else None

# ---- Computation ----
mu = float(annualized_return(portfolio_returns))
sigma = float(annualized_volatility(portfolio_returns))
T = horizon_days / 252

with st.spinner("Running simulation…"):
    paths = simulate_gbm(
        S0=portfolio_value,
        mu=mu,
        sigma=sigma,
        T=T,
        steps=horizon_days,
        simulations=n_simulations,
        random_seed=seed,
    )

final_values = paths[-1]
mc_var = portfolio_value - np.percentile(final_values, (1 - confidence) * 100)
pct_above = float(np.mean(final_values > portfolio_value))

# ---- Summary metrics ----
st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean terminal value", f"${np.mean(final_values):,.0f}")
c2.metric("Median terminal value", f"${np.median(final_values):,.0f}")
c3.metric(f"MC VaR @ {int(confidence*100)}%", f"${mc_var:,.0f}")
c4.metric("Simulations ending above start", f"{pct_above:.1%}")

st.caption(
    f"GBM parameters — annualized drift μ: {mu:.4f}, annualized vol σ: {sigma:.4f}, "
    f"horizon: {horizon_days} trading days"
)

# ---- Fan chart ----
st.subheader("Simulated Portfolio Value Paths")

days = np.arange(paths.shape[0])
pct05 = np.percentile(paths, 5, axis=1)
pct50 = np.percentile(paths, 50, axis=1)
pct95 = np.percentile(paths, 95, axis=1)

fig, ax = plt.subplots(figsize=(10, 5))

# Plot first 200 paths in light gray
display_n = min(200, n_simulations)
ax.plot(days, paths[:, :display_n], color="gray", alpha=0.05, linewidth=0.5)

# Percentile overlays
ax.plot(days, pct05, color="red", linewidth=1.5, label="5th percentile")
ax.plot(days, pct50, color="blue", linewidth=1.5, label="50th percentile")
ax.plot(days, pct95, color="green", linewidth=1.5, label="95th percentile")

# Starting value reference
ax.axhline(portfolio_value, color="black", linestyle="--", linewidth=1.0, label=f"Start (${portfolio_value:,.0f})")

ax.set_xlabel("Trading days")
ax.set_ylabel("Portfolio value ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend()
ax.set_title(f"GBM Fan Chart — {n_simulations:,} simulations, {horizon_days}-day horizon")
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

# ---- Terminal value histogram ----
st.subheader("Distribution of Terminal Portfolio Values")

var_threshold = portfolio_value - mc_var  # the dollar level below which we count losses

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.hist(final_values, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
ax2.axvline(var_threshold, color="red", linewidth=2,
            label=f"MC VaR threshold (${var_threshold:,.0f})")
ax2.axvline(portfolio_value, color="black", linestyle="--", linewidth=1.5,
            label=f"Start value (${portfolio_value:,.0f})")
ax2.set_xlabel("Terminal portfolio value ($)")
ax2.set_ylabel("Count")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax2.legend()
ax2.set_title("Terminal value distribution")
fig2.tight_layout()
st.pyplot(fig2, clear_figure=True)
