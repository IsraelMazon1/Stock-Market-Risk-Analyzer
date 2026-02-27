import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import streamlit as st

from src.insights import generate_insights

st.set_page_config(page_title="Live Portfolio Insights", layout="wide")
st.title("Portfolio Insights")
st.caption("This page uses the latest portfolio configuration computed on the main page.")

bundle = st.session_state.get("risk_app_results")

if bundle is None:
    st.warning("No analysis found yet. Go to the main page, set your portfolio, and run the analysis first.")
    st.stop()

# Unpack
tickers = bundle["tickers"]
weights = bundle["weights"]
confidence = bundle["confidence"]
portfolio_value = bundle["portfolio_value"]
rolling_window = bundle["rolling_window"]

portfolio_returns = bundle["portfolio_returns"]
log_returns_assets = bundle["log_returns_assets"]

var_norm_dollars = bundle["var_norm_dollars"]
var_hist_dollars = bundle["var_hist_dollars"]
cvar_hist_dollars = bundle["cvar_hist_dollars"]

bt_norm = bundle["bt_norm"]
bt_hist = bundle["bt_hist"]
diag = bundle["diag"]

# Generate insights (rule-based)
insights, supporting = generate_insights(
    tickers=tickers,
    weights_raw=np.array(weights, dtype=float),
    portfolio_returns=portfolio_returns,
    log_returns_assets=log_returns_assets,
    confidence_level=confidence,
    var_dollars=var_norm_dollars,
    cvar_dollars=cvar_hist_dollars,
    backtest_norm=bt_norm,
    backtest_hist=bt_hist,
    diag=diag,
)

# Summary
st.subheader("Summary (from main page)")
c1, c2, c3 = st.columns(3)
c1.metric(f"VaR (Normal) @ {int(confidence*100)}%", f"${var_norm_dollars:,.2f}")
c2.metric(f"VaR (Historical) @ {int(confidence*100)}%", f"${var_hist_dollars:,.2f}")
c3.metric(f"CVaR (Historical) @ {int(confidence*100)}%", f"${cvar_hist_dollars:,.2f}")

st.subheader("Live Interpretation")

for ins in insights:
    if ins.severity == "high":
        st.error(f"**{ins.title}** — {ins.message}")
    elif ins.severity == "medium":
        st.warning(f"**{ins.title}** — {ins.message}")
    else:
        st.info(f"**{ins.title}** — {ins.message}")

st.subheader("Supporting Stats")
st.write({
    "HHI (concentration)": supporting["hhi"],
    "Avg pairwise correlation": supporting["avg_corr"],
    "CVaR/VaR ratio": supporting["cvar_var_ratio"],
    "Skew": supporting["skew"],
    "Excess kurtosis": supporting["excess_kurtosis"],
})

st.subheader("Risk Contribution Proxy (w * annualized vol)")
st.write(supporting["risk_contrib_proxy"])