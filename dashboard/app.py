import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
import numpy as np

from src.config import DEFAULT_CONFIG
from src.data_loader import fetch_price_data
from src.returns import compute_log_returns
from src.portfolio import portfolio_daily_returns
from src.risk_metrics import var_parametric_normal



st.title("📊 Stock Risk Analyzer Dashboard")

tickers = st.text_input("Enter tickers (comma-separated):", "AAPL,MSFT,AMZN")
confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)

ticker_list = [t.strip().upper() for t in tickers.split(",")]

data = fetch_price_data(
    ticker_list,
    DEFAULT_CONFIG.start_date,
    price_field=DEFAULT_CONFIG.price_field,
    cache_dir=DEFAULT_CONFIG.data_raw_dir,
).prices

log_returns = compute_log_returns(data)

weights = np.ones(len(ticker_list)) / len(ticker_list)
portfolio_returns = portfolio_daily_returns(log_returns, weights)

var = var_parametric_normal(portfolio_returns, confidence, portfolio_value=1.0)

st.write(f"1-Day VaR ({int(confidence*100)}%): {var:.5f}")
st.line_chart(data)
