import streamlit as st

st.set_page_config(page_title="Risk Interpretation", layout="wide")

st.title("Risk Model Interpretation & Statistical Insights")

st.header("What Does VaR Mean?")

st.markdown("""
**Value-at-Risk (VaR)** estimates the loss threshold not expected to be exceeded 
at a given confidence level over one day.

Example:  
A 95% VaR of $2,800 means there is a 5% chance the portfolio loses more than $2,800 in a day.
""")

st.header("Parametric vs Historical VaR")

st.markdown("""
**Parametric VaR** assumes returns follow a normal distribution.  
**Historical VaR** uses empirical quantiles from actual returns.

If these differ significantly, it suggests the normal assumption may not hold.
""")

st.header("Conditional VaR (Expected Shortfall)")

st.markdown("""
CVaR measures the **average loss in the worst 5% of days**.

It captures tail severity and is more robust than VaR 
because it accounts for extreme downside outcomes.
""")

st.header("Rolling Risk")

st.markdown("""
Rolling volatility and rolling VaR show that risk is **time-varying**.

Financial markets exhibit volatility clustering — 
periods of calm followed by turbulence.
""")

st.header("Backtesting VaR")

st.markdown("""
At 95% confidence, we expect ~5% of days to exceed VaR.

If observed violations are significantly higher:
- The model underestimates risk.

If significantly lower:
- The model overestimates risk.
""")

st.header("Distribution Diagnostics")

st.markdown("""
- **Skewness < 0** → heavier downside tail  
- **Excess kurtosis > 0** → fat tails  
- Fat tails imply normal VaR may underestimate risk.
""")

st.success("This dashboard demonstrates statistical risk modeling, time-series estimation, and model validation techniques used in real-world risk analytics.")